"""Policy/value networks that score the legal-action list.

The policy embeds the (hidden-information) observation and each legal
action's feature vector, then emits one logit per action slot; illegal
(padding) slots are masked to -inf. This is the "legal-action scorer"
pattern used by ygo-agent and DouZero, which avoids any global action
enumeration.

The value head is an *oracle critic*: it reads the full-state observation
(opponent hand/deck included), which is only available at training time.
This is the perfect-training-imperfect-execution trick from PerfectDou —
the critic only shapes gradients, so the deployed policy stays honest.

Architectures:

- `ActionScorerAgent` — the original small MLP (Tanh, 256-wide), kept so
  existing checkpoints load.
- `ResAttnAgent` — the scaled-up network: pre-norm residual GELU trunk
  (512-wide by default) and self-attention *across the legal actions*, so
  actions are scored relative to each other instead of independently.
- `TokenTransformerAgent` — structured encoder: the flat observation is cut
  back into its semantic sections (globals, 8 board slots, 6 zone count
  vectors), each projected to a token, and a TransformerEncoder reasons over
  the token set before the same action-attention scorer.

Both new architectures accept `memory=True`, which inserts a GRU cell over
*decision steps* on the policy path — the agent can integrate what it has
seen across a game (cards drawn, opponent plays) instead of being purely
reactive. The oracle critic stays feedforward: it sees the full state, which
is (near-)Markov, so recurrence buys it nothing and would complicate GAE.

The uniform interface is `act(obs, oracle, feats, mask, h, greedy)` →
`(action, logprob, value, h_new)` with `h_new = None` for memoryless nets,
plus `initial_state(batch, device)` and, for training-time BPTT,
`policy_logits_seq`. `agent_from_state_dict` detects which architecture,
size, and memory setting a state dict was trained with and builds the
matching module.
"""

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for size in hidden:
        layers.append(nn.Linear(last, size))
        layers.append(nn.Tanh())
        last = size
    return nn.Sequential(*layers)


class ActionScorerAgent(nn.Module):
    def __init__(self, obs_dim: int, act_feat_dim: int, hidden: int = 256, act_hidden: int = 64):
        super().__init__()
        self.obs_encoder = mlp(obs_dim, [hidden, hidden])
        self.act_encoder = mlp(act_feat_dim, [act_hidden])
        self.scorer = nn.Sequential(
            nn.Linear(hidden + act_hidden, act_hidden),
            nn.Tanh(),
            nn.Linear(act_hidden, 1),
        )
        self.critic_encoder = mlp(obs_dim, [hidden, hidden])
        self.value_head = nn.Linear(hidden, 1)

    def policy_logits(
        self, obs: torch.Tensor, act_feats: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """obs: [B, obs_dim]; act_feats: [B, N, act_feat_dim]; mask: [B, N] bool."""
        state = self.obs_encoder(obs)
        acts = self.act_encoder(act_feats)
        expanded = state.unsqueeze(1).expand(-1, acts.shape[1], -1)
        logits = self.scorer(torch.cat([expanded, acts], dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def value(self, oracle_obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_encoder(oracle_obs)).squeeze(-1)

    def forward(
        self,
        obs: torch.Tensor,
        oracle_obs: torch.Tensor,
        act_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_logits(obs, act_feats, mask), self.value(oracle_obs)

    def initial_state(self, batch: int, device) -> torch.Tensor | None:
        return None

    @torch.no_grad()
    def act(self, obs, oracle_obs, act_feats, mask, h=None, greedy: bool = False):
        logits = self.policy_logits(obs, act_feats, mask).float()
        value = self.value(oracle_obs).float() if oracle_obs is not None else None
        if greedy:
            return logits.argmax(dim=-1), None, value, None
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, None


class ResBlock(nn.Module):
    """Pre-norm residual MLP block: x + W2(GELU(W1(LN(x))))."""

    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.ln(x))


def res_encoder(in_dim: int, hidden: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        *[ResBlock(hidden) for _ in range(blocks)],
        nn.LayerNorm(hidden),
    )


class _AttnScorerBase(nn.Module):
    """Shared policy/critic machinery for the larger architectures.

    Subclasses provide `obs_encoder`/`critic_encoder` (obs → [B, hidden]),
    plus the action-attention modules and (optionally) `self.gru`. Methods
    only — no parameters — so each subclass keeps a flat, stable state-dict
    key layout.
    """

    gru: nn.GRUCell | None

    def _score_actions(
        self, state: torch.Tensor, act_feats: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.act_encoder(act_feats) + self.state_proj(state).unsqueeze(1)
        normed = self.attn_ln(tokens)
        attended, _ = self.attn(normed, normed, normed, key_padding_mask=~mask)
        tokens = tokens + attended
        logits = self.scorer(tokens).squeeze(-1)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def initial_state(self, batch: int, device) -> torch.Tensor | None:
        if self.gru is None:
            return None
        return torch.zeros(batch, self.gru.hidden_size, device=device)

    def policy_logits(
        self,
        obs: torch.Tensor,
        act_feats: torch.Tensor,
        mask: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """obs: [B, obs_dim]; act_feats: [B, N, act_feat_dim]; mask: [B, N] bool."""
        state = self.obs_encoder(obs)
        if self.gru is not None:
            h = self.gru(state, h)
            state = h
        return self._score_actions(state, act_feats, mask), h if self.gru is not None else None

    def policy_logits_seq(
        self,
        obs: torch.Tensor,
        act_feats: torch.Tensor,
        mask: torch.Tensor,
        h0: torch.Tensor | None,
        resets: torch.Tensor,
    ) -> torch.Tensor:
        """BPTT replay of a rollout: obs [T, B, ...], resets [T, B] (1 marks
        the first decision of a fresh episode, where memory must be zeroed).
        Returns flat logits [T*B, N]."""
        steps, batch = obs.shape[0], obs.shape[1]
        flat_state = self.obs_encoder(obs.reshape(steps * batch, -1))
        if self.gru is None:
            states = flat_state
        else:
            state_seq = flat_state.reshape(steps, batch, -1)
            h = h0
            hs = []
            for t in range(steps):
                h = h * (1.0 - resets[t]).unsqueeze(-1)
                h = self.gru(state_seq[t], h)
                hs.append(h)
            states = torch.stack(hs).reshape(steps * batch, -1)
        return self._score_actions(
            states,
            act_feats.reshape(steps * batch, *act_feats.shape[2:]),
            mask.reshape(steps * batch, -1),
        )

    def value(self, oracle_obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_encoder(oracle_obs)).squeeze(-1)

    def forward(
        self,
        obs: torch.Tensor,
        oracle_obs: torch.Tensor,
        act_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.gru is None, "recurrent agents must use policy_logits_seq for updates"
        logits, _ = self.policy_logits(obs, act_feats, mask)
        return logits, self.value(oracle_obs)

    @torch.no_grad()
    def act(self, obs, oracle_obs, act_feats, mask, h=None, greedy: bool = False):
        logits, h_new = self.policy_logits(obs, act_feats, mask, h)
        logits = logits.float()
        value = self.value(oracle_obs).float() if oracle_obs is not None else None
        if greedy:
            return logits.argmax(dim=-1), None, value, h_new
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, h_new


class ResAttnAgent(_AttnScorerBase):
    """Residual trunk + self-attention over the legal-action set.

    Each legal action becomes a token (its feature embedding plus a
    projection of the state), the tokens attend to each other (padding
    slots masked out as keys), and a per-token head emits the logit — so
    "retreat" can be scored knowing "attack for lethal" is also on the
    menu. With `memory=True` a GRU cell over decision steps sits between
    the trunk and the heads (policy path only; the oracle critic is
    feedforward because the full state is near-Markov).
    """

    def __init__(
        self,
        obs_dim: int,
        act_feat_dim: int,
        hidden: int = 512,
        act_hidden: int = 128,
        blocks: int = 4,
        heads: int = 4,
        memory: bool = False,
    ):
        super().__init__()
        self.obs_encoder = res_encoder(obs_dim, hidden, blocks)
        self.act_encoder = res_encoder(act_feat_dim, act_hidden, 1)
        self.state_proj = nn.Linear(hidden, act_hidden)
        self.attn_ln = nn.LayerNorm(act_hidden)
        self.attn = nn.MultiheadAttention(act_hidden, heads, batch_first=True)
        self.scorer = nn.Sequential(
            nn.LayerNorm(act_hidden),
            nn.Linear(act_hidden, act_hidden),
            nn.GELU(),
            nn.Linear(act_hidden, 1),
        )
        self.gru = nn.GRUCell(hidden, hidden) if memory else None
        self.critic_encoder = res_encoder(obs_dim, hidden, blocks)
        self.value_head = nn.Linear(hidden, 1)


class TokenEncoder(nn.Module):
    """Cuts the flat observation back into semantic tokens and runs a
    transformer over them: 1 globals token, 8 board-slot tokens (shared
    projection + position embedding), 6 zone-count tokens (hands/discards/
    decks for both sides; shared projection + zone embedding). Output is
    the transformed globals token."""

    NUM_SLOTS = 8
    NUM_ZONES = 6
    GLOBALS = 58

    def __init__(self, obs_dim: int, d_model: int, layers: int, heads: int):
        super().__init__()
        # obs_dim = 58 + 8*(23+V) + 6*V  (see RlEnvCore::obs_dim)
        vocab = (obs_dim - self.GLOBALS - 8 * 23) // 14
        self.slot_dim = 23 + vocab
        self.vocab = vocab
        self.global_proj = nn.Linear(self.GLOBALS, d_model)
        self.slot_proj = nn.Linear(self.slot_dim, d_model)
        self.zone_proj = nn.Linear(vocab, d_model)
        self.slot_pos = nn.Parameter(torch.zeros(self.NUM_SLOTS, d_model))
        self.zone_pos = nn.Parameter(torch.zeros(self.NUM_ZONES, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=2 * d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        globals_tok = self.global_proj(obs[:, : self.GLOBALS]).unsqueeze(1)
        slots_flat = obs[:, self.GLOBALS : self.GLOBALS + self.NUM_SLOTS * self.slot_dim]
        slots = self.slot_proj(
            slots_flat.reshape(batch, self.NUM_SLOTS, self.slot_dim)
        ) + self.slot_pos.unsqueeze(0)
        zones_flat = obs[:, self.GLOBALS + self.NUM_SLOTS * self.slot_dim :]
        zones = self.zone_proj(
            zones_flat.reshape(batch, self.NUM_ZONES, self.vocab)
        ) + self.zone_pos.unsqueeze(0)
        tokens = torch.cat([globals_tok, slots, zones], dim=1)
        return self.out_ln(self.encoder(tokens)[:, 0])


class TokenTransformerAgent(_AttnScorerBase):
    """Transformer over structured observation tokens + action attention."""

    def __init__(
        self,
        obs_dim: int,
        act_feat_dim: int,
        hidden: int = 256,
        act_hidden: int = 128,
        blocks: int = 3,
        heads: int = 4,
        memory: bool = False,
    ):
        super().__init__()
        self.obs_encoder = TokenEncoder(obs_dim, hidden, blocks, heads)
        self.act_encoder = res_encoder(act_feat_dim, act_hidden, 1)
        self.state_proj = nn.Linear(hidden, act_hidden)
        self.attn_ln = nn.LayerNorm(act_hidden)
        self.attn = nn.MultiheadAttention(act_hidden, heads, batch_first=True)
        self.scorer = nn.Sequential(
            nn.LayerNorm(act_hidden),
            nn.Linear(act_hidden, act_hidden),
            nn.GELU(),
            nn.Linear(act_hidden, 1),
        )
        self.gru = nn.GRUCell(hidden, hidden) if memory else None
        self.critic_encoder = TokenEncoder(obs_dim, hidden, blocks, heads)
        self.value_head = nn.Linear(hidden, 1)


def make_agent(
    obs_dim: int,
    act_feat_dim: int,
    arch: str = "res",
    hidden: int | None = None,
    blocks: int | None = None,
    heads: int = 4,
    memory: bool = False,
) -> nn.Module:
    if arch == "res":
        return ResAttnAgent(
            obs_dim,
            act_feat_dim,
            hidden=hidden or 512,
            blocks=blocks if blocks is not None else 4,
            heads=heads,
            memory=memory,
        )
    if arch == "tx":
        return TokenTransformerAgent(
            obs_dim,
            act_feat_dim,
            hidden=hidden or 256,
            blocks=blocks if blocks is not None else 3,
            heads=heads,
            memory=memory,
        )
    if arch == "mlp":
        if memory:
            raise ValueError("--memory requires --arch res or tx")
        return ActionScorerAgent(obs_dim, act_feat_dim, hidden=hidden or 256)
    raise ValueError(f"unknown arch: {arch}")


def agent_from_state_dict(
    state_dict: dict[str, torch.Tensor], obs_dim: int, act_feat_dim: int
) -> nn.Module:
    """Build the architecture a state dict was trained with (size included)."""
    memory = "gru.weight_ih" in state_dict
    heads = 4  # not recoverable from shapes; fixed across our runs
    if "obs_encoder.global_proj.weight" in state_dict:
        layer_ids = {
            int(key.split(".")[3])
            for key in state_dict
            if key.startswith("obs_encoder.encoder.layers.")
        }
        agent = TokenTransformerAgent(
            obs_dim,
            act_feat_dim,
            hidden=state_dict["obs_encoder.global_proj.weight"].shape[0],
            act_hidden=state_dict["state_proj.weight"].shape[0],
            blocks=len(layer_ids),
            heads=heads,
            memory=memory,
        )
    elif any(key.startswith("attn.") for key in state_dict):
        block_ids = {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("obs_encoder.") and key.endswith(".ln.weight")
        }
        agent = ResAttnAgent(
            obs_dim,
            act_feat_dim,
            hidden=state_dict["obs_encoder.0.weight"].shape[0],
            act_hidden=state_dict["state_proj.weight"].shape[0],
            blocks=len(block_ids),
            heads=heads,
            memory=memory,
        )
    else:
        agent = ActionScorerAgent(
            obs_dim,
            act_feat_dim,
            hidden=state_dict["obs_encoder.0.weight"].shape[0],
            act_hidden=state_dict["act_encoder.0.weight"].shape[0],
        )
    agent.load_state_dict(state_dict)
    return agent
