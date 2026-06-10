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

Two architectures:

- `ActionScorerAgent` — the original small MLP (Tanh, 256-wide), kept so
  existing checkpoints load.
- `ResAttnAgent` — the scaled-up network: pre-norm residual GELU trunk
  (512-wide by default) and self-attention *across the legal actions*, so
  actions are scored relative to each other instead of independently.

`agent_from_checkpoint` detects which architecture (and size) a state dict
was trained with and builds the matching module.
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

    @torch.no_grad()
    def act(self, obs, oracle_obs, act_feats, mask, greedy: bool = False):
        logits = self.policy_logits(obs, act_feats, mask).float()
        value = self.value(oracle_obs).float() if oracle_obs is not None else None
        if greedy:
            return logits.argmax(dim=-1), None, value
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


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


class ResAttnAgent(nn.Module):
    """Residual trunk + self-attention over the legal-action set.

    Each legal action becomes a token (its feature embedding plus a
    projection of the state), the tokens attend to each other (padding
    slots masked out as keys), and a per-token head emits the logit — so
    "retreat" can be scored knowing "attack for lethal" is also on the
    menu. Drop-in replacement for ActionScorerAgent: same forward/act/value
    interface, separate oracle-critic trunk.
    """

    def __init__(
        self,
        obs_dim: int,
        act_feat_dim: int,
        hidden: int = 512,
        act_hidden: int = 128,
        blocks: int = 4,
        heads: int = 4,
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
        self.critic_encoder = res_encoder(obs_dim, hidden, blocks)
        self.value_head = nn.Linear(hidden, 1)

    def policy_logits(
        self, obs: torch.Tensor, act_feats: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """obs: [B, obs_dim]; act_feats: [B, N, act_feat_dim]; mask: [B, N] bool."""
        state = self.obs_encoder(obs)
        tokens = self.act_encoder(act_feats) + self.state_proj(state).unsqueeze(1)
        normed = self.attn_ln(tokens)
        attended, _ = self.attn(normed, normed, normed, key_padding_mask=~mask)
        tokens = tokens + attended
        logits = self.scorer(tokens).squeeze(-1)
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

    @torch.no_grad()
    def act(self, obs, oracle_obs, act_feats, mask, greedy: bool = False):
        logits = self.policy_logits(obs, act_feats, mask).float()
        value = self.value(oracle_obs).float() if oracle_obs is not None else None
        if greedy:
            return logits.argmax(dim=-1), None, value
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


def make_agent(
    obs_dim: int,
    act_feat_dim: int,
    arch: str = "res",
    hidden: int | None = None,
    blocks: int = 4,
    heads: int = 4,
) -> nn.Module:
    if arch == "res":
        return ResAttnAgent(
            obs_dim, act_feat_dim, hidden=hidden or 512, blocks=blocks, heads=heads
        )
    if arch == "mlp":
        return ActionScorerAgent(obs_dim, act_feat_dim, hidden=hidden or 256)
    raise ValueError(f"unknown arch: {arch}")


def agent_from_state_dict(
    state_dict: dict[str, torch.Tensor], obs_dim: int, act_feat_dim: int
) -> nn.Module:
    """Build the architecture a state dict was trained with (size included)."""
    hidden = state_dict["obs_encoder.0.weight"].shape[0]
    if any(key.startswith("attn.") for key in state_dict):
        block_ids = {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("obs_encoder.") and key.endswith(".ln.weight")
        }
        heads = 4  # not recoverable from shapes; fixed across our runs
        agent = ResAttnAgent(
            obs_dim,
            act_feat_dim,
            hidden=hidden,
            act_hidden=state_dict["state_proj.weight"].shape[0],
            blocks=len(block_ids),
            heads=heads,
        )
    else:
        agent = ActionScorerAgent(
            obs_dim,
            act_feat_dim,
            hidden=hidden,
            act_hidden=state_dict["act_encoder.0.weight"].shape[0],
        )
    agent.load_state_dict(state_dict)
    return agent
