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
        self.config = {
            "arch": "mlp",
            "hidden": hidden,
            "act_hidden": act_hidden,
        }
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

    def aux_belief_loss(self, obs: torch.Tensor, oracle_obs: torch.Tensor) -> torch.Tensor:
        """Auxiliary hidden-state inference: predict the opponent's hand
        composition (a section of the oracle observation) from the policy
        trunk. The gradient shapes the trunk toward belief-tracking features
        even though the prediction itself is never used at play time."""
        assert self.belief_head is not None, "agent was built without a belief head"
        pred = self.belief_head(self.obs_encoder(obs))
        vocab = pred.shape[-1]
        # Oracle layout: 58 globals, 8 slots of (23+V), then card-count
        # sections [my hand, their hand, ...] of V each (see rl_env.rs).
        offset = 58 + 8 * (23 + vocab) + vocab
        target = oracle_obs[..., offset : offset + vocab]
        return nn.functional.mse_loss(pred, target)

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


class CardEncoder(nn.Module):
    """Per-card representation shared by every observation/action section:
    a learned identity embedding over the engine's *global* card index plus
    a projection of the card's static attributes (HP, type, stage, attack
    damage/costs, trainer kind, ...). Identity captures what training saw;
    attributes give semantics that generalize to rarely-seen cards
    (ygo-agent / Cardsformer pattern). Index `num_cards` is the padding id
    (zero embedding, zero attribute row)."""

    def __init__(
        self,
        num_cards: int,
        attr_dim: int,
        d_card: int,
        attr_table=None,
        fusion: str = "sum",
    ):
        super().__init__()
        self.emb = nn.Embedding(num_cards + 1, d_card, padding_idx=num_cards)
        self.attr_proj = nn.Linear(attr_dim, d_card)
        self.fusion = fusion
        if attr_table is None:
            attr_table = torch.zeros(num_cards + 1, attr_dim)
        assert attr_table.shape == (num_cards + 1, attr_dim)
        # Buffer: saved in checkpoints, so a checkpoint is self-contained.
        self.register_buffer("attr_table", attr_table.float())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [..., V] long → [..., V, d_card]."""
        attrs = self.attr_proj(self.attr_table[ids])
        if self.fusion == "mul":
            # ygo-agent style: the attribute projection gates the identity
            # embedding (features MLP output multiplied by the id embedding).
            return self.emb(ids) * nn.functional.silu(attrs)
        return self.emb(ids) + attrs


class GeneralAgent(_AttnScorerBase):
    """Deck-general agent: plays any deck against any deck.

    The env's observation carries, per match, a card vocabulary of up to
    `vocab` slots: board one-hots and zone count vectors are expressed over
    those local slots, and the final `vocab` obs entries hold each slot's
    *global* card id. This agent looks the ids up in a `CardEncoder` and
    replaces every local one-hot/count section with its embedding-space
    projection (counts @ card_repr), so the trunk input has the same
    meaning whatever decks are on the table. Action features are re-embedded
    the same way. Downstream (residual trunk, self-attention over the
    legal-action set, oracle critic, optional GRU memory) matches
    `ResAttnAgent`.
    """

    GLOBALS = 58
    NUM_SLOTS = 8
    SLOT_NUM = 23  # occupied flag + numeric slot features
    NUM_ZONES = 6
    ACT_CLASSES = 18

    def __init__(
        self,
        obs_dim: int,
        act_feat_dim: int,
        hidden: int = 384,
        act_hidden: int = 128,
        blocks: int = 3,
        heads: int = 4,
        memory: bool = False,
        num_cards: int = 0,
        attr_dim: int = 0,
        d_card: int = 64,
        vocab: int = 40,
        attr_table=None,
        fusion: str = "sum",
    ):
        super().__init__()
        assert num_cards > 0 and attr_dim > 0, "gen arch needs the global card table"
        expected = (
            self.GLOBALS
            + self.NUM_SLOTS * (self.SLOT_NUM + vocab)
            + (self.NUM_ZONES + 1) * vocab
        )
        assert obs_dim == expected, f"obs_dim {obs_dim} != expected {expected} (vocab={vocab})"
        self.config = {
            "arch": "gen",
            "hidden": hidden,
            "act_hidden": act_hidden,
            "blocks": blocks,
            "heads": heads,
            "memory": memory,
            "num_cards": num_cards,
            "attr_dim": attr_dim,
            "d_card": d_card,
            "vocab": vocab,
            "fusion": fusion,
        }
        self.vocab = vocab
        self.d_card = d_card
        self.cards = CardEncoder(num_cards, attr_dim, d_card, attr_table, fusion)
        trunk_in = (
            self.GLOBALS + self.NUM_SLOTS * (self.SLOT_NUM + d_card) + self.NUM_ZONES * d_card
        )
        act_in = act_feat_dim - vocab + d_card
        self.belief_head = None
        self.obs_encoder = res_encoder(trunk_in, hidden, blocks)
        self.act_encoder = res_encoder(act_in, act_hidden, 1)
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
        self.critic_encoder = res_encoder(trunk_in, hidden, blocks)
        self.value_head = nn.Linear(hidden, 1)

    def _card_repr(self, obs: torch.Tensor) -> torch.Tensor:
        """[B, obs_dim] → [B, V, d_card] from the trailing vocab-id section."""
        ids = obs[:, -self.vocab :].long()
        return self.cards(ids)

    def _flatten_obs(self, obs: torch.Tensor, card_repr: torch.Tensor) -> torch.Tensor:
        """Project the local one-hot/count sections into embedding space."""
        batch = obs.shape[0]
        v = self.vocab
        globals_sec = obs[:, : self.GLOBALS]
        slots = obs[:, self.GLOBALS : self.GLOBALS + self.NUM_SLOTS * (self.SLOT_NUM + v)]
        # Slot layout (see rl_env.rs): [occupied(1) | card one-hot(v) | numeric(22)]
        slots = slots.reshape(batch, self.NUM_SLOTS, self.SLOT_NUM + v)
        slot_cards = torch.bmm(slots[:, :, 1 : 1 + v], card_repr)
        slot_rest = torch.cat([slots[:, :, :1], slots[:, :, 1 + v :]], dim=-1)
        zones_start = self.GLOBALS + self.NUM_SLOTS * (self.SLOT_NUM + v)
        zones = obs[:, zones_start : zones_start + self.NUM_ZONES * v]
        zone_cards = torch.bmm(zones.reshape(batch, self.NUM_ZONES, v), card_repr)
        return torch.cat(
            [
                globals_sec,
                slot_rest.reshape(batch, -1),
                slot_cards.reshape(batch, -1),
                zone_cards.reshape(batch, -1),
            ],
            dim=-1,
        )

    def _embed_actions(self, act_feats: torch.Tensor, card_repr: torch.Tensor) -> torch.Tensor:
        """Replace the action card one-hot with its card embedding."""
        v = self.vocab
        onehot = act_feats[:, :, self.ACT_CLASSES : self.ACT_CLASSES + v]
        card = torch.bmm(onehot, card_repr)
        rest = torch.cat(
            [act_feats[:, :, : self.ACT_CLASSES], act_feats[:, :, self.ACT_CLASSES + v :]],
            dim=-1,
        )
        return torch.cat([rest, card], dim=-1)

    def policy_logits(
        self,
        obs: torch.Tensor,
        act_feats: torch.Tensor,
        mask: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        card_repr = self._card_repr(obs)
        state = self.obs_encoder(self._flatten_obs(obs, card_repr))
        if self.gru is not None:
            h = self.gru(state, h)
            state = h
        emb_feats = self._embed_actions(act_feats, card_repr)
        logits = self._score_actions(state, emb_feats, mask)
        return logits, h if self.gru is not None else None

    def policy_logits_seq(
        self,
        obs: torch.Tensor,
        act_feats: torch.Tensor,
        mask: torch.Tensor,
        h0: torch.Tensor | None,
        resets: torch.Tensor,
    ) -> torch.Tensor:
        steps, batch = obs.shape[0], obs.shape[1]
        flat_obs = obs.reshape(steps * batch, -1)
        card_repr = self._card_repr(flat_obs)
        flat_state = self.obs_encoder(self._flatten_obs(flat_obs, card_repr))
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
        emb_feats = self._embed_actions(
            act_feats.reshape(steps * batch, *act_feats.shape[2:]), card_repr
        )
        return self._score_actions(states, emb_feats, mask.reshape(steps * batch, -1))

    def value(self, oracle_obs: torch.Tensor) -> torch.Tensor:
        card_repr = self._card_repr(oracle_obs)
        state = self.critic_encoder(self._flatten_obs(oracle_obs, card_repr))
        return self.value_head(state).squeeze(-1)

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
        belief: bool = False,
    ):
        super().__init__()
        self.config = {
            "arch": "res",
            "hidden": hidden,
            "act_hidden": act_hidden,
            "blocks": blocks,
            "heads": heads,
            "memory": memory,
            "belief": belief,
        }
        vocab = (obs_dim - 242) // 14
        self.belief_head = nn.Linear(hidden, vocab) if belief else None
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
        belief: bool = False,
    ):
        super().__init__()
        self.config = {
            "arch": "tx",
            "hidden": hidden,
            "act_hidden": act_hidden,
            "blocks": blocks,
            "heads": heads,
            "memory": memory,
            "belief": belief,
        }
        vocab = (obs_dim - 242) // 14
        self.belief_head = nn.Linear(hidden, vocab) if belief else None
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
    belief: bool = False,
    oracle: bool = False,
    card_table: tuple | None = None,
    fusion: str = "sum",
) -> nn.Module:
    agent = _build_agent(
        obs_dim, act_feat_dim, arch, hidden, blocks, heads, memory, belief, card_table, fusion
    )
    # Oracle agents feed the full-state view to the *policy* (an all-knowing
    # player, like the engine bots). The flag lives in the config so eval and
    # self-play route the right observation automatically.
    agent.config["oracle"] = oracle
    return agent


def fetch_card_table(text_features: str | None = None) -> tuple[torch.Tensor, int]:
    """Global card attribute table from the engine: (attr_table, num_cards).
    The table has num_cards + 1 rows (last = padding id). With
    `text_features` (a .npy produced by rl/build_text_features.py), the
    text-derived vectors are concatenated onto the numeric attributes,
    giving the CardEncoder wording-level semantics (heal/search/status
    effects cluster together) on top of stats."""
    import deckgym

    flat, attr_dim = deckgym.card_attr_table()
    num_cards = deckgym.num_global_cards()
    table = torch.from_numpy(flat.reshape(num_cards + 1, attr_dim))
    if text_features:
        import numpy as np

        text = torch.from_numpy(np.load(text_features).astype("float32"))
        assert text.shape[0] == num_cards + 1, (
            f"text feature rows {text.shape[0]} != {num_cards + 1}; "
            "regenerate with rl/build_text_features.py"
        )
        table = torch.cat([table, text], dim=1)
    return table, num_cards


def _build_agent(
    obs_dim: int,
    act_feat_dim: int,
    arch: str,
    hidden: int | None,
    blocks: int | None,
    heads: int,
    memory: bool,
    belief: bool,
    card_table: tuple | None = None,
    fusion: str = "sum",
) -> nn.Module:
    if arch == "gen":
        if belief:
            raise ValueError("--aux-belief is not supported with --arch gen")
        attr_table, num_cards = card_table if card_table is not None else fetch_card_table()
        return GeneralAgent(
            obs_dim,
            act_feat_dim,
            hidden=hidden or 384,
            blocks=blocks if blocks is not None else 3,
            heads=heads,
            memory=memory,
            num_cards=num_cards,
            attr_dim=attr_table.shape[1],
            attr_table=attr_table,
            fusion=fusion,
        )
    if arch == "res":
        return ResAttnAgent(
            obs_dim,
            act_feat_dim,
            hidden=hidden or 512,
            blocks=blocks if blocks is not None else 4,
            heads=heads,
            memory=memory,
            belief=belief,
        )
    if arch == "tx":
        return TokenTransformerAgent(
            obs_dim,
            act_feat_dim,
            hidden=hidden or 256,
            blocks=blocks if blocks is not None else 3,
            heads=heads,
            memory=memory,
            belief=belief,
        )
    if arch == "mlp":
        if memory:
            raise ValueError("--memory requires --arch res or tx")
        return ActionScorerAgent(obs_dim, act_feat_dim, hidden=hidden or 256)
    raise ValueError(f"unknown arch: {arch}")


_ARCH_CLASSES = {
    "mlp": ActionScorerAgent,
    "res": ResAttnAgent,
    "tx": TokenTransformerAgent,
    "gen": GeneralAgent,
}


def agent_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    obs_dim: int,
    act_feat_dim: int,
    config: dict | None = None,
) -> nn.Module:
    """Build the architecture a state dict was trained with (size included).

    Modern checkpoints carry an explicit `config`; for legacy raw state
    dicts the architecture is inferred from tensor shapes, which cannot
    recover the attention head count (assumed 4 — the historical default).
    """
    if config is not None:
        cls = _ARCH_CLASSES[config["arch"]]
        kwargs = {k: v for k, v in config.items() if k not in ("arch", "oracle")}
        agent = cls(obs_dim, act_feat_dim, **kwargs)
        agent.config["oracle"] = config.get("oracle", False)
        agent.load_state_dict(state_dict)
        return agent
    memory = "gru.weight_ih" in state_dict
    belief = "belief_head.weight" in state_dict
    heads = 4  # legacy checkpoints: not recoverable from shapes
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
            belief=belief,
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
            belief=belief,
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
