"""Policy/value network that scores the legal-action list.

The policy embeds the (hidden-information) observation and each legal
action's feature vector, then emits one logit per action slot; illegal
(padding) slots are masked to -inf. This is the "legal-action scorer"
pattern used by ygo-agent and DouZero, which avoids any global action
enumeration.

The value head is an *oracle critic*: it reads the full-state observation
(opponent hand/deck included), which is only available at training time.
This is the perfect-training-imperfect-execution trick from PerfectDou —
the critic only shapes gradients, so the deployed policy stays honest.
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
