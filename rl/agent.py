"""Policy/value network that scores the legal-action list.

The network embeds the observation and each legal action's feature vector,
then emits one logit per action slot; illegal (padding) slots are masked to
-inf. This is the "legal-action scorer" pattern used by ygo-agent and
DouZero, which avoids any global action enumeration.
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
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self, obs: torch.Tensor, act_feats: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: [B, obs_dim]; act_feats: [B, N, act_feat_dim]; mask: [B, N] bool.

        Returns (logits [B, N] with -inf at masked slots, value [B]).
        """
        state = self.obs_encoder(obs)
        acts = self.act_encoder(act_feats)
        expanded = state.unsqueeze(1).expand(-1, acts.shape[1], -1)
        logits = self.scorer(torch.cat([expanded, acts], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, self.value_head(state).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, act_feats, mask, greedy: bool = False):
        logits, value = self(obs, act_feats, mask)
        if greedy:
            action = logits.argmax(dim=-1)
            return action, None, value
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value
