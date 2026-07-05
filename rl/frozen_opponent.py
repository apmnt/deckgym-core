"""Wrap an arbitrary checkpoint as a self-play roster opponent.

`SelfPlayVecEnv` drives seat 1 with any module exposing `act`, `config`,
`gru`, and `initial_state` — normally frozen copies of the learner. This
shim lets a *different* architecture join the roster, most importantly the
legacy mirror-match champion: legacy checkpoints are auto-detected and
their observations/action features are sliced through `LegacyObsAdapter`
(see head_to_head.py) before the wrapped net sees them.

Training directly against a strong frozen opponent is the exploiter
pattern (AlphaStar main exploiters; "Learning to Beat ByteRL"): a
stationary policy is a far easier target than a search bot, and PFSP will
focus on it exactly while it still beats the learner.
"""

import torch
import torch.nn as nn

from head_to_head import load_agent


class FrozenOpponent(nn.Module):
    def __init__(self, agent: nn.Module, adapter=None, force_greedy: bool = True):
        super().__init__()
        self.agent = agent
        self.adapter = adapter
        # Exploit the *deployment* policy: checkpoints are evaluated greedy,
        # so the roster arm plays greedy regardless of the caller's default
        # (the self-play env samples frozen nets for diversity).
        self.force_greedy = force_greedy
        self.config = {"oracle": agent.config.get("oracle", False)}
        self.gru = None  # adapter path supports stateless opponents only

    def initial_state(self, batch: int, device):
        return None

    @torch.no_grad()
    def act(self, obs, oracle_obs, act_feats, mask, h=None, greedy: bool = False):
        if self.adapter is not None:
            obs = obs[:, self.adapter.obs_idx]
            act_feats = act_feats[:, :, self.adapter.act_idx]
        action, logprob, value, _ = self.agent.act(
            obs, oracle_obs, act_feats, mask, None, greedy or self.force_greedy
        )
        return action, logprob, value, None


def load_frozen(path: str, probe_obs_dim: int, probe_feat_dim: int, device) -> FrozenOpponent:
    agent, adapter = load_agent(path, probe_obs_dim, probe_feat_dim, device)
    if adapter is not None and getattr(agent, "gru", None) is not None:
        raise SystemExit(f"{path}: recurrent legacy opponents are not supported")
    return FrozenOpponent(agent, adapter).to(device).eval()
