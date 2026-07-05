"""Thin numpy/torch-friendly wrapper around deckgym.PyRlVecEnv.

The Rust env returns flat arrays; this wrapper reshapes them to
(num_envs, obs_dim), (num_envs, max_actions, act_feat_dim) and builds the
boolean legality mask from the per-env legal-action counts.

Two observation views are returned: `obs` hides the opponent's hand and deck
composition (what a real player sees) and feeds the policy; `oracle_obs` is
the full state and feeds the critic during training (PerfectDou-style
perfect-training-imperfect-execution).
"""

import numpy as np
from deckgym import PyRlVecEnv


class VecEnv:
    def __init__(
        self,
        deck_a: str,
        deck_b: str,
        opponent: str = "r",
        num_envs: int = 32,
        seed: int = 0,
        shaping_coef: float = 0.0,
    ):
        self._env = PyRlVecEnv(deck_a, deck_b, opponent, num_envs, seed, shaping_coef)
        self.num_envs = num_envs
        self.obs_dim = self._env.obs_dim()
        self.act_feat_dim = self._env.action_feat_dim()
        self.max_actions = self._env.max_actions()

    def _unpack(self, obs, oracle_obs, feats, n_actions):
        obs = obs.reshape(self.num_envs, self.obs_dim)
        oracle_obs = oracle_obs.reshape(self.num_envs, self.obs_dim)
        feats = feats.reshape(self.num_envs, self.max_actions, self.act_feat_dim)
        mask = np.arange(self.max_actions)[None, :] < n_actions[:, None]
        return obs, oracle_obs, feats, mask

    def reset(self):
        return self._unpack(*self._env.reset())

    def step(self, actions: np.ndarray):
        obs, oracle_obs, feats, n_actions, rewards, dones, outcomes = self._env.step(
            [int(a) for a in actions]
        )
        return (*self._unpack(obs, oracle_obs, feats, n_actions), rewards, dones, outcomes)

    def legal_action_strings(self, env_idx: int = 0):
        return self._env.legal_action_strings(env_idx)
