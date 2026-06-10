"""Self-play vec-env: learner in seat 0, frozen policies in seat 1.

Wraps `PyRlVecEnv(opponent="self")`, which pauses at *both* seats'
decisions. This wrapper drives seat 1 with frozen opponent networks — a
copy of the latest learner weights or a snapshot from a historical pool,
sampled per episode — so the PPO loop sees the exact single-agent interface
of `VecEnv`: every `step(actions)` consumes seat-0 actions for all envs and
returns the next seat-0 decision, with rewards/dones accumulated across any
interleaved opponent moves.

Opponents play honestly: they get the hidden-information observation from
their own perspective and sample (rather than argmax) for diversity.
"""

import copy

import numpy as np
import torch
from deckgym import PyRlVecEnv

LATEST = -1  # assignment key for "frozen copy of current learner"


class SelfPlayVecEnv:
    def __init__(
        self,
        deck_a: str,
        deck_b: str,
        num_envs: int = 32,
        seed: int = 0,
        shaping_coef: float = 0.0,
        latest_prob: float = 0.5,
    ):
        self._env = PyRlVecEnv(deck_a, deck_b, "self", num_envs, seed, shaping_coef)
        self.num_envs = num_envs
        self.obs_dim = self._env.obs_dim()
        self.act_feat_dim = self._env.action_feat_dim()
        self.max_actions = self._env.max_actions()
        self.latest_prob = latest_prob
        self._latest: torch.nn.Module | None = None
        self._pool: list[torch.nn.Module] = []
        self._rng = np.random.default_rng(seed)
        self._assignments = np.full(num_envs, LATEST, dtype=np.int64)
        self._pending_reward = np.zeros(num_envs, dtype=np.float32)
        self._pending_done = np.zeros(num_envs, dtype=bool)
        self._pending_outcome = np.zeros(num_envs, dtype=np.int8)

    def set_latest(self, agent: torch.nn.Module):
        """Refresh the frozen copy of the learner used as the 'latest' opponent."""
        self._latest = copy.deepcopy(agent).eval()

    def add_snapshot(self, agent: torch.nn.Module, max_pool: int):
        self._pool.append(copy.deepcopy(agent).eval())
        if len(self._pool) > max_pool:
            self._pool.pop(0)
            # Pool indices shifted down; remap live assignments.
            self._assignments = np.where(
                self._assignments > 0,
                self._assignments - 1,
                np.where(self._assignments == 0, LATEST, self._assignments),
            )

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    def _resample_assignment(self, i: int):
        if self._pool and self._rng.random() > self.latest_prob:
            self._assignments[i] = self._rng.integers(len(self._pool))
        else:
            self._assignments[i] = LATEST

    def _opponent_net(self, key: int) -> torch.nn.Module:
        net = self._latest if key == LATEST else self._pool[key]
        assert net is not None, "call set_latest() before stepping"
        return net

    def _unpack(self, obs, oracle_obs, feats, n_actions):
        obs = obs.reshape(self.num_envs, self.obs_dim)
        oracle_obs = oracle_obs.reshape(self.num_envs, self.obs_dim)
        feats = feats.reshape(self.num_envs, self.max_actions, self.act_feat_dim)
        mask = np.arange(self.max_actions)[None, :] < n_actions[:, None]
        return obs, oracle_obs, feats, mask

    def _record(self, env_ids, rewards, dones, outcomes):
        self._pending_reward[env_ids] += rewards
        for env_id, done, outcome in zip(env_ids, dones, outcomes):
            if done:
                self._pending_done[env_id] = True
                self._pending_outcome[env_id] = outcome
                self._resample_assignment(env_id)

    def _advance_opponents(self):
        """Play frozen-policy moves until every env awaits a seat-0 decision."""
        while True:
            seats = np.asarray(self._env.seats())
            waiting = np.flatnonzero(seats == 1)
            if waiting.size == 0:
                return
            obs, _, feats, mask = self._unpack(*self._env.observe())
            # Snapshot the grouping: stepping a group can finish an episode
            # and resample that env's assignment, which must not re-enter a
            # later group within this pass (its observation would be stale).
            keys = self._assignments[waiting].copy()
            for key in np.unique(keys):
                env_ids = waiting[keys == key]
                net = self._opponent_net(int(key))
                with torch.no_grad():
                    actions, _, _ = net.act(
                        torch.as_tensor(obs[env_ids], dtype=torch.float32),
                        None,
                        torch.as_tensor(feats[env_ids], dtype=torch.float32),
                        torch.as_tensor(mask[env_ids]),
                    )
                rewards, dones, outcomes = self._env.step_some(
                    [int(i) for i in env_ids], [int(a) for a in actions]
                )
                self._record(env_ids, rewards, dones, outcomes)

    def reset(self):
        self._env.reset()
        self._pending_reward[:] = 0.0
        self._pending_done[:] = False
        self._pending_outcome[:] = 0
        for i in range(self.num_envs):
            self._resample_assignment(i)
        self._advance_opponents()
        return self._unpack(*self._env.observe())

    def step(self, actions: np.ndarray):
        """Seat-0 actions for all envs; returns the next seat-0 decision."""
        all_ids = np.arange(self.num_envs)
        rewards, dones, outcomes = self._env.step_some(
            [int(i) for i in all_ids], [int(a) for a in actions]
        )
        self._record(all_ids, rewards, dones, outcomes)
        self._advance_opponents()
        obs, oracle_obs, feats, mask = self._unpack(*self._env.observe())
        out = (
            self._pending_reward.copy(),
            self._pending_done.copy(),
            self._pending_outcome.copy(),
        )
        self._pending_reward[:] = 0.0
        self._pending_done[:] = False
        self._pending_outcome[:] = 0
        return obs, oracle_obs, feats, mask, *out
