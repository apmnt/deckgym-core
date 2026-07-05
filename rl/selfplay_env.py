"""Self-play vec-env: learner in seat 0, frozen policies or engine bots in seat 1.

Wraps `PyRlVecEnv(opponent="self")`, which pauses at *both* seats'
decisions. This wrapper drives seat 1 with a per-episode opponent — a copy
of the latest learner weights, a snapshot from a historical pool, or an
engine bot (e.g. e1/e2/e3) decided in Rust — so the PPO loop sees the exact
single-agent interface of `VecEnv`: every `step(actions)` consumes seat-0
actions for all envs and returns the next seat-0 decision, with rewards and
dones accumulated across any interleaved opponent moves.

Opponent selection is PFSP (prioritized fictitious self-play, AlphaStar
style): with `latest_prob` the opponent is the latest learner copy
(self-play anchor); otherwise pool snapshots and bots are sampled with
probability ∝ (1 - winrate)^pfsp_power, focusing training on the opponents
the learner still loses to.

Frozen-net opponents play honestly: they get the hidden-information
observation from their own perspective and sample (rather than argmax) for
diversity. Engine bots see everything, as they always do.
"""

import copy
from collections import deque

import numpy as np
import torch
from deckgym import PyRlVecEnv

LATEST = -1  # assignment key for "frozen copy of current learner"
BOT_BASE = -2  # bot j is encoded as BOT_BASE - j


class SelfPlayVecEnv:
    def __init__(
        self,
        deck_a: str,
        deck_b: str,
        num_envs: int = 32,
        seed: int = 0,
        shaping_coef: float = 0.0,
        latest_prob: float = 0.5,
        device: str | torch.device = "cpu",
        amp_enabled: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        bot_opponents: list[str] | None = None,
        pfsp_power: float = 2.0,
    ):
        self._env = PyRlVecEnv(deck_a, deck_b, "self", num_envs, seed, shaping_coef)
        self.num_envs = num_envs
        self.obs_dim = self._env.obs_dim()
        self.act_feat_dim = self._env.action_feat_dim()
        self.max_actions = self._env.max_actions()
        self.latest_prob = latest_prob
        self.device = torch.device(device)
        self.amp_enabled = amp_enabled and self.device.type == "cuda"
        self.amp_dtype = amp_dtype
        self.bot_opponents = list(bot_opponents or [])
        self.pfsp_power = pfsp_power
        self._latest: torch.nn.Module | None = None
        self._pool: list[torch.nn.Module] = []
        # Learner outcomes (1 win / 0.5 tie / 0 loss) per opponent arm,
        # rolling window; drives the PFSP sampling weights.
        self._results: dict[int, deque[float]] = {}
        self._rng = np.random.default_rng(seed)
        self._assignments = np.full(num_envs, LATEST, dtype=np.int64)
        self._pending_reward = np.zeros(num_envs, dtype=np.float32)
        self._pending_done = np.zeros(num_envs, dtype=bool)
        self._pending_outcome = np.zeros(num_envs, dtype=np.int8)
        # Per-env GRU state of whichever frozen net plays seat 1 (recurrent
        # agents only). Zeroed when an episode ends (assignments resample at
        # the same moment, so the state never leaks across opponents).
        self._opp_h: torch.Tensor | None = None

    def set_latest(self, agent: torch.nn.Module):
        """Refresh the frozen copy of the learner used as the 'latest' opponent."""
        self._latest = copy.deepcopy(agent).to(self.device).eval()
        if getattr(agent, "gru", None) is not None and self._opp_h is None:
            self._opp_h = torch.zeros(
                self.num_envs, agent.gru.hidden_size, device=self.device
            )

    def add_snapshot(self, agent: torch.nn.Module, max_pool: int):
        self._pool.append(copy.deepcopy(agent).to(self.device).eval())
        if len(self._pool) > max_pool:
            self._pool.pop(0)
            # Pool indices shifted down; remap live assignments and stats.
            self._assignments = np.where(
                self._assignments > 0,
                self._assignments - 1,
                np.where(self._assignments == 0, LATEST, self._assignments),
            )
            self._results = {
                (key - 1 if key > 0 else key): res
                for key, res in self._results.items()
                if key != 0
            }

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    def _winrate(self, key: int) -> float:
        results = self._results.get(key)
        if not results or len(results) < 8:
            return 0.5  # unexplored arms sample at the neutral weight
        return sum(results) / len(results)

    def _resample_assignment(self, i: int):
        arms = list(range(len(self._pool))) + [
            BOT_BASE - j for j in range(len(self.bot_opponents))
        ]
        if not arms or self._rng.random() < self.latest_prob:
            self._assignments[i] = LATEST
            return
        # PFSP: focus on opponents the learner still loses to.
        weights = np.array(
            [(1.0 - self._winrate(key)) ** self.pfsp_power + 0.05 for key in arms]
        )
        self._assignments[i] = self._rng.choice(arms, p=weights / weights.sum())

    def _opponent_net(self, key: int) -> torch.nn.Module:
        net = self._latest if key == LATEST else self._pool[key]
        assert net is not None, "call set_latest() before stepping"
        return net

    def opponent_stats(self) -> dict[str, tuple[float, int]]:
        """Learner winrate and sample count per arm (for logging)."""
        stats = {}
        for key, results in sorted(self._results.items()):
            if key == LATEST:
                name = "latest"
            elif key <= BOT_BASE:
                name = self.bot_opponents[BOT_BASE - key]
            else:
                name = f"pool{key}"
            if results:
                stats[name] = (sum(results) / len(results), len(results))
        return stats

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
                arm = int(self._assignments[env_id])
                self._results.setdefault(arm, deque(maxlen=64)).append(
                    1.0 if outcome > 0 else 0.0 if outcome < 0 else 0.5
                )
                self._resample_assignment(env_id)
                if self._opp_h is not None:
                    self._opp_h[env_id] = 0.0

    def _advance_opponents(self):
        """Play frozen-policy moves until every env awaits a seat-0 decision."""
        while True:
            seats = np.asarray(self._env.seats())
            waiting = np.flatnonzero(seats == 1)
            if waiting.size == 0:
                return
            obs, oracle, feats, mask = self._unpack(*self._env.observe())
            # Snapshot the grouping: stepping a group can finish an episode
            # and resample that env's assignment, which must not re-enter a
            # later group within this pass (its observation would be stale).
            keys = self._assignments[waiting].copy()
            for key in np.unique(keys):
                env_ids = waiting[keys == key]
                if key <= BOT_BASE:
                    # Engine-bot opponent: the decision is made in Rust.
                    code = self.bot_opponents[BOT_BASE - int(key)]
                    ids = [int(i) for i in env_ids]
                    bot_actions = self._env.bot_decide_many(ids, code)
                    rewards, dones, outcomes = self._env.step_some(ids, bot_actions)
                    self._record(env_ids, rewards, dones, outcomes)
                    continue
                net = self._opponent_net(int(key))
                # Oracle agents play with the full-state view; honest agents
                # get the hidden-information view from their own perspective.
                view = oracle if net.config.get("oracle", False) else obs
                h_in = self._opp_h[env_ids] if self._opp_h is not None else None
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    actions, _, _, h_new = net.act(
                        torch.as_tensor(view[env_ids], dtype=torch.float32, device=self.device),
                        None,
                        torch.as_tensor(feats[env_ids], dtype=torch.float32, device=self.device),
                        torch.as_tensor(mask[env_ids], device=self.device),
                        h_in,
                    )
                rewards, dones, outcomes = self._env.step_some(
                    [int(i) for i in env_ids], [int(a) for a in actions.cpu()]
                )
                # Scatter the new memory before _record, which zeroes the
                # rows of episodes that just ended.
                if self._opp_h is not None and h_new is not None:
                    self._opp_h[env_ids] = h_new.float().clone()
                self._record(env_ids, rewards, dones, outcomes)

    def reset(self):
        self._env.reset()
        self._pending_reward[:] = 0.0
        self._pending_done[:] = False
        self._pending_outcome[:] = 0
        if self._opp_h is not None:
            self._opp_h.zero_()
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
