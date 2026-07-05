"""Pit two trained checkpoints against each other in the same games.

Seat 0 gets --agent-a, seat 1 gets --agent-b; the env pauses at both
seats' decisions (PyRlVecEnv "self" mode) and each policy acts on the
observation from its own perspective. Outcomes are reported from agent A's
side. Run it twice with --swap to cancel any seat asymmetry, or pass
--both to do that automatically.

Legacy checkpoints (trained before the deck-general observation layout)
are detected by their input width and fed through `LegacyObsAdapter`,
which slices the fixed-40-slot vocabulary sections down to the match's
actual vocabulary and drops the trailing global-id section. Both layouts
sort the vocabulary by card id, so the slice is lossless — a legacy
specialist sees exactly the observation it was trained on, provided the
matchup is the one it specialized in.

Example (original mirror champion vs the deck-general agent, its home turf):
    python rl/head_to_head.py rl/checkpoints/general_pfsp.pt \
        rl/checkpoints/bc_pfsp_champion.pt \
        --deck example_decks/venusaur-exeggutor.txt --episodes 400 --both
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deckgym import PyRlVecEnv  # noqa: E402
from torch_runtime import agent_from_checkpoint, resolve_device  # noqa: E402

GLOBALS = 58
NUM_SLOTS = 8
SLOT_NUM = 23
NUM_ZONES = 6
ACT_CLASSES = 18
GEN_VOCAB = 40
ACT_TAIL = 4 + 1 + 4 + 10 + 3  # target slot+side, attack idx, energy, scalars (rl_env.rs)


def gen_obs_dim(vocab: int) -> int:
    return GLOBALS + NUM_SLOTS * (SLOT_NUM + vocab) + NUM_ZONES * vocab


class LegacyObsAdapter:
    """Slice deck-general observations/action features down to the legacy
    per-match layout with `vocab` entries (vocab <= 40)."""

    def __init__(self, vocab: int):
        self.vocab = vocab
        # Observation gather index: globals, then per-slot
        # [occupied | onehot[:vocab] | numeric], then zone counts[:vocab].
        idx = list(range(GLOBALS))
        for slot in range(NUM_SLOTS):
            base = GLOBALS + slot * (SLOT_NUM + GEN_VOCAB)
            idx.append(base)  # occupied flag
            idx.extend(range(base + 1, base + 1 + vocab))  # card one-hot
            idx.extend(range(base + 1 + GEN_VOCAB, base + SLOT_NUM + GEN_VOCAB))
        zones_base = GLOBALS + NUM_SLOTS * (SLOT_NUM + GEN_VOCAB)
        for zone in range(NUM_ZONES):
            base = zones_base + zone * GEN_VOCAB
            idx.extend(range(base, base + vocab))
        self.obs_idx = np.array(idx)
        # Action features: [classes | onehot[:vocab] | tail].
        act_idx = list(range(ACT_CLASSES))
        act_idx.extend(range(ACT_CLASSES, ACT_CLASSES + vocab))
        act_idx.extend(range(ACT_CLASSES + GEN_VOCAB, ACT_CLASSES + GEN_VOCAB + ACT_TAIL))
        self.act_idx = np.array(act_idx)

    def obs(self, obs: np.ndarray) -> np.ndarray:
        return obs[:, self.obs_idx]

    def feats(self, feats: np.ndarray) -> np.ndarray:
        return feats[:, :, self.act_idx]


def load_agent(path: str, probe_obs_dim: int, probe_feat_dim: int, device):
    """Load a checkpoint, detecting whether it uses the legacy layout."""
    import torch as _torch

    obj = _torch.load(path, map_location=device, weights_only=True)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    config = obj.get("config") if isinstance(obj, dict) else None
    if config is not None and config.get("arch") == "gen":
        agent = agent_from_checkpoint(path, probe_obs_dim, probe_feat_dim, device)
        return agent.to(device).eval(), None
    # Legacy: infer its obs/action dims from the first layer weights.
    obs_key = next(
        k for k in ("obs_encoder.0.weight", "obs_encoder.global_proj.weight") if k in state
    )
    legacy_obs_dim = state[obs_key].shape[1] if obs_key == "obs_encoder.0.weight" else None
    if legacy_obs_dim is None:
        raise SystemExit("tx-arch legacy checkpoints are not supported here")
    legacy_vocab = (legacy_obs_dim - 242) // 14
    legacy_feat_dim = state["act_encoder.0.weight"].shape[1]
    agent = agent_from_checkpoint(path, legacy_obs_dim, legacy_feat_dim, device)
    return agent.to(device).eval(), LegacyObsAdapter(legacy_vocab)


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p_hat = wins / total
    denom = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / total + z * z / (4 * total * total)) / denom
    return center - margin, center + margin


def run(agent_a, adapter_a, agent_b, adapter_b, args, seed, device):
    """A in seat 0, B in seat 1. Returns (a_wins, b_wins, ties)."""
    env = PyRlVecEnv(args.deck, args.opponent_deck or args.deck, "self", args.num_envs, seed, 0.0)
    obs_dim, feat_dim, max_actions = env.obs_dim(), env.action_feat_dim(), env.max_actions()
    env.reset()
    agents = [(agent_a, adapter_a), (agent_b, adapter_b)]
    # sample_flags is aligned with *this call's* seating: index = seat.
    sample_flags = getattr(args, "_seat_sample", (args.sample, args.sample))
    h = [a.initial_state(args.num_envs, device) for a, _ in agents]
    a_wins = b_wins = ties = 0
    while a_wins + b_wins + ties < args.episodes:
        seats = np.asarray(env.seats())
        obs, _, feats, n_actions = env.observe()
        obs = obs.reshape(args.num_envs, obs_dim)
        feats = feats.reshape(args.num_envs, max_actions, feat_dim)
        mask = np.arange(max_actions)[None, :] < np.asarray(n_actions)[:, None]
        for seat in (0, 1):
            ids = np.flatnonzero(seats == seat)
            if ids.size == 0:
                continue
            agent, adapter = agents[seat]
            view, av = obs[ids], feats[ids]
            if adapter is not None:
                view, av = adapter.obs(view), adapter.feats(av)
            h_in = h[seat][ids] if h[seat] is not None else None
            with torch.inference_mode():
                actions, _, _, h_new = agent.act(
                    torch.as_tensor(view, dtype=torch.float32, device=device),
                    None,
                    torch.as_tensor(av, dtype=torch.float32, device=device),
                    torch.as_tensor(mask[ids], device=device),
                    h_in,
                    greedy=not sample_flags[seat],
                )
            if h[seat] is not None and h_new is not None:
                h[seat][ids] = h_new.float()
            _, dones, outcomes = env.step_some(
                [int(i) for i in ids], [int(x) for x in actions.cpu()]
            )
            for env_id, done, outcome in zip(ids, dones, outcomes):
                if done:
                    if outcome > 0:
                        a_wins += 1
                    elif outcome < 0:
                        b_wins += 1
                    else:
                        ties += 1
                    for s in (0, 1):
                        if h[s] is not None:
                            h[s][env_id] = 0.0
    return a_wins, b_wins, ties


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("agent_a", help="checkpoint for seat 0")
    p.add_argument("agent_b", help="checkpoint for seat 1")
    p.add_argument("--deck", default="example_decks/venusaur-exeggutor.txt")
    p.add_argument("--opponent-deck", default=None)
    p.add_argument("--episodes", type=int, default=400, help="episodes per seating")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seeds", default="999,4242", help="comma-separated env seeds")
    p.add_argument("--sample", action="store_true", help="both agents sample instead of argmax")
    p.add_argument("--sample-a", action="store_true", help="agent A samples (stochastic policy)")
    p.add_argument("--sample-b", action="store_true", help="agent B samples (stochastic policy)")
    p.add_argument("--swap", action="store_true", help="B in seat 0 instead")
    p.add_argument("--both", action="store_true", help="run both seatings and pool")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = resolve_device(args.device)

    probe = PyRlVecEnv(args.deck, args.opponent_deck or args.deck, "r", 1, 0, 0.0)
    agent_a, adapter_a = load_agent(args.agent_a, probe.obs_dim(), probe.action_feat_dim(), device)
    agent_b, adapter_b = load_agent(args.agent_b, probe.obs_dim(), probe.action_feat_dim(), device)
    for name, adapter in ((args.agent_a, adapter_a), (args.agent_b, adapter_b)):
        kind = f"legacy (vocab {adapter.vocab})" if adapter else "deck-general"
        print(f"{name}: {kind}")

    sample_a = args.sample or args.sample_a
    sample_b = args.sample or args.sample_b
    seatings = [False, True] if args.both else [args.swap]
    total_a = total_b = total_t = 0
    for swap in seatings:
        for seed in (int(s) for s in args.seeds.split(",")):
            args._seat_sample = (sample_b, sample_a) if swap else (sample_a, sample_b)
            if swap:
                b, a, t = run(agent_b, adapter_b, agent_a, adapter_a, args, seed, device)
            else:
                a, b, t = run(agent_a, adapter_a, agent_b, adapter_b, args, seed, device)
            seat_a = 1 if swap else 0
            print(f"seed {seed}, A in seat {seat_a}: A {a} B {b} ties {t}")
            total_a, total_b, total_t = total_a + a, total_b + b, total_t + t
    total = total_a + total_b + total_t
    lo, hi = wilson_interval(total_a, total)
    print(
        f"\nTOTAL ({total} episodes): A wins {total_a / total:.3f} [{lo:.3f}, {hi:.3f}] | "
        f"B wins {total_b / total:.3f} | ties {total_t / total:.3f}"
    )


if __name__ == "__main__":
    main()
