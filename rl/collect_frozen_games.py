"""Collect games between two frozen checkpoints into a BC dataset.

Both seats are driven by trained policies (legacy checkpoints adapted, as
in head_to_head.py) and every decision from BOTH seats is recorded in
bc_pretrain's dataset format — (obs, oracle_obs, feats[n], action,
outcome) with observation and outcome from the acting seat's perspective.

Use case: distill-then-exploit. BC-cloning the mirror champion's own games
puts the general architecture *inside* the champion's strategy region,
from which self-play against the frozen champion searches for deviations
that beat it (fictitious-play intuition), instead of approaching from the
far-away general policy.

Example:
    python rl/collect_frozen_games.py rl/checkpoints/bc_pfsp_champion.pt \
        rl/checkpoints/bc_pfsp_champion.pt --episodes 3000 \
        --out runs/champ_games.pkl
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deckgym import PyRlVecEnv  # noqa: E402
from head_to_head import load_agent  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("agent_a", help="seat-0 checkpoint")
    p.add_argument("agent_b", help="seat-1 checkpoint")
    p.add_argument("--deck", default="example_decks/venusaur-exeggutor.txt")
    p.add_argument("--opponent-deck", default=None)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sample", action="store_true", help="agents sample instead of argmax")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cpu")

    env = PyRlVecEnv(
        args.deck, args.opponent_deck or args.deck, "self", args.num_envs, args.seed, 0.0
    )
    obs_dim, feat_dim, max_actions = env.obs_dim(), env.action_feat_dim(), env.max_actions()
    agents = [
        load_agent(args.agent_a, obs_dim, feat_dim, device),
        load_agent(args.agent_b, obs_dim, feat_dim, device),
    ]
    env.reset()

    samples = []
    # Per env: open decisions [(seat, obs, oracle, feats, action), ...]
    open_eps: list[list] = [[] for _ in range(args.num_envs)]
    finished = 0
    start = time.time()
    while finished < args.episodes:
        seats = np.asarray(env.seats())
        obs, oracle, feats, n_actions = env.observe()
        obs = obs.reshape(args.num_envs, obs_dim)
        oracle = oracle.reshape(args.num_envs, obs_dim)
        feats = feats.reshape(args.num_envs, max_actions, feat_dim)
        mask = np.arange(max_actions)[None, :] < np.asarray(n_actions)[:, None]
        for seat in (0, 1):
            ids = np.flatnonzero(seats == seat)
            if ids.size == 0:
                continue
            agent, adapter = agents[seat]
            view, av = obs[ids], feats[ids]
            if adapter is not None:
                view = view[:, adapter.obs_idx]
                av = av[:, :, adapter.act_idx]
            with torch.inference_mode():
                actions, _, _, _ = agent.act(
                    torch.as_tensor(view, dtype=torch.float32),
                    None,
                    torch.as_tensor(av, dtype=torch.float32),
                    torch.as_tensor(mask[ids]),
                    None,
                    greedy=not args.sample,
                )
            actions = [int(x) for x in actions]
            # Record the *unadapted* observation: the BC consumer is the
            # deck-general architecture.
            for k, env_id in enumerate(ids):
                n = int(n_actions[env_id])
                open_eps[env_id].append(
                    (seat, obs[env_id].copy(), oracle[env_id].copy(),
                     feats[env_id, :n].copy(), actions[k])
                )
            _, dones, outcomes = env.step_some([int(i) for i in ids], actions)
            for env_id, done, outcome in zip(ids, dones, outcomes):
                if done:
                    for seat_k, o, orc, f, a in open_eps[env_id]:
                        # Outcome from the acting seat's perspective.
                        signed = int(outcome) if seat_k == 0 else -int(outcome)
                        samples.append((o, orc, f, a, signed))
                    open_eps[env_id] = []
                    finished += 1
                    if finished % 500 == 0:
                        eps = finished / (time.time() - start)
                        print(
                            f"collected {finished}/{args.episodes} episodes "
                            f"({len(samples)} samples, {eps:.1f} eps/s)",
                            flush=True,
                        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump((samples, obs_dim, feat_dim), f)
    print(f"wrote {args.out}: {len(samples)} samples")


if __name__ == "__main__":
    main()
