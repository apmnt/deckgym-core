"""Collect a checkpoint's games against an engine bot into a BC dataset.

The checkpoint plays seat 0 (greedy) with `--deck`; the engine bot plays
seat 1 with `--opponent-deck` (specs may be pools). Seat-0 decisions are
recorded in bc_pretrain's dataset format with the final outcome.

Use case: specialist distillation. Per-deck RL specialists reliably gain
large per-deck lifts vs e3 but erode under further RL; recording their
play and merging it by supervised distillation preserves the lines
(proven by the M1 champion-beater merge).

Example:
    python rl/collect_vs_bot.py runs/spec_arceusdialga/final.pt \
        --deck example_decks/arceusdialga.txt --opponent-deck rl/pools/train.pool \
        --bot e3 --episodes 1500 --out runs/spec_arceusdialga_games.pkl
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_wrapper import VecEnv  # noqa: E402
from torch_runtime import agent_from_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--deck", required=True)
    p.add_argument("--opponent-deck", default="rl/pools/train.pool")
    p.add_argument("--bot", default="e3")
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cpu")

    env = VecEnv(
        args.deck, args.opponent_deck, opponent=args.bot,
        num_envs=args.num_envs, seed=args.seed,
    )
    agent = agent_from_checkpoint(args.checkpoint, env.obs_dim, env.act_feat_dim, device)
    agent.eval()

    samples = []
    open_eps: list[list] = [[] for _ in range(args.num_envs)]
    finished = 0
    start = time.time()
    obs, oracle, feats, mask = env.reset()
    h = agent.initial_state(args.num_envs, device)
    while finished < args.episodes:
        with torch.inference_mode():
            actions, _, _, h = agent.act(
                torch.as_tensor(obs, dtype=torch.float32),
                None,
                torch.as_tensor(feats, dtype=torch.float32),
                torch.as_tensor(mask),
                h,
                greedy=not args.sample,
            )
        acts = [int(a) for a in actions]
        for i in range(args.num_envs):
            n = int(mask[i].sum())
            open_eps[i].append(
                (obs[i].copy(), oracle[i].copy(), feats[i, :n].copy(), acts[i])
            )
        obs, oracle, feats, mask, _, dones, outcomes = env.step(np.array(acts))
        if h is not None:
            done_t = torch.as_tensor(dones, dtype=torch.float32)
            h = h.clone() * (1.0 - done_t).unsqueeze(-1)
        for i, (done, outcome) in enumerate(zip(dones, outcomes)):
            if done:
                for o, orc, f, a in open_eps[i]:
                    samples.append((o, orc, f, a, int(outcome)))
                open_eps[i] = []
                finished += 1
                if finished % 500 == 0:
                    print(
                        f"{finished}/{args.episodes} eps "
                        f"({len(samples)} samples, {finished / (time.time() - start):.1f} eps/s)",
                        flush=True,
                    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump((samples, env.obs_dim, env.act_feat_dim), f)
    print(f"wrote {args.out}: {len(samples)} samples")


if __name__ == "__main__":
    main()
