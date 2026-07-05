"""Per-deck evaluation: how does a checkpoint fare with each deck it can pilot?

For every deck in a pool, plays `--episodes` games with that deck in seat 0
against opponents piloting decks sampled from the (full) pool, and reports
the per-deck win rate sorted ascending. This turns the aggregate pool win
rate into a diagnosis: which decks the agent pilots poorly are the targets
for focused training legs (the M1 lesson — focused training reaches levels
that uniform multi-deck training does not).

Example:
    python rl/matchup_matrix.py runs/gen_m2/final.pt --opponent e3 \
        --pool rl/pools/train.pool --episodes 40
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_wrapper import VecEnv  # noqa: E402
from torch_runtime import agent_from_checkpoint, resolve_device  # noqa: E402


def read_pool(path: str) -> list[str]:
    base = Path(path).parent
    decks = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            decks.append(str((base / line).resolve()))
    return decks


def eval_deck(agent, deck: str, pool_spec: str, opponent: str, episodes: int,
              num_envs: int, seed: int, device) -> tuple[int, int, int]:
    env = VecEnv(deck, pool_spec, opponent=opponent, num_envs=num_envs, seed=seed)
    obs, oracle, feats, mask = env.reset()
    h = agent.initial_state(num_envs, device)
    wins = losses = ties = 0
    while wins + losses + ties < episodes:
        with torch.inference_mode():
            action, _, _, h = agent.act(
                torch.as_tensor(obs, dtype=torch.float32, device=device),
                None,
                torch.as_tensor(feats, dtype=torch.float32, device=device),
                torch.as_tensor(mask, device=device),
                h,
                greedy=True,
            )
        obs, oracle, feats, mask, _, dones, outcomes = env.step(action.cpu().numpy())
        if h is not None:
            done_t = torch.as_tensor(dones, dtype=torch.float32, device=device)
            h = h.clone() * (1.0 - done_t).unsqueeze(-1)
        for done, outcome in zip(dones, outcomes):
            if done:
                wins += outcome > 0
                losses += outcome < 0
                ties += outcome == 0
    return wins, losses, ties


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--pool", default="rl/pools/train.pool")
    p.add_argument("--opponent", default="e3")
    p.add_argument("--episodes", type=int, default=40, help="episodes per seat-0 deck")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = resolve_device(args.device)

    decks = read_pool(args.pool)
    probe = VecEnv(decks[0], decks[0], opponent="r", num_envs=1)
    agent = agent_from_checkpoint(
        args.checkpoint, probe.obs_dim, probe.act_feat_dim, map_location=device
    ).to(device)
    agent.eval()

    results = []
    for deck in decks:
        wins, losses, ties = eval_deck(
            agent, deck, args.pool, args.opponent, args.episodes,
            args.num_envs, args.seed, device,
        )
        total = wins + losses + ties
        results.append((wins / total, deck, wins, total))
        print(f"{Path(deck).stem:35s} {wins / total:.3f} ({wins}/{total})", flush=True)

    results.sort()
    overall = sum(r[2] for r in results) / sum(r[3] for r in results)
    print(f"\noverall vs {args.opponent}: {overall:.3f}")
    print("\nworst decks (focused-training targets):")
    for rate, deck, wins, total in results[:8]:
        print(f"  {Path(deck).stem:35s} {rate:.3f}")


if __name__ == "__main__":
    main()
