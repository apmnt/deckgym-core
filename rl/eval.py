"""Evaluate a trained checkpoint against built-in opponents.

Reports the win rate with a Wilson 95% confidence interval per opponent.
`--opponent` accepts a comma-separated panel (e.g. r,v,e2,e3) and `--seeds`
a comma-separated list whose episode counts are pooled per opponent.

Example:
    python rl/eval.py runs/<run>/latest.pt --opponent r,v,e2,e3 --episodes 200 --seeds 999,4242
"""

import argparse
import math

import numpy as np
import torch

from env_wrapper import VecEnv
from torch_runtime import (
    agent_from_checkpoint,
    configure_device,
    resolve_amp_dtype,
    resolve_device,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--deck", default="example_decks/venusaur-exeggutor.txt")
    p.add_argument("--opponent-deck", default=None)
    p.add_argument("--opponent", default="r", help="player code or comma-separated panel")
    p.add_argument("--episodes", type=int, default=500, help="episodes per opponent per seed")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--seeds", default=None, help="comma-separated env seeds (overrides --seed)")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--amp", action="store_true", help="use autocast on CUDA")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--sample", action="store_true", help="sample instead of argmax")
    return p.parse_args()


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p_hat = wins / total
    denom = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / total + z * z / (4 * total * total)) / denom
    return center - margin, center + margin


def run_matches(agent, args, opponent: str, seed: int, device, amp_dtype, use_cuda_amp):
    env = VecEnv(
        args.deck,
        args.opponent_deck or args.deck,
        opponent=opponent,
        num_envs=args.num_envs,
        seed=seed,
    )
    obs, _oracle, feats, mask = env.reset()
    h = agent.initial_state(args.num_envs, device)
    wins = losses = ties = 0
    while wins + losses + ties < args.episodes:
        # Evaluation is honest: only the hidden-information observation is
        # used; the oracle view exists solely for the training-time critic.
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_cuda_amp
        ):
            action, _, _, h = agent.act(
                torch.as_tensor(obs, dtype=torch.float32, device=device),
                None,
                torch.as_tensor(feats, dtype=torch.float32, device=device),
                torch.as_tensor(mask, device=device),
                h,
                greedy=not args.sample,
            )
        obs, _oracle, feats, mask, _, dones, outcomes = env.step(action.cpu().numpy())
        if h is not None:
            done_t = torch.as_tensor(dones, dtype=torch.float32, device=device)
            h = h.clone() * (1.0 - done_t).unsqueeze(-1)
        for done, outcome in zip(dones, outcomes):
            if done:
                if outcome > 0:
                    wins += 1
                elif outcome < 0:
                    losses += 1
                else:
                    ties += 1
    return wins, losses, ties


def main():
    args = parse_args()
    device = resolve_device(args.device)
    use_cuda_amp = args.amp and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    configure_device(device)

    # Any env instance of the right matchup defines the dims.
    probe = VecEnv(args.deck, args.opponent_deck or args.deck, opponent="r", num_envs=1)
    agent = agent_from_checkpoint(
        args.checkpoint, probe.obs_dim, probe.act_feat_dim, map_location=device
    ).to(device)
    agent.eval()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    for opponent in args.opponent.split(","):
        wins = losses = ties = 0
        for seed in seeds:
            w, l, t = run_matches(agent, args, opponent, seed, device, amp_dtype, use_cuda_amp)
            wins, losses, ties = wins + w, losses + l, ties + t
        total = wins + losses + ties
        lo, hi = wilson_interval(wins, total)
        print(
            f"vs {opponent}: {total} episodes | "
            f"win {wins / total:.3f} [{lo:.3f}, {hi:.3f}] | "
            f"loss {losses / total:.3f} | tie {ties / total:.3f}"
        )


if __name__ == "__main__":
    main()
