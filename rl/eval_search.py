"""Evaluate a checkpoint with test-time determinized one-ply search.

For each decision, the env computes per-action values by re-dealing the
hidden information (opponent hand/deck split, deck orders) and scoring
the state after each action with the engine's baseline value function
(`PyRlVecEnv.action_values`). Those values are z-scored within the legal
set and added to the policy's logits:

    score(a) = logits(a) + beta * zscore(V_determinized(a))

beta = 0 is the plain policy; larger beta trusts the search more. The
search reads only legal information, so this remains an honest player.

Example (sweep beta against e3 on the pool):
    uv run --no-sync python rl/eval_search.py rl/checkpoints/general_m2.pt \
        --deck rl/pools/train.pool --opponent e3 --betas 0,1,2,4 \
        --determinizations 12 --episodes 150
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_wrapper import VecEnv  # noqa: E402
from torch_runtime import agent_from_checkpoint, resolve_device  # noqa: E402


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p_hat = wins / total
    denom = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / total + z * z / (4 * total * total)) / denom
    return center - margin, center + margin


def run(agent, args, beta: float, seed: int, device) -> tuple[int, int, int]:
    env = VecEnv(
        args.deck,
        args.opponent_deck or args.deck,
        opponent=args.opponent,
        num_envs=args.num_envs,
        seed=seed,
    )
    obs, _, feats, mask = env.reset()
    h = agent.initial_state(args.num_envs, device)
    wins = losses = ties = 0
    step = 0
    while wins + losses + ties < args.episodes:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        feats_t = torch.as_tensor(feats, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask, device=device)
        with torch.inference_mode():
            logits, h_new = agent.policy_logits(obs_t, feats_t, mask_t, h)
            logits = logits.float()
        if beta > 0:
            raw = np.asarray(
                env._env.action_values(args.determinizations, seed * 1_000_003 + step)
            ).reshape(args.num_envs, env.max_actions)
            values = torch.from_numpy(raw).to(device)
            # z-score within each decision's legal set: scale-free blending.
            v_mean = (values * mask_t).sum(-1, keepdim=True) / mask_t.sum(-1, keepdim=True)
            centered = (values - v_mean) * mask_t
            v_std = (centered.pow(2).sum(-1, keepdim=True) / mask_t.sum(-1, keepdim=True)).sqrt()
            logits = logits + beta * centered / (v_std + 1e-6)
            logits = logits.masked_fill(~mask_t, torch.finfo(logits.dtype).min)
        action = logits.argmax(dim=-1)
        h = h_new
        obs, _, feats, mask, _, dones, outcomes = env.step(action.cpu().numpy())
        if h is not None:
            done_t = torch.as_tensor(dones, dtype=torch.float32, device=device)
            h = h.clone() * (1.0 - done_t).unsqueeze(-1)
        step += 1
        for done, outcome in zip(dones, outcomes):
            if done:
                wins += outcome > 0
                losses += outcome < 0
                ties += outcome == 0
    return wins, losses, ties


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--deck", default="rl/pools/train.pool")
    p.add_argument("--opponent-deck", default=None)
    p.add_argument("--opponent", default="e3")
    p.add_argument("--betas", default="0,1,2,4", help="comma-separated blend weights")
    p.add_argument("--determinizations", type=int, default=12)
    p.add_argument("--episodes", type=int, default=150, help="per beta per seed")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seeds", default="999", help="comma-separated env seeds")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = resolve_device(args.device)

    probe = VecEnv(args.deck, args.opponent_deck or args.deck, opponent="r", num_envs=1)
    agent = agent_from_checkpoint(
        args.checkpoint, probe.obs_dim, probe.act_feat_dim, map_location=device
    ).to(device)
    agent.eval()

    for beta in (float(b) for b in args.betas.split(",")):
        wins = losses = ties = 0
        for seed in (int(s) for s in args.seeds.split(",")):
            w, l, t = run(agent, args, beta, seed, device)
            wins, losses, ties = wins + w, losses + l, ties + t
        total = wins + losses + ties
        lo, hi = wilson_interval(wins, total)
        print(
            f"beta={beta:g}: {total} episodes | win {wins / total:.3f} "
            f"[{lo:.3f}, {hi:.3f}] | loss {losses / total:.3f} | tie {ties / total:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
