"""Evaluate a trained checkpoint against a built-in opponent.

Example:
    python rl/eval.py runs/<run>/latest.pt --opponent r --episodes 500
"""

import argparse

import numpy as np
import torch

from agent import ActionScorerAgent
from env_wrapper import VecEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--deck", default="example_decks/venusaur-exeggutor.txt")
    p.add_argument("--opponent-deck", default=None)
    p.add_argument("--opponent", default="r")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--sample", action="store_true", help="sample instead of argmax")
    return p.parse_args()


def main():
    args = parse_args()
    env = VecEnv(
        args.deck,
        args.opponent_deck or args.deck,
        opponent=args.opponent,
        num_envs=args.num_envs,
        seed=args.seed,
    )
    agent = ActionScorerAgent(env.obs_dim, env.act_feat_dim)
    agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    agent.eval()

    obs, _oracle, feats, mask = env.reset()
    wins = losses = ties = 0
    while wins + losses + ties < args.episodes:
        # Evaluation is honest: only the hidden-information observation is
        # used; the oracle view exists solely for the training-time critic.
        action, _, _ = agent.act(
            torch.as_tensor(obs, dtype=torch.float32),
            None,
            torch.as_tensor(feats, dtype=torch.float32),
            torch.as_tensor(mask),
            greedy=not args.sample,
        )
        obs, _oracle, feats, mask, _, dones, outcomes = env.step(action.numpy())
        for done, outcome in zip(dones, outcomes):
            if done:
                if outcome > 0:
                    wins += 1
                elif outcome < 0:
                    losses += 1
                else:
                    ties += 1

    total = wins + losses + ties
    print(
        f"vs {args.opponent}: {total} episodes | "
        f"win {wins / total:.3f} | loss {losses / total:.3f} | tie {ties / total:.3f}"
    )


if __name__ == "__main__":
    main()
