"""Behavior-cloning warm start from engine-bot games.

Plays bot-vs-bot games (default expectiminimax-3 on both seats) through the
self-play env, records seat-0 decisions, and distills them into a policy:
cross-entropy on the chosen action over masked legal-action logits, plus a
value regression toward the final outcome. The policy is trained on the
*hidden-information* observation while the demonstrator saw everything —
perfect-training-imperfect-execution distillation. The resulting checkpoint
is meant to seed PPO/self-play fine-tuning (--resume), not to be deployed
as-is.

Example:
    python rl/bc_pretrain.py --bot e3 --episodes 2000 --out runs/bc_e3/bc.pt
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from deckgym import PyRlVecEnv

from agent import make_agent
from torch_runtime import resolve_device, save_agent_state


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--deck",
        default="example_decks/venusaur-exeggutor.txt",
        help="seat-0 deck: file, folder of decks, or comma-separated list "
        "(a pool is sampled per episode — use folders for deck-general training)",
    )
    p.add_argument("--opponent-deck", default=None, help="seat-1 deck spec (defaults to --deck)")
    p.add_argument("--bot", default="e3", help="demonstrator player code for both seats")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument(
        "--dataset",
        default=None,
        help="cache path for the collected games; reused if it exists (for ablations)",
    )
    p.add_argument("--arch", choices=["res", "tx", "mlp", "gen"], default="res")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--memory", action="store_true", help="GRU agent (trained stateless in BC)")
    p.add_argument(
        "--oracle",
        action="store_true",
        help="all-knowing policy: train on the full-state view (like the demonstrator)",
    )
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument(
        "--aux-belief",
        type=float,
        default=0.0,
        help="also train an opponent-hand prediction head (supervised by the oracle view)",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/bc/bc.pt")
    return p.parse_args()


def collect(args):
    env = PyRlVecEnv(
        args.deck, args.opponent_deck or args.deck, "self", args.num_envs, args.seed, 0.0
    )
    obs_dim = env.obs_dim()
    feat_dim = env.action_feat_dim()
    max_actions = env.max_actions()
    env.reset()

    samples = []  # finished: (obs, oracle, feats[n, F], action, outcome)
    open_eps: list[list] = [[] for _ in range(args.num_envs)]
    finished = 0
    start = time.time()
    while finished < args.episodes:
        seats = np.asarray(env.seats())
        obs, oracle, feats, n_actions = env.observe()
        obs = obs.reshape(args.num_envs, obs_dim)
        oracle = oracle.reshape(args.num_envs, obs_dim)
        feats = feats.reshape(args.num_envs, max_actions, feat_dim)
        all_ids = list(range(args.num_envs))
        actions = env.bot_decide_many(all_ids, args.bot)
        for i in all_ids:
            if seats[i] == 0:
                n = int(n_actions[i])
                open_eps[i].append(
                    (obs[i].copy(), oracle[i].copy(), feats[i, :n].copy(), actions[i])
                )
        _, dones, outcomes = env.step_some(all_ids, actions)
        for i, (done, outcome) in enumerate(zip(dones, outcomes)):
            if done:
                for rec in open_eps[i]:
                    samples.append((*rec, int(outcome)))
                open_eps[i] = []
                finished += 1
                if finished % 200 == 0:
                    eps = finished / (time.time() - start)
                    print(
                        f"collected {finished}/{args.episodes} episodes "
                        f"({len(samples)} samples, {eps:.1f} eps/s)",
                        flush=True,
                    )
    return samples, obs_dim, feat_dim


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    if args.dataset and Path(args.dataset).exists():
        with open(args.dataset, "rb") as f:
            samples, obs_dim, feat_dim = pickle.load(f)
        print(f"loaded {len(samples)} samples from {args.dataset}")
    else:
        samples, obs_dim, feat_dim = collect(args)
        if args.dataset:
            Path(args.dataset).parent.mkdir(parents=True, exist_ok=True)
            with open(args.dataset, "wb") as f:
                pickle.dump((samples, obs_dim, feat_dim), f)
            print(f"cached dataset to {args.dataset}")
    outcomes = np.array([s[4] for s in samples], dtype=np.float32)
    print(
        f"dataset: {len(samples)} decisions, "
        f"demonstrator seat-0 win rate {np.mean(outcomes > 0):.3f}"
    )

    if args.oracle and args.aux_belief > 0:
        print("oracle policy sees everything; disabling --aux-belief")
        args.aux_belief = 0.0
    agent = make_agent(
        obs_dim,
        feat_dim,
        args.arch,
        args.hidden,
        args.blocks,
        args.heads,
        memory=args.memory,
        belief=args.aux_belief > 0,
        oracle=args.oracle,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    indices = np.arange(len(samples))
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        total_loss = total_acc = batches = 0
        for s in range(0, len(indices), args.batch_size):
            batch = [samples[j] for j in indices[s : s + args.batch_size]]
            n_max = max(len(rec[2]) for rec in batch)
            bsz = len(batch)
            # rec = (obs, oracle, feats, action, outcome); oracle agents
            # train the policy on the full-state view like the demonstrator.
            pol_field = 1 if args.oracle else 0
            obs_b = torch.from_numpy(np.stack([rec[pol_field] for rec in batch])).to(device)
            oracle_b = torch.from_numpy(np.stack([rec[1] for rec in batch])).to(device)
            feats_b = torch.zeros(bsz, n_max, feat_dim, device=device)
            mask_b = torch.zeros(bsz, n_max, dtype=torch.bool, device=device)
            for k, rec in enumerate(batch):
                n = len(rec[2])
                feats_b[k, :n] = torch.from_numpy(rec[2])
                mask_b[k, :n] = True
            actions_b = torch.tensor([rec[3] for rec in batch], device=device)
            outcome_b = torch.tensor([float(rec[4]) for rec in batch], device=device)

            if args.memory:
                # i.i.d. samples: the GRU runs one step from zero hidden, so
                # it learns a state-independent transform; RL fine-tuning
                # later exploits the recurrence.
                logits, _ = agent.policy_logits(obs_b, feats_b, mask_b)
                value = agent.value(oracle_b)
            else:
                logits, value = agent(obs_b, oracle_b, feats_b, mask_b)
            policy_loss = F.cross_entropy(logits, actions_b)
            value_loss = F.mse_loss(value, outcome_b)
            loss = policy_loss + args.vf_coef * value_loss
            if args.aux_belief > 0:
                loss = loss + args.aux_belief * agent.aux_belief_loss(obs_b, oracle_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (logits.argmax(-1) == actions_b).float().mean().item()
            batches += 1
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={total_loss / batches:.4f} "
            f"top1_acc={total_acc / batches:.3f}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_agent_state(agent, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
