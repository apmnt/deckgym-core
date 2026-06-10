"""PPO training loop for the deckgym RL environment (CleanRL-style).

Example:
    python rl/train_ppo.py --opponent r --total-steps 2000000
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agent import ActionScorerAgent
from env_wrapper import VecEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deck", default="example_decks/venusaur-exeggutor.txt")
    p.add_argument("--opponent-deck", default=None, help="defaults to --deck (mirror match)")
    p.add_argument("--opponent", default="r", help="engine player code: r, w, aa, v, e2, m, ...")
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--shaping", type=float, default=0.0, help="potential-based shaping coef")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume weights from")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    run_name = args.run_name or f"ppo_{args.opponent}_{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = VecEnv(
        args.deck,
        args.opponent_deck or args.deck,
        opponent=args.opponent,
        num_envs=args.num_envs,
        seed=args.seed,
        shaping_coef=args.shaping,
    )
    agent = ActionScorerAgent(env.obs_dim, env.act_feat_dim).to(device)
    if args.resume:
        agent.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed weights from {args.resume}")
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    num_steps, num_envs = args.num_steps, args.num_envs
    batch_size = num_steps * num_envs
    minibatch_size = batch_size // args.num_minibatches

    obs_buf = torch.zeros(num_steps, num_envs, env.obs_dim)
    feats_buf = torch.zeros(num_steps, num_envs, env.max_actions, env.act_feat_dim)
    mask_buf = torch.zeros(num_steps, num_envs, env.max_actions, dtype=torch.bool)
    actions_buf = torch.zeros(num_steps, num_envs, dtype=torch.long)
    logprobs_buf = torch.zeros(num_steps, num_envs)
    rewards_buf = torch.zeros(num_steps, num_envs)
    dones_buf = torch.zeros(num_steps, num_envs)
    values_buf = torch.zeros(num_steps, num_envs)

    obs, feats, mask = env.reset()
    next_done = torch.zeros(num_envs)
    recent_outcomes: deque[int] = deque(maxlen=200)
    global_step = 0
    start = time.time()

    while global_step < args.total_steps:
        for step in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            feats_t = torch.as_tensor(feats, dtype=torch.float32)
            mask_t = torch.as_tensor(mask)
            with torch.no_grad():
                action, logprob, value = agent.act(
                    obs_t.to(device), feats_t.to(device), mask_t.to(device)
                )
            obs_buf[step], feats_buf[step], mask_buf[step] = obs_t, feats_t, mask_t
            actions_buf[step] = action.cpu()
            logprobs_buf[step] = logprob.cpu()
            values_buf[step] = value.cpu()
            dones_buf[step] = next_done

            obs, feats, mask, rewards, dones, outcomes = env.step(action.cpu().numpy())
            rewards_buf[step] = torch.as_tensor(rewards)
            next_done = torch.as_tensor(dones, dtype=torch.float32)
            for done, outcome in zip(dones, outcomes):
                if done:
                    recent_outcomes.append(int(outcome))
            global_step += num_envs

        # GAE. Done envs auto-reset, so the value of the post-done observation
        # belongs to the next episode and is masked out by next_done.
        with torch.no_grad():
            next_value = agent.act(
                torch.as_tensor(obs, dtype=torch.float32).to(device),
                torch.as_tensor(feats, dtype=torch.float32).to(device),
                torch.as_tensor(mask).to(device),
            )[2].cpu()
        advantages = torch.zeros_like(rewards_buf)
        last_gae = torch.zeros(num_envs)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_nonterminal = 1.0 - next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - dones_buf[t + 1]
                next_values = values_buf[t + 1]
            delta = rewards_buf[t] + args.gamma * next_values * next_nonterminal - values_buf[t]
            last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values_buf

        b_obs = obs_buf.reshape(batch_size, -1)
        b_feats = feats_buf.reshape(batch_size, env.max_actions, -1)
        b_mask = mask_buf.reshape(batch_size, -1)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        indices = np.arange(batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            for mb_start in range(0, batch_size, minibatch_size):
                mb = indices[mb_start : mb_start + minibatch_size]
                logits, value = agent(
                    b_obs[mb].to(device), b_feats[mb].to(device), b_mask[mb].to(device)
                )
                dist = torch.distributions.Categorical(logits=logits)
                new_logprob = dist.log_prob(b_actions[mb].to(device))
                entropy = dist.entropy()
                logratio = new_logprob - b_logprobs[mb].to(device)
                ratio = logratio.exp()
                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_adv = b_advantages[mb].to(device)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                v_loss = 0.5 * (value - b_returns[mb].to(device)).pow(2).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if recent_outcomes:
            wins = sum(1 for o in recent_outcomes if o > 0)
            losses = sum(1 for o in recent_outcomes if o < 0)
            win_rate = wins / len(recent_outcomes)
        else:
            win_rate, losses = float("nan"), 0
        sps = int(global_step / (time.time() - start))
        print(
            f"step={global_step} win_rate={win_rate:.3f} "
            f"(n={len(recent_outcomes)}, losses={losses}) "
            f"v_loss={v_loss.item():.3f} pg_loss={pg_loss.item():.3f} "
            f"clipfrac={np.mean(clipfracs):.3f} sps={sps}",
            flush=True,
        )
        torch.save(agent.state_dict(), run_dir / "latest.pt")

    torch.save(agent.state_dict(), run_dir / "final.pt")
    print(f"done. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
