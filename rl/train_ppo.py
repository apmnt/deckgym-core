"""PPO training loop for the deckgym RL environment (CleanRL-style).

The policy acts on hidden-information observations; the critic reads the
full-state (oracle) observation, which exists only at training time.

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
from torch_runtime import (
    configure_device,
    load_agent_state,
    make_grad_scaler,
    maybe_compile,
    resolve_amp_dtype,
    resolve_device,
    save_agent_state,
)


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
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--amp", action="store_true", help="use autocast on CUDA")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--compile", action="store_true", help="compile the agent with torch.compile")
    p.add_argument("--profile", action="store_true", help="print per-update phase timings")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume weights from")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    use_cuda_amp = args.amp and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    configure_device(device)

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
    raw_agent = ActionScorerAgent(env.obs_dim, env.act_feat_dim).to(device)
    if args.resume:
        load_agent_state(raw_agent, args.resume, map_location=device)
        print(f"resumed weights from {args.resume}")
    agent = maybe_compile(raw_agent, args.compile)
    optimizer = torch.optim.Adam(raw_agent.parameters(), lr=args.lr, eps=1e-5)
    scaler = make_grad_scaler(device, use_cuda_amp and amp_dtype == torch.float16)

    num_steps, num_envs = args.num_steps, args.num_envs
    batch_size = num_steps * num_envs
    minibatch_size = batch_size // args.num_minibatches

    obs_buf = torch.zeros(num_steps, num_envs, env.obs_dim, device=device)
    oracle_buf = torch.zeros(num_steps, num_envs, env.obs_dim, device=device)
    feats_buf = torch.zeros(
        num_steps, num_envs, env.max_actions, env.act_feat_dim, device=device
    )
    mask_buf = torch.zeros(num_steps, num_envs, env.max_actions, dtype=torch.bool, device=device)
    actions_buf = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
    logprobs_buf = torch.zeros(num_steps, num_envs, device=device)
    rewards_buf = torch.zeros(num_steps, num_envs, device=device)
    dones_buf = torch.zeros(num_steps, num_envs, device=device)
    values_buf = torch.zeros(num_steps, num_envs, device=device)

    obs, oracle_obs, feats, mask = env.reset()
    next_done = torch.zeros(num_envs, device=device)
    recent_outcomes: deque[int] = deque(maxlen=200)
    global_step = 0
    start = time.time()

    def sync_profile():
        if args.profile and device.type == "cuda":
            torch.cuda.synchronize(device)

    def begin_profile():
        sync_profile()
        return time.perf_counter()

    def add_profile(times, name, section_start):
        sync_profile()
        times[name] = times.get(name, 0.0) + time.perf_counter() - section_start

    while global_step < args.total_steps:
        profile_times = {}
        update_start = begin_profile() if args.profile else None
        for step in range(num_steps):
            section_start = begin_profile() if args.profile else None
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            oracle_t = torch.as_tensor(oracle_obs, dtype=torch.float32, device=device)
            feats_t = torch.as_tensor(feats, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(mask, device=device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_cuda_amp
            ):
                action, logprob, value = raw_agent.act(
                    obs_t, oracle_t, feats_t, mask_t
                )
            if section_start is not None:
                add_profile(profile_times, "policy", section_start)
            obs_buf[step], oracle_buf[step] = obs_t, oracle_t
            feats_buf[step], mask_buf[step] = feats_t, mask_t
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value
            dones_buf[step] = next_done

            section_start = begin_profile() if args.profile else None
            obs, oracle_obs, feats, mask, rewards, dones, outcomes = env.step(
                action.cpu().numpy()
            )
            if section_start is not None:
                add_profile(profile_times, "env_step", section_start)
            rewards_buf[step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(dones, dtype=torch.float32, device=device)
            for done, outcome in zip(dones, outcomes):
                if done:
                    recent_outcomes.append(int(outcome))
            global_step += num_envs

        # GAE. Done envs auto-reset, so the value of the post-done observation
        # belongs to the next episode and is masked out by next_done.
        section_start = begin_profile() if args.profile else None
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_cuda_amp
        ):
            next_value = raw_agent.value(
                torch.as_tensor(oracle_obs, dtype=torch.float32, device=device)
            )
        if section_start is not None:
            add_profile(profile_times, "bootstrap", section_start)

        section_start = begin_profile() if args.profile else None
        advantages = torch.zeros_like(rewards_buf)
        last_gae = torch.zeros(num_envs, device=device)
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
        b_oracle = oracle_buf.reshape(batch_size, -1)
        b_feats = feats_buf.reshape(batch_size, env.max_actions, -1)
        b_mask = mask_buf.reshape(batch_size, -1)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        if section_start is not None:
            add_profile(profile_times, "gae", section_start)

        section_start = begin_profile() if args.profile else None
        clipfracs = []
        for _ in range(args.update_epochs):
            indices = torch.randperm(batch_size, device=device)
            for mb_start in range(0, batch_size, minibatch_size):
                mb = indices[mb_start : mb_start + minibatch_size]
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda_amp):
                    logits, value = agent(
                        b_obs[mb],
                        b_oracle[mb],
                        b_feats[mb],
                        b_mask[mb],
                    )
                    logits = logits.float()
                    value = value.float()
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logprob = dist.log_prob(b_actions[mb])
                    entropy = dist.entropy()
                    logratio = new_logprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    mb_adv = b_advantages[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(
                        -mb_adv * ratio,
                        -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                    ).mean()
                    v_loss = 0.5 * (value - b_returns[mb]).pow(2).mean()
                    loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().detach()
                    )

                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(raw_agent.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(raw_agent.parameters(), args.max_grad_norm)
                    optimizer.step()
        if section_start is not None:
            add_profile(profile_times, "ppo_update", section_start)

        if recent_outcomes:
            wins = sum(1 for o in recent_outcomes if o > 0)
            n_losses = sum(1 for o in recent_outcomes if o < 0)
            win_rate = wins / len(recent_outcomes)
        else:
            win_rate, n_losses = float("nan"), 0
        sps = int(global_step / (time.time() - start))
        mean_clipfrac = torch.stack(clipfracs).mean().item() if clipfracs else float("nan")
        print(
            f"step={global_step} win_rate={win_rate:.3f} "
            f"(n={len(recent_outcomes)}, losses={n_losses}) "
            f"v_loss={v_loss.item():.3f} pg_loss={pg_loss.item():.3f} "
            f"clipfrac={mean_clipfrac:.3f} sps={sps}",
            flush=True,
        )

        section_start = begin_profile() if args.profile else None
        save_agent_state(raw_agent, run_dir / "latest.pt")
        if section_start is not None:
            add_profile(profile_times, "checkpoint", section_start)
        if args.profile and update_start is not None:
            profile_times["total_update"] = time.perf_counter() - update_start
            print(
                "profile "
                + " ".join(
                    f"{name}={profile_times.get(name, 0.0):.3f}s"
                    for name in [
                        "policy",
                        "env_step",
                        "bootstrap",
                        "gae",
                        "ppo_update",
                        "checkpoint",
                        "total_update",
                    ]
                ),
                flush=True,
            )

    save_agent_state(raw_agent, run_dir / "final.pt")
    print(f"done. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
