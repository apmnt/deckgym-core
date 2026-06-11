"""PPO training loop for the deckgym RL environment (CleanRL-style).

The policy acts on hidden-information observations; the critic reads the
full-state (oracle) observation, which exists only at training time.

Example:
    python rl/train_ppo.py --opponent r --total-steps 2000000
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agent import make_agent
from env_wrapper import VecEnv
from torch_runtime import (
    agent_from_checkpoint,
    configure_device,
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
    p.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="stop the update early when approx KL exceeds this",
    )
    p.add_argument(
        "--clip-vloss", action="store_true", help="PPO value clipping around old predictions"
    )
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument(
        "--ent-coef-final",
        type=float,
        default=None,
        help="linearly anneal entropy coef to this value over the run",
    )
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--shaping", type=float, default=0.0, help="potential-based shaping coef")
    p.add_argument(
        "--arch", choices=["res", "tx", "mlp"], default="res", help="network architecture"
    )
    p.add_argument("--hidden", type=int, default=None, help="trunk width (default: arch default)")
    p.add_argument(
        "--blocks", type=int, default=None, help="residual blocks / transformer layers"
    )
    p.add_argument("--heads", type=int, default=4, help="attention heads")
    p.add_argument(
        "--memory", action="store_true", help="GRU over decision steps (res/tx arch)"
    )
    p.add_argument(
        "--oracle",
        action="store_true",
        help="all-knowing policy: feeds the full-state view to the policy too",
    )
    p.add_argument(
        "--aux-belief",
        type=float,
        default=0.0,
        help="weight of the opponent-hand prediction auxiliary loss (res/tx arch)",
    )
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
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    env = VecEnv(
        args.deck,
        args.opponent_deck or args.deck,
        opponent=args.opponent,
        num_envs=args.num_envs,
        seed=args.seed,
        shaping_coef=args.shaping,
    )
    if args.resume:
        # The checkpoint dictates the architecture; --arch/--hidden are ignored.
        raw_agent = agent_from_checkpoint(
            args.resume, env.obs_dim, env.act_feat_dim, map_location=device
        ).to(device)
        args.memory = getattr(raw_agent, "gru", None) is not None
        args.oracle = raw_agent.config.get("oracle", False)
        if args.aux_belief > 0 and getattr(raw_agent, "belief_head", None) is None:
            print("checkpoint has no belief head; disabling --aux-belief")
            args.aux_belief = 0.0
        print(f"resumed weights from {args.resume} ({type(raw_agent).__name__})")
    else:
        raw_agent = make_agent(
            env.obs_dim,
            env.act_feat_dim,
            args.arch,
            args.hidden,
            args.blocks,
            args.heads,
            memory=args.memory,
            belief=args.aux_belief > 0,
            oracle=args.oracle,
        ).to(device)
    if args.oracle and args.aux_belief > 0:
        print("oracle policy sees everything; disabling --aux-belief")
        args.aux_belief = 0.0
    if args.oracle:
        print("ORACLE mode: policy receives the full-state view")
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
    h = raw_agent.initial_state(num_envs, device)
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
        # Hidden state at rollout start, replayed during the BPTT update.
        h0 = h.clone() if h is not None else None
        for step in range(num_steps):
            section_start = begin_profile() if args.profile else None
            pol_view = oracle_obs if args.oracle else obs
            obs_t = torch.as_tensor(pol_view, dtype=torch.float32, device=device)
            oracle_t = torch.as_tensor(oracle_obs, dtype=torch.float32, device=device)
            feats_t = torch.as_tensor(feats, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(mask, device=device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_cuda_amp
            ):
                action, logprob, value, h = raw_agent.act(
                    obs_t, oracle_t, feats_t, mask_t, h
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
            if h is not None:
                # Memory belongs to an episode: zero it where one just ended.
                # clone() also escapes inference mode for the h0 snapshot.
                h = h.clone() * (1.0 - next_done).unsqueeze(-1)
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
        b_values = values_buf.reshape(-1)
        if section_start is not None:
            add_profile(profile_times, "gae", section_start)

        section_start = begin_profile() if args.profile else None
        clipfracs = []
        kls = []
        kl_stop = False
        envs_per_mb = max(1, num_envs // args.num_minibatches)
        if args.ent_coef_final is not None:
            progress = min(1.0, global_step / args.total_steps)
            ent_coef = args.ent_coef + (args.ent_coef_final - args.ent_coef) * progress
        else:
            ent_coef = args.ent_coef
        for _ in range(args.update_epochs):
            if args.memory:
                # Sequence (env-major) minibatches: the GRU must be replayed
                # over whole rollouts, so we shuffle envs, not flat steps.
                perm = torch.randperm(num_envs, device=device)
                mb_list = [perm[s : s + envs_per_mb] for s in range(0, num_envs, envs_per_mb)]
            else:
                indices = torch.randperm(batch_size, device=device)
                mb_list = [
                    indices[s : s + minibatch_size] for s in range(0, batch_size, minibatch_size)
                ]
            for mb in mb_list:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda_amp):
                    if args.memory:
                        logits = raw_agent.policy_logits_seq(
                            obs_buf[:, mb],
                            feats_buf[:, mb],
                            mask_buf[:, mb],
                            h0[mb] if h0 is not None else None,
                            dones_buf[:, mb],
                        )
                        value = raw_agent.value(
                            oracle_buf[:, mb].reshape(logits.shape[0], -1)
                        )
                        mb_actions = actions_buf[:, mb].reshape(-1)
                        mb_logprobs = logprobs_buf[:, mb].reshape(-1)
                        mb_adv = advantages[:, mb].reshape(-1)
                        mb_returns = returns[:, mb].reshape(-1)
                        mb_values = values_buf[:, mb].reshape(-1)
                    else:
                        logits, value = agent(
                            b_obs[mb],
                            b_oracle[mb],
                            b_feats[mb],
                            b_mask[mb],
                        )
                        mb_actions = b_actions[mb]
                        mb_logprobs = b_logprobs[mb]
                        mb_adv = b_advantages[mb]
                        mb_returns = b_returns[mb]
                        mb_values = b_values[mb]
                    logits = logits.float()
                    value = value.float()
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logprob = dist.log_prob(mb_actions)
                    entropy = dist.entropy()
                    logratio = new_logprob - mb_logprobs
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(
                        -mb_adv * ratio,
                        -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                    ).mean()
                    if args.clip_vloss:
                        v_clipped = mb_values + (value - mb_values).clamp(
                            -args.clip_coef, args.clip_coef
                        )
                        v_loss = 0.5 * torch.max(
                            (value - mb_returns).pow(2), (v_clipped - mb_returns).pow(2)
                        ).mean()
                    else:
                        v_loss = 0.5 * (value - mb_returns).pow(2).mean()
                    loss = pg_loss - ent_coef * entropy.mean() + args.vf_coef * v_loss
                    if args.aux_belief > 0:
                        if args.memory:
                            belief_obs = obs_buf[:, mb].reshape(-1, env.obs_dim)
                            belief_oracle = oracle_buf[:, mb].reshape(-1, env.obs_dim)
                        else:
                            belief_obs, belief_oracle = b_obs[mb], b_oracle[mb]
                        loss = loss + args.aux_belief * raw_agent.aux_belief_loss(
                            belief_obs, belief_oracle
                        )
                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().detach()
                    )
                    kls.append(approx_kl.detach())

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
                if args.target_kl is not None and approx_kl.item() > args.target_kl:
                    kl_stop = True
                    break
            if kl_stop:
                break
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
        mean_kl = torch.stack(kls).mean().item() if kls else float("nan")
        with torch.no_grad():
            var_returns = returns.var()
            explained_var = (
                float("nan")
                if var_returns.item() == 0
                else (1.0 - (returns - values_buf).var() / var_returns).item()
            )
        print(
            f"step={global_step} win_rate={win_rate:.3f} "
            f"(n={len(recent_outcomes)}, losses={n_losses}) "
            f"v_loss={v_loss.item():.3f} pg_loss={pg_loss.item():.3f} "
            f"clipfrac={mean_clipfrac:.3f} kl={mean_kl:.4f} ev={explained_var:.2f} sps={sps}",
            flush=True,
        )
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "global_step": global_step,
                        "win_rate": win_rate,
                        "v_loss": v_loss.item(),
                        "pg_loss": pg_loss.item(),
                        "clipfrac": mean_clipfrac,
                        "approx_kl": mean_kl,
                        "explained_variance": explained_var,
                        "ent_coef": ent_coef,
                        "sps": sps,
                    }
                )
                + "\n"
            )

        section_start = begin_profile() if args.profile else None
        # Throttled: synchronous serialization in the hot loop is wasted I/O.
        if (global_step // (num_steps * num_envs)) % 5 == 0:
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
