# RL Agent Plan

Goal: train a reinforcement learning agent for deckgym-core, starting with a
single mirror match (venusaur-exeggutor vs itself), beating progressively
stronger built-in opponents, then self-play.

## Prior art the design borrows from

- **ygo-agent** (Yu-Gi-Oh, PPO self-play): policy scores the *legal-action
  list* (one logit per action, masked) instead of a global action head;
  observations are per-card tokens; trained random-opponent → self-play.
- **ByteRL** (Hearthstone, beat a top-10 human): pure ±1 win/loss reward with
  γ = 1.0; fictitious-play opponent pool (add checkpoint at ~55% win rate).
- **DouZero**: action-as-input scoring sidesteps a 10^4 action head.
- **friday-james/tcg-pocket-rl**: existing PTCGP precedent (MaskablePPO,
  flat ~512 actions) — works, but flat spaces are deck-specific; we use the
  action-scorer instead.

## Why this game is friendlier than most CCGs

The engine is **fully observable except deck order** (both hands visible to
all `Player`s). The agent, however, is trained to play the *real* game: its
policy observes only legal information (opponent hand/deck composition
hidden, sizes visible), while the critic reads the full-state oracle view
during training — PerfectDou's perfect-training-imperfect-execution.
Consequences:

- The built-in ValueFunction/Expectiminimax bots see everything and thus
  play with an information edge over the agent; beating them is a stronger
  result than parity suggests.
- AlphaZero-style MCTS with chance nodes would be sound for the *open-hand*
  variant of this engine; for the hidden-hand policy it would need
  determinization (IS-MCTS). Deferred either way: every strong CCG agent
  shipped model-free policy-gradient first, and search multiplies per-move
  cost.

## Architecture

- `src/rl_env.rs` — `RlEnvCore`: single-agent view (agent = seat 0), opponent
  bot plays inside `step`, forced single-action states auto-applied, episodes
  auto-reset. Pure Rust, unit-tested.
- `PyRlVecEnv` (python_bindings) — vectorized wrapper returning flat numpy
  arrays: observation, per-legal-action features (padded to 128 slots), legal
  counts, rewards, dones, outcomes.
- `rl/` — PyTorch side: `agent.py` (action-scorer policy + value head),
  `train_ppo.py` (CleanRL-style PPO), `eval.py`, `env_wrapper.py`.

Observation (~400 f32 for a mirror): points, turn, energy zones, per-slot
board features (card one-hot over the match card vocabulary, HP, energy,
status, stage, tool), per-card-ID counts of hands/discards/decks.

Action features (~50 f32 per legal action): coarse action-class one-hot,
referenced-card one-hot, target slot + side, attack index, energy type,
amount scalars.

Reward: ±1 win/loss, 0 tie, γ = 1.0. Optional potential-based shaping on the
point differential (`--shaping`), telescopes to zero per episode.

## Training schedule

| Phase | Opponent | Gate |
|---|---|---|
| 1 | `r` random | >90% win |
| 2 | `w`/`aa` heuristics | sanity |
| 3 | `v` value-function | >60% win |
| 4 | `e2`/`e3` expectiminimax | >55% win (slow: mostly eval) |
| 5 | self-play, historical pool | Elo curve vs fixed gates |
| 6 (opt) | AZ-style PUCT, PPO net as priors | beats phase-5 agent |

Throughput on 4 CPU cores: env alone ~15k agent-steps/s (32 envs,
single-threaded); with the network in the loop ~1k SPS. Self-play scale
benefits from more cores; the code stays CPU-friendly (small MLPs).

## Measurements (this machine, 4 cores)

- Random vs random mirror: ~9% win / ~11% loss / ~80% tie (turn-30 limit),
  ~42 agent decisions per episode.

## Results — hidden-information agent, venusaur-exeggutor mirror

Curriculum: 400k steps vs `r` → 250k vs `v` (cut at plateau) → 3M steps of
fictitious self-play (pool of 20 snapshots, latest-prob 0.5), resumed from
the stage-2 weights. Greedy-policy evals; e2 numbers pool 364–400 episodes
across seeds. Checkpoints: `rl/checkpoints/hidden_info_stage2.pt` (before
self-play) and `rl/checkpoints/selfplay_3m.pt` (final, best).

| Opponent | stage 2 | + self-play | Notes |
|---|---|---|---|
| `r` random | 100% | 100% | |
| `v` value-function | ~96% | ~97% | opponent sees the agent's hand; agent does not |
| `e2` expectiminimax-2 | ~44% | **~49%** | parity with an open-hand searcher |

## Network scaling (phase 6)

`ResAttnAgent` (default `--arch res`, ~4.8M params vs ~360k for the original
MLP): pre-norm residual GELU trunk (512×4 blocks) for both the policy and
oracle-critic encoders, plus multi-head self-attention over the legal-action
tokens — each action is scored in the context of the other available actions
(combo/tempo comparisons the independent scorer couldn't express). Checkpoint
architecture is auto-detected on load, so the committed small-net checkpoints
remain usable. The next architectural lever after this is recurrent memory
(GRU over decision steps) for opponent-hand inference, which requires
per-env hidden-state plumbing through the rollout and the self-play opponent
pool.

Lessons so far:

- The stage-3 fine-tune *directly against e2* did not improve on the stage-2
  checkpoint (paired-seed evals put it ~4pp worse) — sparse ±1 rewards
  against a much stronger opponent stall. **Self-play with a historical pool
  is what worked**: +5pp vs e2 (44% → 49%) with no forgetting vs r/v, while
  the in-training self-play win rate stayed near 50% throughout (as it
  must). The midpoint probe at 1.4M steps was still flat vs e2 — the gain
  materialized in the second half; don't judge self-play runs early.
- Hiding the opponent's hand barely slowed early learning (stage 1 reached
  99% vs random on the same schedule as the open-hand run), and the oracle
  critic carried stage 2 to ~96% vs `v`.
- Training SPS on 4 CPU cores: ~900 vs `r`/`v`, ~360 vs `e2` (opponent runs
  a depth-2 search per move).
