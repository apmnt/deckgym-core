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
remain usable.

Implemented on top of it:

- `--memory`: GRU cell over decision steps (policy path only — the oracle
  critic sees the full, near-Markov state and stays feedforward). Training
  uses env-major sequence minibatches with BPTT over the rollout; frozen
  self-play opponents keep per-env hidden states, zeroed at episode ends.
  This is the lever that lets the policy *infer* the opponent's hidden hand
  from observed plays rather than reacting to snapshots.
- `--arch tx` (`TokenTransformerAgent`): structured encoder — the flat
  observation is sliced back into 15 semantic tokens (1 globals, 8 board
  slots with shared projection + position embeddings, 6 zone count vectors
  with zone embeddings) and a 3-layer TransformerEncoder reasons over them.
  Within a fixed mirror match its payoff over `res` is uncertain; its real
  value is multi-deck generalization, where shared card/zone projections
  transfer across matchups. Per-card hand/board tokens (instead of zone
  count vectors) would need the Rust env to emit token observations — the
  natural next step if multi-deck training begins.

## Beating expectiminimax-3 (phase 7 experiment grid)

Goal: >50% vs `e3` (depth-3 search, sees the agent's hand). All e3 numbers
are greedy evals pooled over fixed seeds; ±3.5–5pp at the listed sample
sizes. Baseline going in: `selfplay_3m` at ~33.5% vs e3.

| Combination | vs e2 | vs e3 | Verdict |
|---|---|---|---|
| A: PFSP p=2 + bots e1,e2,e3 (from selfplay_3m) | 55% | ~42% @700k, drifted to ~38% @1.2M | PFSP+bots works vs e2; e3 drifts without sharper focus |
| A2: PFSP p=4, latest-prob 0.3 (from A@700k) | 61.5% | ~40.5% (400 eps) | best e2; e3 plateau ~40% — small net capacity-bound |
| BC imitator of e3 (2500 games, 12 min, no RL) | 51.7% | **~45.3%** (400 eps) | best cost/benefit step in the project |
| **C: BC → PFSP p=4 + bots, entropy annealed, 800k** | **60.3%** | **54.2% (801 eps, all 4 seeds >50%)** | **goal met** — `rl/checkpoints/bc_pfsp_champion.pt` |

Champion gauntlet (res 256×2, ~1.9M params): 99% vs r, 96.5% vs v,
87.5% vs e1, 60.3% vs e2, 54.2% vs e3 — all while playing blind against
search opponents that see its hand.

## Ablation slate (phase 9)

Two grids, both on the venusaur-exeggutor mirror. BC grid: 5-epoch
distillation cells on one shared 70k-decision e3 dataset, evaluated vs e3
(400 episodes pooled, ±5pp). RL grid: 200k-step PFSP fine-tunes from the
same BC checkpoint, one variable per cell (reference cell 46.0%).

| Family | Cells (vs e3) | Verdict |
|---|---|---|
| Architecture (BC) | **res 45.3%**, tx 37.8%, mlp 26.5% | dominant factor; action-attention residual net is worth ~18pp |
| Width (BC) | 384: 44.8%, **512: 47.5%** | mildly positive |
| Heads (BC) | 8: 47.0% | mildly positive |
| Belief loss (BC) | 0.05/0.1/0.25: 44.0–46.0% | neutral |
| Memory | BC stateless: 37.8% → RL fine-tune: 45.8% | recovers but no gain; ~40% slower |
| Shaping (RL) | 0.1: 46.0%, 0.2: 44.5% | neutral |
| Entropy schedule (RL) | fixed: 45.3%, hi-anneal: 45.5% | neutral |
| Curriculum (RL) | PFSP p4: 46.0%, uniform bots: 49.5%, **latest-only: 43.0%** | bots in the roster are the active ingredient; PFSP vs uniform within noise at this scale |
| Pool config (RL) | pool 8/snap 5: 49.5% | within noise |

Conclusion: in this regime nearly all the variance lives in
**architecture, the BC warm start, and bot-anchored opponent rosters** —
the PPO dials (shaping, entropy, pool sizing, PFSP exponent) are flat
within ±3pp. The PFSP-vs-uniform comparison may differ at full 800k-step
scale, where Combo A showed drift away from e3 without prioritization.

## Oracle (all-knowing) agent experiment (phase 8)

An information-matched counterpart to the engine bots: `--oracle` feeds the
full-state view to the policy too (checkpoint:
`rl/checkpoints/oracle_pfsp.pt`). Same recipe as the champion (BC from the
cached e3 dataset, then 800k PFSP steps with e1/e2/e3 in the roster).

| Agent | vs e2 | vs e3 (801 eps) |
|---|---|---|
| Honest champion (hidden info) | 60.3% | **54.2%** [50.8, 57.6] |
| Oracle (sees everything) | 52.6% | 45.2% [41.8, 48.7] |

Findings (one run each — interpret with care):

- **Full information did not help.** The oracle BC imitator matched the
  honest one (44.5% vs 45.3% — e3's moves are as predictable from the
  legal view), and the oracle's RL fine-tune then plateaued (+1pp; its
  450k and 800k checkpoints are statistically identical) while the honest
  fine-tune climbed +9pp. The hidden-information bottleneck appears to act
  as a regularizer: the honest policy trains on a smaller input manifold
  with the same capacity and budget.
- The oracle run survived a host kill + resume (optimizer/pool reset at
  450k); ruled out as the cause by evaluating the pre-restart checkpoint.

Lessons so far:

- **Behavior cloning from the target bot is the highest-leverage step**:
  distilling 2500 e3-vs-e3 games (12 minutes, `rl/bc_pretrain.py`) beat
  every pure-RL combination tried before it, and PFSP fine-tuning from
  that start added another ~+9pp vs e3. Imitating an oracle demonstrator
  from hidden-information inputs is PTIE distillation and it works.
- PFSP with engine bots in the roster reliably converts training into
  wins against the *targeted* bot (e2: 49% → 61.5%), but without a strong
  enough starting policy it plateaus below parity on e3 — and plain
  (non-prioritized) training *drifts away* from the hardest opponent.
- Evaluation variance vs e3 is large: 200-episode single-seed evals
  ranged 36–53% for the *same* checkpoint. Claim wins only on multi-seed
  pooled evals (we used 4×200).

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
