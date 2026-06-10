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
