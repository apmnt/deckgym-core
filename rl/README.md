# deckgym RL

Reinforcement learning agent for the deckgym-core Pokémon TCG Pocket engine.

The agent plays seat 0 of a game through `deckgym.PyRlVecEnv`, a vectorized
environment implemented in Rust (`src/rl_env.rs`). Each decision the env
returns the observation plus a feature vector for every *legal* action, and
the policy scores the legal-action list directly (one logit per action, the
ygo-agent / DouZero pattern) — no global flat action space is needed.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install maturin numpy
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/maturin develop --release   # builds the deckgym extension with the rl env
```

## Train

```bash
# Phase 1: beat the random player
.venv/bin/python rl/train_ppo.py --opponent r --total-steps 2000000

# Phase 2/3: harder built-in opponents
.venv/bin/python rl/train_ppo.py --opponent v  --resume runs/<run>/latest.pt
.venv/bin/python rl/train_ppo.py --opponent e2 --resume runs/<run>/latest.pt
```

Opponent codes are the engine's player codes: `r` random, `w` weighted-random,
`aa` attach-attack, `et` end-turn, `v` value-function, `e<N>` expectiminimax
of depth N, `m` MCTS.

```bash
# Phase 4: self-play with a historical opponent pool (fictitious self-play)
.venv/bin/python rl/train_selfplay.py --resume rl/checkpoints/hidden_info_stage2.pt
```

In self-play the learner (seat 0) faces frozen policies in seat 1: a copy of
its latest weights with `--latest-prob`, otherwise a uniform sample from a
pool snapshotted every `--snapshot-every` updates. The logged win rate hovers
near 50% by construction — measure progress with `rl/eval.py` against fixed
opponents.

## Evaluate

```bash
.venv/bin/python rl/eval.py runs/<run>/latest.pt --opponent r --episodes 500
```

## Design notes

- **Observation** (flat f32, ~400 dims for a mirror match): points, turn,
  energy zones, per-slot board features (card one-hot over the match's card
  vocabulary, HP, attached energy, status, stage, tool), and per-card-ID
  counts of hands, discards, and decks. The *policy* sees only legal
  information — the opponent's hand and deck composition are zeroed (sizes
  stay visible). The *critic* reads the full-state oracle view during
  training (PerfectDou-style); evaluation uses the policy view only. Note
  the built-in engine bots (v, e2, m) do see everything, so they play with
  an information advantage over the agent.
- **Reward**: +1 win / −1 loss / 0 tie, gamma = 1.0 by default (the ByteRL
  recipe). `--shaping <c>` enables potential-based shaping on the prize-point
  differential; it telescopes to zero over an episode so the optimal policy
  is unchanged.
- **Episodes auto-reset** inside the env; ties happen at turn 30.
