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

## Evaluate

```bash
.venv/bin/python rl/eval.py runs/<run>/latest.pt --opponent r --episodes 500
```

## Design notes

- **Observation** (flat f32, ~420 dims for a mirror match): points, turn,
  energy zones, per-slot board features (card one-hot over the match's card
  vocabulary, HP, attached energy, status, stage, tool), and per-card-ID
  counts for both hands, discards, and decks. The engine is fully observable
  except deck order, and the built-in bots see everything too, so the agent
  observes the opponent's hand as well.
- **Reward**: +1 win / −1 loss / 0 tie, gamma = 1.0 by default (the ByteRL
  recipe). `--shaping <c>` enables potential-based shaping on the prize-point
  differential; it telescopes to zero over an episode so the optimal policy
  is unchanged.
- **Episodes auto-reset** inside the env; ties happen at turn 30.
