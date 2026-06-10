# deckgym RL

Reinforcement learning agent for the deckgym-core Pokémon TCG Pocket engine.

The agent plays seat 0 of a game through `deckgym.PyRlVecEnv`, a vectorized
environment implemented in Rust (`src/rl_env.rs`). Each decision the env
returns the observation plus a feature vector for every *legal* action, and
the policy scores the legal-action list directly (one logit per action, the
ygo-agent / DouZero pattern) — no global flat action space is needed.

## Setup

```bash
uv venv
uv pip install maturin numpy torch
uv run maturin develop --release   # builds the deckgym extension with the rl env
```

Check that PyTorch can see your GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Train

```bash
# Phase 1: beat the random player
uv run python rl/train_ppo.py --opponent r --total-steps 2000000 --amp --compile

# Phase 2/3: harder built-in opponents
uv run python rl/train_ppo.py --opponent v  --resume runs/<run>/latest.pt --amp --compile
uv run python rl/train_ppo.py --opponent e2 --resume runs/<run>/latest.pt --amp --compile
```

The training scripts use `--device auto` by default: CUDA is selected when
PyTorch can see a CUDA device, otherwise CPU is used. Use `--device cuda` if
you want the run to fail instead of silently falling back to CPU. On Ada GPUs
such as RTX 40-series cards, `--amp` defaults to BF16 autocast and enables
TF32 matmul.

CUDA training works, but this workload may not keep the GPU near 100%
utilization. The neural network runs on the GPU, while game simulation,
legal-action generation, and action handoff still happen in the CPU-side
Rust/Python environment. If GPU utilization is low, increase `--num-envs` to
feed larger batches to the model; if CPU usage is saturated or throughput
drops, reduce it.

`PyRlVecEnv.step` parallelizes environment stepping across CPU threads with
Rayon. Rayon uses its default global thread pool unless `RAYON_NUM_THREADS` is
set:

```bash
RAYON_NUM_THREADS=12 uv run python rl/train_ppo.py --opponent e2 \
  --num-envs 128 --num-steps 256 --num-minibatches 8 \
  --device cuda --amp
```

For more throughput, increase the rollout batch until the CPU-side environment
or GPU memory becomes the limiter:

```bash
uv run python rl/train_ppo.py --opponent r --total-steps 2000000 \
  --num-envs 128 --num-steps 256 --num-minibatches 8 \
  --amp --compile --device cuda
```

Opponent codes are the engine's player codes: `r` random, `w` weighted-random,
`aa` attach-attack, `et` end-turn, `v` value-function, `e<N>` expectiminimax
of depth N, `m` MCTS.

```bash
# Phase 4: self-play with a historical opponent pool (fictitious self-play)
uv run python rl/train_selfplay.py --resume rl/checkpoints/hidden_info_stage2.pt --amp --compile
```

In self-play the learner (seat 0) faces frozen policies in seat 1: a copy of
its latest weights with `--latest-prob`, otherwise a uniform sample from a
pool snapshotted every `--snapshot-every` updates. The logged win rate hovers
near 50% by construction — measure progress with `rl/eval.py` against fixed
opponents.

## Evaluate

```bash
uv run python rl/eval.py runs/<run>/latest.pt --opponent r --episodes 500 --amp
```

## Profiling

Print per-update PPO phase timings:

```bash
uv run python rl/train_ppo.py --opponent e2 --total-steps 32768 \
  --num-envs 128 --num-steps 256 --num-minibatches 8 \
  --device cuda --amp --profile
```

Profile the Rust expectiminimax search internals:

```bash
DECKGYM_PROFILE_EXPECTIMINIMAX=1 \
DECKGYM_PROFILE_EXPECTIMINIMAX_EVERY=1000 \
uv run python rl/train_ppo.py --opponent e2 --device cuda --amp
```

The expectiminimax profile reports total decision time plus time spent in
forecasting action branches, cloning/applying branch states, generating
recursive legal actions, and evaluating leaf states.

## Architectures

Training defaults to `--arch res` (`ResAttnAgent`, ~4.8M params): a pre-norm
residual GELU trunk (`--hidden 512`, `--blocks 4`) plus self-attention across
the legal-action tokens (`--heads 4`), so actions are scored relative to each
other rather than independently. `--arch tx` (`TokenTransformerAgent`) cuts
the flat observation back into semantic tokens — globals, 8 board slots, 6
zone count vectors — and runs a TransformerEncoder over them (`--hidden 256`,
`--blocks 3` layers). `--arch mlp` is the original small network (~360k
params).

`--memory` (res/tx) inserts a GRU cell over decision steps on the policy
path, giving the agent within-game memory for hidden-information inference;
the oracle critic stays feedforward (full state ≈ Markov). With memory, PPO
minibatches become env-major sequences and the GRU is replayed over the
rollout with stored initial hidden states (BPTT); self-play opponents carry
their own per-env hidden state.

Checkpoints embed their architecture implicitly — `eval.py` and `--resume`
auto-detect arch, sizes, and memory from the state dict, so old checkpoints
keep working (`--arch`/`--hidden`/`--memory` are ignored when resuming).

When scaling further (per the upgrades guide): raise `--ent-coef` to
0.02–0.03 early in training and grow the per-update batch via `--num-envs`
before touching `--lr`.

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
