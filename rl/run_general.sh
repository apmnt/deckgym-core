#!/bin/bash
# Deck-general agent pipeline: BC warm start from expectiminimax-3 games
# sampled across all training-pool pairings, then PFSP self-play fine-tune
# with engine bots in the roster, then eval on pool and held-out matchups.
# See rl/README.md ("Deck-general agent") for background.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="uv run python"
EPISODES="${BC_EPISODES:-6000}"
STEPS="${RL_STEPS:-800000}"

# 1. Behavior-cloning warm start (dataset cached for reuse).
$PY rl/bc_pretrain.py --arch gen --hidden 384 --blocks 3 \
  --deck rl/pools/train.pool --bot e3 --episodes "$EPISODES" \
  --epochs 5 --dataset runs/bc_gen_dataset.pkl --out runs/bc_gen/bc.pt

# 2. Quick BC gauntlet (in-pool random matchups).
$PY rl/eval.py runs/bc_gen/bc.pt --deck rl/pools/train.pool \
  --opponent r,e1,e3 --episodes 100 --seeds 999,4242

# 3. PFSP self-play fine-tune, engine bots anchoring the roster.
$PY rl/train_selfplay.py --resume runs/bc_gen/bc.pt \
  --deck rl/pools/train.pool --total-steps "$STEPS" \
  --bots e1,e2,e3 --latest-prob 0.3 --pfsp-power 4 \
  --ent-coef-final 0.003 --run-name gen_pfsp

# 4. Final evals: pool matchups and zero-shot held-out decks.
$PY rl/eval.py runs/gen_pfsp/final.pt --deck rl/pools/train.pool \
  --opponent r,e1,e2,e3 --episodes 200 --seeds 999,4242
$PY rl/eval.py runs/gen_pfsp/final.pt --deck rl/pools/heldout.pool \
  --opponent e1,e2,e3 --episodes 200 --seeds 999,4242
