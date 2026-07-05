#!/bin/bash
# Specialist-distillation pipeline (phase 13): for each skill-gap deck,
# train a short throwaway specialist from the general checkpoint (seat 0
# locked to the deck, e3 anchoring the roster), then record its games
# vs e3 into a BC dataset. The datasets are merged with the multi-deck
# e3 dataset by supervised distillation afterwards — specialists erode
# under further RL, but their recorded lines survive BC merging (the
# mechanism that produced general_v2's champion property).
# Usage: bash rl/run_specialists.sh <start_ckpt> <steps> <episodes> deck1 ...
set -euo pipefail
cd "$(dirname "$0")/.."

RESUME="$1"; STEPS="$2"; EPISODES="$3"; shift 3

for deck in "$@"; do
  name="spec_$(basename "$deck" .txt)"
  echo "=== specialist $name (from $RESUME, $STEPS steps) ==="
  RAYON_NUM_THREADS=4 uv run --no-sync python rl/train_selfplay.py --resume "$RESUME" \
    --deck "example_decks/$deck.txt" --opponent-deck rl/pools/train.pool \
    --total-steps "$STEPS" --bots e3 --latest-prob 0.25 --pfsp-power 4 \
    --lr 1e-4 --target-kl 0.02 --clip-vloss \
    --ent-coef 0.004 --ent-coef-final 0.002 \
    --run-name "$name"
  echo "=== recording $name games vs e3 ==="
  RAYON_NUM_THREADS=4 uv run --no-sync python rl/collect_vs_bot.py "runs/$name/final.pt" \
    --deck "example_decks/$deck.txt" --opponent-deck rl/pools/train.pool \
    --bot e3 --episodes "$EPISODES" --out "runs/${name}_games.pkl"
done
echo "ALL SPECIALISTS DONE"
