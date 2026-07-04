#!/bin/bash
# Focused training legs: short PFSP runs with seat 0 locked to one weak deck
# (diagnosed by rl/matchup_matrix.py) and seat 1 sampling the full pool,
# e3 anchoring the roster. Each leg resumes the previous leg's final
# checkpoint; conservative KL protects the other matchups (measured: 950k
# mirror-only steps cost no pool generality). Usage:
#   bash rl/run_focus_legs.sh <start_checkpoint> <steps_per_leg> deck1 deck2 ...
set -euo pipefail
cd "$(dirname "$0")/.."

RESUME="$1"; STEPS="$2"; shift 2

for deck in "$@"; do
  name="focus_$(basename "$deck" .txt)"
  echo "=== leg $name (resume $RESUME) ==="
  RAYON_NUM_THREADS=4 uv run python rl/train_selfplay.py --resume "$RESUME" \
    --deck "example_decks/$deck.txt" --opponent-deck rl/pools/train.pool \
    --total-steps "$STEPS" --bots e3 --latest-prob 0.25 --pfsp-power 4 \
    --lr 1e-4 --target-kl 0.02 --clip-vloss \
    --ent-coef 0.004 --ent-coef-final 0.002 \
    --run-name "$name"
  RESUME="runs/$name/final.pt"
done
echo "ALL LEGS DONE: final checkpoint $RESUME"
