#!/bin/bash
# RL ablation grid: 200k-step PFSP fine-tunes from a fixed BC warm start,
# one variable per cell, two-seed e3 eval each.
set -u
PY=.venv/bin/python
BC=runs/abl_base_res256/bc.pt
BASE="--total-steps 200000 --num-envs 32 --bots e1,e2,e3 --latest-prob 0.3 --pfsp-power 4 --ent-coef-final 0.003 --device cpu"
EVAL() {
  for s in 999 4242; do
    $PY rl/eval.py "runs/abl_$1/final.pt" --opponent e3 --episodes 200 --num-envs 8 --seed $s --device cpu \
      | sed "s|^|[$1] |"
  done
}
CELL() { # name extra-args... (extra args override BASE because argparse takes the last value)
  name=$1; shift
  echo "=== cell $name ==="
  $PY rl/train_selfplay.py $BASE --resume $BC --run-name "abl_$name" --seed 71 "$@" 2>&1 | tail -1
  EVAL "$name"
}

CELL rl_base
CELL shaping01      --shaping 0.1
CELL shaping02      --shaping 0.2
CELL ent_fixed      --ent-coef 0.01 --ent-coef-final 0.01
CELL ent_hi_anneal  --ent-coef 0.02 --ent-coef-final 0.005
CELL uniform_roster --pfsp-power 0
CELL pool8_snap5    --pool-size 8 --snapshot-every 5
CELL latest_only    --latest-prob 1.0
echo "=== cell memory_ft ==="
$PY rl/train_selfplay.py $BASE --resume runs/abl_mem_res256/bc.pt --run-name abl_memory_ft --seed 71 2>&1 | tail -1
EVAL memory_ft
echo "RL-ABLATION-DONE"
