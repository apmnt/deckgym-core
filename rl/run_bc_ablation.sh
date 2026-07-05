#!/bin/bash
# BC distillation ablation grid: shared e3 dataset, one variable per cell.
set -u
PY=.venv/bin/python
DS=/tmp/bc_e3_dataset.pkl
EVAL() { # name ckpt
  for s in 999 4242; do
    $PY rl/eval.py "$2" --opponent e3 --episodes 200 --num-envs 8 --seed $s --device cpu \
      | sed "s|^|[$1] |"
  done
}
CELL() { # name extra-args...
  name=$1; shift
  echo "=== cell $name ==="
  $PY rl/bc_pretrain.py --bot e3 --episodes 2500 --num-envs 32 --epochs 5 \
    --dataset $DS --out "runs/abl_$name/bc.pt" --device cpu "$@" 2>&1 | tail -2
  EVAL "$name" "runs/abl_$name/bc.pt"
}

CELL base_res256                                  # collects + caches the dataset
CELL arch_mlp        --arch mlp --hidden 256
CELL arch_tx         --arch tx --hidden 256 --blocks 3
CELL width_res384    --hidden 384
CELL width_res512    --hidden 512
CELL heads8_res256   --heads 8
CELL belief005       --aux-belief 0.05
CELL belief01        --aux-belief 0.1
CELL belief025       --aux-belief 0.25
CELL mem_res256      --memory
echo "ABLATION-GRID-DONE"
