# Next Steps — GPU Runbook

The prioritized experiment queue for a CUDA machine, with exact commands,
what each run should beat, and decision gates. Everything here follows
from measured evidence in `docs/rl-agent-plan.md` (phases 10–13); read
`docs/rl-research-guide.md` first if you're new to the setup.

**Baseline to beat: `rl/checkpoints/general_m2.pt` — 41.1% [36.4, 46.0]
vs e3 on random train-pool matchups.** Milestone: >50% pooled, every seed
≥ 50% (the "beat expectiminimax-3 pool-wide" bar).

## 0. Setup on the GPU box

```bash
uv venv && uv pip install maturin numpy torch pytest
uv run maturin develop --release
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run --no-sync python -m pytest rl/tests -q     # sanity
```

Notes that will save you an evening:

- After any `maturin develop`, use **`uv run --no-sync`** for everything —
  plain `uv run` can silently re-sync a stale cached wheel over your
  fresh build.
- The GPU only runs the network. Game simulation, legal-action generation,
  and (crucially) **e3 roster opponents are CPU-bound** — a depth-3 search
  per opponent move. Give the box as many cores as you can and set
  `RAYON_NUM_THREADS=<cores>`. On 4 cores, self-play with e3 arms ran
  55–100 SPS; the GPU will not be the limiter until cores stop being one.
- Scale batch via `--num-envs` (128–256) and `--num-steps 256
  --num-minibatches 8`, add `--amp --compile`. Watch `sps` in the logs.
- Eval variance is brutal (same checkpoint: 36–53% vs e3 across seeds).
  Never accept/reject on fewer than 2 seeds x 200 episodes; the commands
  below bake that in.

## 1. Full line with keyword attributes (highest priority)

Why: the 24 keyword+evolution flags were the only phase-13 lever that
moved a number (+5.1pp at BC level, 29.9%→35.0%). They have never ridden
a full BC → PFSP line. Also scale width to 512 (the BC width ablation was
mildly positive and capacity was the suspected e3 bottleneck).

```bash
# 1a. Regenerate the feature table (engine may have new cards).
uv run --no-sync python rl/keyword_features.py --out runs/card_keywords.npy

# 1b. BC warm start (reuse the cached 489k-decision dataset if present;
#     otherwise collect fresh with --episodes 12000 — cheap on many cores).
uv run --no-sync python rl/bc_pretrain.py --arch gen --hidden 512 --blocks 3 \
  --deck rl/pools/train.pool --dataset runs/phase13_merged.pkl --epochs 4 \
  --card-text runs/card_keywords.npy --out runs/bc_kw512/bc.pt

# 1c. Long conservative PFSP with snapshot harvesting. 2M steps.
RAYON_NUM_THREADS=$(nproc) uv run --no-sync python rl/train_selfplay.py \
  --resume runs/bc_kw512/bc.pt \
  --deck rl/pools/train.pool --total-steps 2000000 \
  --bots e1,e2,e3 --latest-prob 0.3 --pfsp-power 4 \
  --lr 1e-4 --target-kl 0.02 --clip-vloss \
  --ent-coef 0.005 --ent-coef-final 0.002 \
  --num-envs 128 --amp --compile --snapshot-keep 24 \
  --run-name kw512_pfsp

# 1d. Harvest: eval EVERY snap_*.pt (and final.pt) vs e3, keep the argmax.
for ck in runs/kw512_pfsp/snap_*.pt runs/kw512_pfsp/final.pt; do
  echo "== $ck"; uv run --no-sync python rl/eval.py "$ck" \
    --deck rl/pools/train.pool --opponent e3 --episodes 200 \
    --seeds 999,4242 --amp
done
```

Gate: best snapshot pooled > 41.1% → it becomes `general_m4`; also run
the full gauntlet (r,e1,e2 + heldout.pool e1,e2,e3) before shipping.
Interleaved oversampling of skill-gap decks (see §3's matrix) onto 1c's
`--deck` spec is a known further +3pp.

## 2. Specialist distillation, done right (second priority)

Phase 13 showed the mechanism needs two fixes, both compute-hungry:

- **Specialists must have real deltas.** 120k-step legs from an already-
  oversampled base gained ~3pp on-deck (nothing to distill). Use 400k+
  legs, and *verify the delta* (≥ +8pp on-deck vs the base, 200 eps)
  before recording; drop specialists that don't clear it.
- **The merge must not discard RL gains.** Include the base generalist's
  own recorded play on ALL pool decks in the merged dataset, so the BC
  target is "the generalist, upgraded where specialists are better" —
  not a from-scratch imitation of e3.

```bash
# 2a. Recompute the skill-gap list from your current best (see §3).
# 2b. Specialists: long legs + on-deck verification.
bash rl/run_specialists.sh rl/checkpoints/<best>.pt 400000 2000 <deck1> <deck2> ...
for d in <deck1> <deck2> ...; do
  uv run --no-sync python rl/eval.py runs/spec_$d/final.pt \
    --deck example_decks/$d.txt --opponent-deck rl/pools/train.pool \
    --opponent e3 --episodes 200 --seeds 999,4242 --amp
done
# 2c. Record the BASE generalist on the full pool (the anti-forgetting term).
uv run --no-sync python rl/collect_vs_bot.py rl/checkpoints/<best>.pt \
  --deck rl/pools/train.pool --opponent-deck rl/pools/train.pool \
  --bot e3 --episodes 8000 --out runs/base_games.pkl
# 2d. Merge only verified specialists + base games, BC 4 epochs, eval, then
#     a short conservative PFSP polish (300k) with --snapshot-keep.
```

## 3. Refresh the skill-gap diagnosis (cheap, do before 1c/2a)

```bash
uv run --no-sync python rl/matchup_matrix.py rl/checkpoints/<best>.pt \
  --opponent e3 --pool rl/pools/train.pool --episodes 100
# e3's own per-deck baseline (run once, it changes only with the engine):
# same command with an e3-vs-e3 protocol — see docs/rl-agent-plan.md
# phase 12 for the seat-0 reference numbers (mean 47.5%).
```

Target decks = largest (our rate − e3's rate with the same deck), not
lowest absolute rate.

## 4. Learned policy-view value head → honest test-time search

Determinized 1-ply search *hurt* with the engine's hand-crafted value
function (e1-grade judgment diluting a stronger policy). The honest
version needs a value function that (a) reads the policy view and (b) is
at least as good as the policy. Smallest path:

1. Add a second value head trained on the *policy view* alongside the
   oracle critic (one extra regression term in BC and PPO — small diff in
   `rl/agent.py` / `rl/bc_pretrain.py` / `rl/train_selfplay.py`).
2. In `rl/eval_search.py`, replace the engine value with that head
   applied to the *observation after each determinized action* — needs a
   `forecast_obs` variant of `RlEnvCore::action_values` returning
   post-action observations instead of engine values (moderate Rust diff).
3. Re-run the beta sweep; gate on beating the plain policy at equal
   episodes.

This is the only route to strength gains without any retraining once the
head exists — and the head also enables AlphaZero-style deeper search
later (the phase-6 plan's PUCT idea).

## 5. Exploiter arm against your own best (robustness audit)

Learning-to-Beat-ByteRL showed self-play CCG agents hide 60-70%
exploits. Quantify yours (and harvest the exploiter's games as training
data — they are, by construction, lines your agent mishandles):

```bash
uv run --no-sync python rl/train_selfplay.py --resume rl/checkpoints/<best>.pt \
  --deck rl/pools/train.pool --total-steps 600000 \
  --frozen-opponents rl/checkpoints/<best>.pt --latest-prob 0.15 \
  --lr 1.5e-4 --target-kl 0.03 --clip-vloss --num-envs 128 --amp \
  --run-name exploit_best
uv run --no-sync python rl/head_to_head.py runs/exploit_best/final.pt \
  rl/checkpoints/<best>.pt --deck rl/pools/train.pool --episodes 400 --both
```

If the exploiter wins big, feed its recorded games back through the §2
merge; if it can't (like the mirror champion at ~49-51%), that is itself
a strong robustness result worth documenting.

## 6. Stretch: architecture and scale

Only after 1–5, and one variable at a time (the ablation discipline in
`docs/rl-agent-plan.md` exists because everything else measured flat):

- `--hidden 768 --blocks 4` with proportional `--num-envs` (capacity was
  the suspected BC ceiling; GPU makes this free to try).
- Per-card *token* observations (hand/board as card tokens instead of
  zone count vectors) — needs the Rust env to emit token layouts; the
  biggest representation swing available, and the tx-arch's real purpose.
- BC from an `e4` demonstrator (e3→e4 measured ≈ +6pp bot strength;
  collection is pure CPU, so overlap it with GPU training).

## Reporting

For any claimed improvement: pooled ≥ 2 seeds x 200 episodes vs e3 on
the unweighted train pool + the same on heldout.pool, Wilson CIs, and a
row appended to `docs/rl-agent-plan.md`. Keep checkpoints named
`general_m<N>.pt` and update the README table only on a confirmed win.
