# deckgym RL

Reinforcement learning agent for the deckgym-core Pokémon TCG Pocket engine.

The agent plays seat 0 of a game through `deckgym.PyRlVecEnv`, a vectorized
environment implemented in Rust (`src/rl_env.rs`). Each decision the env
returns the observation plus a feature vector for every *legal* action, and
the policy scores the legal-action list directly (one logit per action, the
ygo-agent / DouZero pattern) — no global flat action space is needed.

**Current best (honest agent, venusaur-exeggutor mirror)**: 99% vs random,
96.5% vs value-function, 87.5% vs expectiminimax-1, 60.3% vs e2, **54.2% vs
e3** (801 episodes, 4 seeds) — playing blind against search bots that see
its hand. Checkpoint: `rl/checkpoints/bc_pfsp_champion.pt`. Full experiment
history and lessons: `docs/rl-agent-plan.md`.

## Deck-general agent (`--arch gen`)

The mirror-match agents above are matchup specialists: their observation
one-hots cards over the *match's* vocabulary, so the input meaning changes
with the decks. `--arch gen` is the deck-general agent — **one network that
plays any deck against any deck**.

**Current best general agent** (`rl/checkpoints/general_m2.pt`, ~3.1M
params; greedy, pooled 2 seeds x 200 episodes, random matchups):

| Opponent | Train pool (25 decks) | Held-out decks (zero-shot) |
|---|---|---|
| `r` random | 98.3% | — |
| `e1` expectiminimax-1 | **79.5%** | **74.3%** |
| `e2` expectiminimax-2 | 48.5% | 36.2% |
| `e3` expectiminimax-3 | **41.1%** | 24.7% |

(r/e1 and held-out e1/e2 rows measured on its parent `general_pfsp.pt`;
the burst changed only e2/e3-relevant play. `general_v2.pt` is the
variant that beats the single-deck mirror champion 54.2% head-to-head at
33% pool e3 — see phases 11-12 in `docs/rl-agent-plan.md`.)

It plays blind (hidden opponent hand/deck) against search bots that see
everything, across all 625 train-pool pairings with a single set of
weights — and transfers most of that strength to decks it never trained
on. The BC-only warm start (`rl/checkpoints/general_bc.pt`) sits at
43.8%/36.1% vs e2/e3 in-pool. Full experiment log: `docs/rl-agent-plan.md`
phase 10.

- The env pads every card-vocabulary section to a fixed 40 slots and appends
  each slot's *global* card id to the observation, so observation/action
  dims are identical for every matchup (`src/rl_env.rs::VOCAB_SIZE`).
- `GeneralAgent` (rl/agent.py) looks those ids up in a `CardEncoder`: a
  learned identity embedding over the engine's full card index **plus a
  projection of the card's static attributes** (HP, type, stage, attack
  damage/costs, trainer kind — `deckgym.card_attr_table()`). Every local
  one-hot/count section is projected into embedding space
  (`counts @ card_repr`) before the usual residual trunk + action-attention
  scorer. Identity embeddings capture what training saw; attribute features
  carry semantics to rarely-seen and *unseen* cards. This is the card-token
  pattern of ygo-agent (Yu-Gi-Oh) and Cardsformer (Hearthstone).
- Deck specs everywhere (`--deck`, `--opponent-deck`) now accept a file, a
  folder, a comma-separated list, or a `.pool` list file; each episode
  samples one deck per seat, so training mixes all pairings.
  `rl/pools/train.pool` (25 decks) and `rl/pools/heldout.pool` (4 decks kept
  out of training, for zero-shot eval) define the standard split.
- Decklists are treated as public information (like the metagame): both
  seats' vocabularies are visible, but the opponent's hand and remaining
  deck contents stay hidden exactly as before. Engine search bots still see
  strictly more than the agent.

Train it with the same proven recipe, just on the deck pool:

```bash
# 1. BC warm start: distill e3-vs-e3 games sampled across all pool pairings.
uv run python rl/bc_pretrain.py --arch gen --deck rl/pools/train.pool \
  --bot e3 --episodes 6000 --dataset runs/bc_gen_dataset.pkl --out runs/bc_gen/bc.pt

# 2. PFSP self-play fine-tune with engine bots anchoring the roster.
uv run python rl/train_selfplay.py --resume runs/bc_gen/bc.pt \
  --deck rl/pools/train.pool --total-steps 800000 \
  --bots e1,e2,e3 --latest-prob 0.3 --pfsp-power 4 --ent-coef-final 0.003

# 3. Evaluate on random pool matchups and on never-seen decks.
uv run python rl/eval.py runs/<run>/latest.pt --deck rl/pools/train.pool \
  --opponent r,e1,e2,e3 --episodes 200 --seeds 999,4242
uv run python rl/eval.py runs/<run>/latest.pt --deck rl/pools/heldout.pool \
  --opponent e3 --episodes 200 --seeds 999,4242
```

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

The proven recipe (what produced the champion) is **behavior-cloning warm
start, then PFSP self-play with engine bots in the roster**:

```bash
# 1. Distill expectiminimax-3 from bot-vs-bot games (~15 min on CPU).
#    --dataset caches the collected games for reuse across runs/ablations.
uv run python rl/bc_pretrain.py --bot e3 --episodes 2500 \
  --dataset runs/bc_e3_dataset.pkl --out runs/bc_e3/bc.pt

# 2. PFSP self-play fine-tune with e1/e2/e3 anchoring the roster.
uv run python rl/train_selfplay.py --resume runs/bc_e3/bc.pt \
  --total-steps 800000 --bots e1,e2,e3 --latest-prob 0.3 --pfsp-power 4 \
  --ent-coef-final 0.003 --amp --compile
```

In self-play the learner (seat 0) faces seat-1 opponents sampled per
episode: a frozen copy of its latest weights with `--latest-prob`, otherwise
a historical snapshot or engine bot chosen PFSP-style (probability ∝
`(1 - learner winrate)^pfsp_power`), focusing training on whatever still
beats it. The logged self-play win rate hovers near 50% by construction —
measure progress with `rl/eval.py` against fixed opponents. Per-arm
winrates are printed every 10 updates and logged to `metrics.jsonl`.

The plain curriculum (`train_ppo.py` vs `r`, then `v`, then self-play
without bots) also works but plateaus lower; ablations showed the BC warm
start and the bot-anchored PFSP roster are the two ingredients that matter,
while reward shaping and entropy schedules are neutral in this regime.

Every run writes `runs/<name>/config.json`, per-update `metrics.jsonl`
(losses, clipfrac, approx KL, explained variance, SPS, self-play arms), and
checkpoints (`latest.pt` every 5th update, `final.pt` at the end).
Diagnostics flags: `--target-kl` (early-stop updates), `--clip-vloss`.

Opponent codes are the engine's player codes: `r` random, `w` weighted-random,
`aa` attach-attack, `et` end-turn, `v` value-function, `e<N>` expectiminimax
of depth N, `m` MCTS. Depth study: e<N> stops improving at ~depth 4
(e3→e4 ≈ +6pp, e4→e5 ≈ nothing).

### Oracle (all-knowing) agents

`--oracle` (in `bc_pretrain.py`, `train_ppo.py`, `train_selfplay.py`) trains
a policy that sees the **full state** — opponent hand and deck included —
like the engine bots do. The flag is stored in the checkpoint config, so
`eval.py` and the self-play wrapper automatically route the right view:
oracle and honest agents share every script and can be benchmarked against
the same opponents (or each other via the self-play pool). Keep oracle
checkpoints clearly named; they are not legal players, they are strength
ceilings and sparring partners.

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

## Evaluate

```bash
# Panel of opponents, multiple seeds pooled, Wilson 95% CIs per opponent:
uv run python rl/eval.py runs/<run>/latest.pt \
  --opponent r,v,e1,e2,e3 --episodes 200 --seeds 999,4242 --amp
```

Evaluation variance against search bots is large (single 200-episode seeds
ranged 36–53% for the *same* checkpoint vs e3) — claim improvements only on
multi-seed pooled evals.

## Tests and ablations

```bash
uv run python -m pytest rl/tests -q   # checkpoint round-trips, masking, smoke
bash rl/run_bc_ablation.sh            # arch/width/heads/belief BC grid
bash rl/run_rl_ablation.sh            # shaping/entropy/curriculum/pool RL grid
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

`--aux-belief <w>` (res/tx) adds an opponent-hand prediction head trained
against the oracle view — auxiliary hidden-state inference shaping the
policy trunk (neutral in mirror-match ablations; disabled automatically for
oracle agents).

Checkpoints carry their architecture config explicitly (arch, sizes, heads,
memory, belief, oracle) and load with `weights_only=True`; legacy raw state
dicts are still detected from tensor shapes. `--arch`/`--hidden`/`--memory`
are ignored when resuming — the checkpoint dictates the network.

Ablation results (BC distillation of e3, 400 episodes/cell): architecture
dominates — res ≈ 45% vs e3, tx ≈ 38%, mlp ≈ 27%; hidden 512 and 8 heads
are mildly positive; belief weight is neutral. When scaling further: grow
the per-update batch via `--num-envs` before touching `--lr`.

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
