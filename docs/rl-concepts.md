# RL Concepts in the deckgym Agent — What, Where, and Why Here

This document explains every reinforcement-learning concept used in the
`rl/` stack and, for each one, why it helps *specifically* for Pokémon TCG
Pocket as simulated by deckgym-core. File references point at the
implementation. For the experimental history and numbers, see
`rl-agent-plan.md`.

## 1. The problem, framed precisely

Pokémon TCG Pocket battles are **two-player, zero-sum, stochastic,
imperfect-information** games: you cannot see the opponent's hand or either
deck's order, coin flips and shuffles inject chance, and a game lasts ~40
decisions per side. Each property pulls in a design decision:

| Property | Consequence in this stack |
|---|---|
| Two-player zero-sum | self-play training; opponent is part of the environment |
| Imperfect information | hidden-info observations, oracle critic, GRU memory, belief loss |
| Stochastic | many parallel episodes, multi-seed evaluation, chance handled by the env |
| Long horizon, sparse outcome | γ=1.0 terminal reward, GAE, optional potential shaping |
| Variable legal moves | legal-action scorer instead of a global action space |

## 2. Environment design (`src/rl_env.rs`, `src/python_bindings.rs`)

**Single-agent view of a two-seat game.** `RlEnvCore` exposes the game as a
Markov decision process for seat 0: opponent moves and *forced* actions
(states with exactly one legal action) are auto-applied inside `step`, so
the policy only ever sees genuine decisions. This roughly halves the
sequence length the network must model and keeps credit assignment on real
choices. Rewards are always from seat 0's perspective, so the PPO loop
stays textbook-simple.

**Vectorized, auto-resetting envs.** Training steps 32+ games in lockstep
(Rayon-parallel in Rust), and finished episodes reset themselves so every
returned observation belongs to a live game. Card-game simulation is
CPU-bound; batching decisions across environments is what makes
neural-network inference and PPO updates efficient.

**Self-play mode.** With `opponent="self"` the env pauses at *both* seats'
decisions and reports whose turn it is; observations are written from the
pending seat's perspective. This is what lets frozen networks (and engine
bots, via `bot_decide_many`) occupy seat 1 without duplicating any game
logic.

## 3. Legal-action scorer (the action space)

Card games explode under a fixed global action enumeration: "play card X
targeting slot Y with choice Z" yields thousands of mostly-illegal IDs. We
instead return, at each decision, a **feature vector per legal action**
(action class, card identity, target slot, attack index, energy type,
scalar effect sizes), and the policy emits **one logit per legal action**.
Illegal padding slots are masked to −∞ before the softmax. This is the
DouZero / ygo-agent pattern.

Why it matters here: Pocket decisions range from 2 to dozens of legal
actions whose *meaning* depends entirely on context. Scoring the actual
options sidesteps both the combinatorial action space and the
invalid-action-masking problem, and lets one network play any matchup the
feature schema covers.

## 4. PPO with GAE (`rl/train_ppo.py`, `rl/train_selfplay.py`)

**PPO (clipped surrogate)** is the on-policy policy-gradient workhorse: it
improves the policy in small trust-region-like steps by clipping the
probability ratio between new and old policies. We chose it because it is
stable under the two things this domain throws at it — sparse ±1 rewards
and a nonstationary opponent distribution in self-play — and because every
strong card-game agent lineage (ByteRL, DouZero variants, PerfectDou)
shipped a policy-gradient learner before anything fancier.

**GAE (generalized advantage estimation)** blends n-step returns
(λ = 0.95) to trade bias against variance when estimating "how much better
was this action than expected". With ~40 decisions and a single terminal
reward, pure Monte-Carlo returns are noisy and pure one-step bootstraps
lean too hard on the critic; GAE sits between.

**γ = 1.0**: the game is short and only the outcome matters — there is no
reason to prefer early points over late wins. Discounting would distort
the objective; episode length provides the horizon.

**Entropy bonus** keeps the policy stochastic enough to explore lines it
would otherwise abandon; `--ent-coef-final` anneals it as the policy
matures. (Empirically neutral in our ablations at 200k-step scale — kept
because it is cheap insurance on longer runs.)

**Diagnostics**: `approx_kl` (how far each update moves the policy),
optional `--target-kl` early stop (abort an update drifting too far from
the data-collecting policy), optional clipped value loss, and explained
variance (how much of the return the critic actually predicts). These
don't change learning; they make failures visible before they cost a day.

## 5. Hidden information: policy view vs oracle view

The engine itself is open-handed (both `Player`s see everything — the
built-in bots exploit this). The RL agent is trained to play the *real*
game: `write_observation(.., include_hidden=false)` zeroes the opponent's
hand contents and deck composition (sizes stay visible). Every step the
env returns **both** views.

**Oracle critic (PerfectDou's "perfect training, imperfect execution").**
The value network reads the *full* state during training while the policy
reads only the legal view. This is sound because the critic exists only to
reduce gradient variance — it is discarded at play time. It helps
enormously here because the true value of a position depends heavily on
the opponent's hand; a critic that sees it gives far better baselines than
one that must guess, while the deployed policy stays honest.

Consequence worth remembering: the engine bots (`v`, `e<N>`, `m`) see the
agent's hand during evaluation, so beating them is a *stronger* result
than the number suggests.

## 6. Reward design

Terminal **+1 / −1 / 0** only. Win-rate is the actual objective, and dense
hand-crafted rewards invite reward hacking. The one safe additive signal
is **potential-based shaping** (`--shaping c`): r += Φ(s′) − Φ(s) with
Φ = c · (point differential) and Φ(terminal) = 0. Because the shaping
telescopes to zero over any episode, the optimal policy is provably
unchanged — it only redistributes credit earlier. (Tested: neutral in this
game; prize points are already closely coupled to winning.)

## 7. Network architectures (`rl/agent.py`)

**ResAttnAgent (default)** — three ideas stacked:

- *Residual GELU trunk with pre-norm LayerNorm*: standard deep-learning
  hygiene that lets 4+ layer MLPs optimize stably; the original 2-layer
  tanh MLP underfit e3's decision function badly (BC top-1 65% vs 87%).
- *Action self-attention*: each legal action becomes a token (its features
  plus a projection of the state), tokens attend to each other, then a
  head scores each token. "Retreat" is scored *knowing* "attack for
  lethal" is also on the menu — relative action quality is exactly what a
  card turn is about. This was the single largest measured factor in the
  whole project (≈ +18pp vs e3 over the MLP on identical training data).
- *Separate critic encoder*: the policy and critic read different inputs
  (hidden vs oracle view), so sharing a trunk is not possible by design.

**TokenTransformerAgent (`--arch tx`)** restores structure to the flat
observation: 1 globals token, 8 board-slot tokens (shared projection +
position embeddings), 6 zone-count tokens, processed by a
TransformerEncoder. In a *fixed mirror match* it underperforms `res` (the
flat encoding loses nothing when the card pool is static); its value is
expected in multi-deck generalization, where shared slot/zone projections
transfer.

**GRU memory (`--memory`)**. From the policy's seat the game is a POMDP:
two states identical on the board can differ in what the opponent has
already shown. A GRUCell over *decision steps* lets the policy integrate
the game's history (cards seen, plays made) instead of reacting to
snapshots. Training requires sequence-aware PPO: minibatches become
env-major sequences, the GRU is replayed over the rollout from a stored
initial hidden state (truncated BPTT), and hidden state is zeroed exactly
at episode boundaries — in the rollout, in the replay, and for every
frozen self-play opponent. The critic stays feedforward: it sees the full
state, which is near-Markov, so recurrence buys it nothing.

**Belief auxiliary loss (`--aux-belief`)** trains a small head to predict
the opponent's hand composition (a slice of the oracle view) from the
policy trunk. The prediction is never used at play time; its gradient
pushes the trunk toward hidden-state-inference features. This is the
cheap, supervised end of the belief-modeling spectrum (the expensive end
is public-belief-state methods like ReBeL). Measured neutral in the mirror
match so far — kept as an option because its cost is near zero.

## 8. Self-play (`rl/selfplay_env.py`, `rl/train_selfplay.py`)

**Why self-play at all**: fixed opponents cap what can be learned — direct
fine-tuning against e2/e3 stalled because a much stronger fixed opponent
gives sparse, mostly-negative reward and nothing to climb. Self-play
supplies an opponent that is always *exactly* at the learner's level.

**Historical pool (fictitious self-play)**: playing only the latest self
invites strategy cycling (A beats B beats C beats A) and catastrophic
forgetting. Keeping frozen snapshots (every `--snapshot-every` updates,
capped pool) forces the learner to stay strong against its own past.

**PFSP (prioritized fictitious self-play, AlphaStar)**: opponents are
sampled with probability ∝ (1 − learner winrate)^power from a rolling
per-arm outcome window. Uniform sampling wastes most games on opponents
already beaten; PFSP concentrates training where the learner still loses.
With `--pfsp-power 4` the hardest arm dominates the mix.

**Engine bots in the roster (`--bots e1,e2,e3`)**: pure self-play drifts
in its own meta and can lose ground against search-based play. Putting the
target bots *inside* the PFSP roster anchors training to the actual goal —
this moved e2 from 49% to 61.5% — while the self-play arms prevent
overfitting to any single bot's quirks.

**Frozen opponents sample (rather than argmax)** their actions, keeping
the opponent distribution diverse; each frozen net plays from its own
perspective with its own hidden-info view (or full view, if it is an
oracle agent — the checkpoint config decides).

## 9. Behavior-cloning warm start (`rl/bc_pretrain.py`)

The highest-leverage step in the project. Expectiminimax-3 plays itself
for a few thousand games (both seats decided in Rust, in parallel); seat-0
decisions are recorded; the policy is trained with cross-entropy on the
chosen action over masked legal-action logits, plus a value regression
toward the final outcome.

Two details make it work:

- The policy is trained on the **hidden-information view while the
  demonstrator saw everything** — distillation across the information gap
  (the PTIE idea again). It learns "what would an oracle searcher do, as
  far as my information can tell".
- The same cached dataset (`--dataset`) serves every architecture/width
  ablation, making representation comparisons cheap and exactly matched.

Why it beats pure RL here: PPO's exploration never stumbles onto
e3-quality lines from a 364k-parameter policy's neighborhood — the
plateau at ~40% vs e3 was an exploration ceiling, not a capacity ceiling.
Imitation hands the search bot's competence over directly (45% vs e3
before any RL), and PFSP fine-tuning then *surpasses* the demonstrator
(54.2%) because RL can deviate where the search bot is wrong.

## 10. Oracle (all-knowing) agents (`--oracle`)

A second agent kind whose *policy* also receives the full-state view —
information-matched with the engine bots. Implementation is a checkpoint
config flag; `eval.py` and the self-play wrapper route the correct view
automatically, so oracle and honest agents share all scripts and
opponents. Uses: an upper bound on achievable strength (how much is the
information handicap costing?), an information-fair fight against the
search bots, and a strong sparring partner for the honest agent's pool.
Finding so far: at the *imitation* stage the oracle view adds nothing
(e3's moves are as predictable from the legal view); the advantage, if
any, must come from RL exploitation.

## 11. Evaluation methodology (`rl/eval.py`)

- **Greedy (argmax) evaluation** of a separately-trained sampling policy:
  the deployed agent shouldn't gamble, and greedy is consistently a few
  points stronger than sampling.
- **Multi-seed pooling with Wilson 95% CIs**: single 200-episode evals of
  the *same* checkpoint vs e3 ranged 36–53%. Every claim in this project
  is made on ≥2 seeds pooled (the champion's 54.2% is 801 episodes over 4
  seeds, all individually >50%).
- **Fixed panels** (`--opponent r,v,e1,e2,e3`) keep regression checks
  cheap; in-training self-play winrate is *not* a progress metric (it
  hovers at 50% by construction) — per-arm PFSP stats and offline evals
  are.

## 12. Engineering choices that earn their keep

- **Checkpoints carry their config** (arch, sizes, heads, memory, belief,
  oracle) and load via `weights_only=True` — reproducible reconstruction
  without unpickling arbitrary code; legacy raw state dicts still load via
  shape inference.
- **`config.json` + `metrics.jsonl` per run** — every experiment is
  reconstructible after the fact (this recovered two runs killed by the
  host mid-training).
- **Rayon parallelism** across envs for stepping, observation writing, and
  bot decisions — the simulator is the bottleneck, and the bot-decision
  path is what makes putting e3 in the training roster affordable.
- **Tests** (`rl/tests/`): checkpoint round-trips, the
  hidden-information masking invariant (policy view ⊆ oracle view), and a
  CPU smoke run of the full trainer.

## 13. What was measured *not* to matter (here)

Negative results from the ablation grids, recorded so nobody re-spends the
compute: reward shaping coefficients (0.05–0.2), entropy schedules (fixed
vs annealed at several levels), belief-loss weights at BC stage, and the
token-transformer encoder in a fixed mirror match. The variance lives in
**architecture (action attention + depth), the BC warm start, and the
opponent curriculum** — in that order.
