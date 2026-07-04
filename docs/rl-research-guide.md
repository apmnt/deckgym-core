# The deckgym RL Research, Explained

A self-contained guide to every concept used in this repository's
reinforcement-learning work: what each technique is, why it was chosen,
and what we measured. Companion documents: `docs/rl-agent-plan.md` (the
chronological experiment log), `docs/rl-concepts.md` (technique-by-
technique rationale from the early phases), `docs/rl-literature.md`
(sources and ranked next steps), `rl/README.md` (how to run everything).

## 1. The game, as a learning problem

Pokémon TCG Pocket is a two-player, zero-sum, imperfect-information card
game: 20-card decks, up to 2 copies per card, 4 board slots per side
(1 active + 3 bench), first to 3 points wins, ties at turn 30. Two
properties shape everything below:

- **Hidden information.** A player sees their own hand and deck contents
  but only the *size* of the opponent's. Decklists themselves are treated
  as public (like knowing the metagame), which the engine encodes by
  exposing both decks' card vocabularies.
- **Chance.** Shuffles, coin-flip attacks, and random energy generation
  make every action's outcome a distribution, not a single state.

The Rust engine (`src/`) is the ground truth for rules; the RL layer
never re-implements game logic.

### The built-in opponents ("engine bots")

`r` uniform random, `w` weighted random, `v` a greedy 1-step value
function, `e<N>` **expectiminimax** — minimax search over the game tree
with chance nodes averaged, depth N, using a hand-crafted state value
function at the leaves. Crucially, **all engine bots read the full game
state, including the agent's hand and both decks' order**. Beating them
"blind" is therefore a stronger result than the raw number suggests:
depth-3 expectiminimax (`e3`), our milestone opponent, is both a search
algorithm *and* an oracle.

## 2. Problem formulation

`src/rl_env.rs` (`RlEnvCore`) wraps one game as a single-agent decision
process:

- The learner sits in **seat 0**. In bot mode the opponent plays inside
  `step()`; in self-play mode the env pauses at *both* seats' decisions
  and reports whose turn it is, so Python can route each seat to a
  different policy.
- **Forced actions** (states with exactly one legal move) are applied
  automatically — the agent only ever sees real decisions.
- **Reward** is ±1 win/loss, 0 tie, discount γ = 1.0 (the ByteRL
  recipe): no reward shaping is needed because episodes are short
  (~30-40 decisions). Optional potential-based shaping on the point
  differential exists but measured neutral.
- Episodes auto-reset; a vectorized wrapper (`PyRlVecEnv`) steps dozens
  of envs in parallel across CPU threads (Rayon) and ships flat numpy
  arrays to PyTorch.

## 3. The action space: legal-action scoring

Card games have enormous nominal action spaces (every card × every
target × every mode), but only a handful of actions are legal at any
decision. Instead of a fixed global action head, the policy **scores the
legal-action list**: the env emits a feature vector per legal action
(action class, the card it references, target slot, attack index, energy
type, amounts), the network embeds each one, lets them **attend to each
other** (multi-head attention across the action set — "retreat" is
scored knowing "attack for lethal" is also available), and outputs one
logit per action. Illegal padding slots are masked to −∞.

This is the ygo-agent / DouZero pattern. It sidesteps action-space
explosion entirely and — combined with the card representation below —
is what makes one network playable with any deck.

## 4. Hidden information and the oracle critic (PTIE)

Two observation views are written for every decision:

- the **policy view**: what a real player could see — own hand/deck,
  both boards, discard piles, zone sizes; the opponent's hand and deck
  *composition* zeroed out;
- the **oracle view**: everything, including opponent hand and deck.

During training, the actor (policy) reads the policy view while the
**critic** (value function used to compute advantages) reads the oracle
view. This is *perfect-training-imperfect-execution* (PerfectDou): the
critic only shapes gradients and is discarded at play time, so the
deployed policy remains honest while enjoying a far less noisy training
signal. At evaluation the agent uses the policy view only.

Related, measured findings: giving the *policy* the oracle view
("oracle agents") did **not** make a stronger player — the hidden-
information bottleneck seems to act as a regularizer. A GRU memory over
decision steps (for inferring hidden cards from observed plays) and an
auxiliary opponent-hand prediction loss both measured neutral in this
regime.

## 5. Card representation: how one network plays every deck

The central design problem for a deck-general agent: observations must
mean the same thing regardless of which 40 cards are on the table.

- **Per-match vocabulary, fixed size.** Each matchup's distinct cards
  (≤ 40) get local indices; all one-hot/count observation sections are
  padded to `VOCAB_SIZE = 40`, so tensor shapes never change. The
  observation's last 40 entries hold each local slot's **global card
  id** — a stable index over the engine's full card enum (3,406 cards).
- **`CardEncoder`** (in `GeneralAgent`, `--arch gen`) maps each global
  id to a vector: a **learned identity embedding** (what training
  experienced with this exact card) **plus a projection of static
  attributes** — HP, type, stage, ex, retreat cost, per-attack
  damage/cost/effect flags, trainer kind (`deckgym.card_attr_table()`).
  Every one-hot/count section is then projected through the per-match
  card matrix (`counts @ card_repr`) before the trunk, and action
  features are re-embedded the same way.
- **Why attributes matter:** identity embeddings can't say anything
  about a card never seen in training; attributes make an unseen card
  "a 120-HP stage-1 water Pokémon with a 60-damage two-energy attack".
  This is what carries the zero-shot results (74% vs `e1` on decks the
  agent never trained on). **Effect-text embeddings** (TF-IDF and
  MiniLM variants, `--card-text`) were built on the same principle but
  measured *neutral at BC level* — the attribute + identity signal
  apparently already covers what the pool's cards need.

The pre-general agents (phases 1–9) used match-local one-hots directly:
maximally informative for one fixed matchup, meaningless across
matchups. That specialist/generalist tension is measured concretely:
the mirror-match champion beats the generalist 61–39 *on its one
matchup* — and cannot play any other matchup at all.

## 6. The training pipeline

Three stages, each with a measured reason to exist:

### 6.1 Behavior cloning (BC) warm start

Play thousands of e3-vs-e3 games (both seats the engine bot), record
seat-0 decisions, and train the network to predict the bot's chosen
action (cross-entropy over masked legal-action logits) plus a value
regression to the final outcome. Because the policy trains on the
*hidden* view while the demonstrator saw everything, this is PTIE
distillation — and it works: BC alone reaches ~44% vs `e2` across all
625 pool pairings.

Measured boundaries: BC quality is **not data-limited** (doubling
167k → 333k decisions changed nothing) but **overfits with epochs**
(8 epochs: train accuracy 95%, play strength collapsed; 4–5 epochs is
the working schedule). The BC ceiling appears to be
representation/demonstrator-limited.

### 6.2 PPO — and why its stability settings matter here

Proximal Policy Optimization with GAE, entropy regularization, and the
oracle critic. The single most expensive lesson of the project: **the
hyperparameters that fine-tuned the small mirror-match net destroy the
bigger general net.** At lr 2.5e-4 with 4 update epochs and no KL guard,
approx-KL ran ~0.06/update and greedy strength collapsed 43.8% → 21% vs
`e2` within 165k steps — PPO's clipping alone did not protect the BC
prior. The fix (lr 1e-4, `--target-kl 0.02` early-stopping updates,
value clipping) holds KL ≈ 0.01 and never regressed afterwards. When a
warm start carries most of the skill, protecting it is the first-order
concern.

### 6.3 PFSP self-play with engine bots in the roster

Pure self-play drifts: the agent gets better against itself and *worse*
against the search bot it's supposed to beat. **Prioritized fictitious
self-play** (AlphaStar): seat 1 is sampled per episode from a roster —
a frozen copy of the current learner, historical snapshots, engine bots
(`e1,e2,e3`), and optionally **frozen checkpoint arms** — with
probability ∝ (1 − winrate)^p, focusing training on whatever still
beats the learner. The bots in the roster are the measured active
ingredient (ablation: +6.5pp vs uniform-self-play-only). The in-training
win rate hovers near 50% *by construction*; progress is only measurable
by separate greedy evals against fixed opponents.

## 7. Exploiters and the two distillation mechanisms

### Distill-then-exploit (how the champion was beaten)

Training a best response to a frozen strong opponent from a *distant*
starting policy kept converging to parity (~49% over four runs, 1.35M
steps): conservative updates that protect skills also prevent strategy
jumps. The working recipe: **first BC-clone the opponent from its own
recorded games** (84.9% move agreement), **then run the exploiter from
inside the opponent's strategy region** — small deviations from the
opponent's own play are easy to find and learn (fictitious-play
intuition). Result: 54.0% vs a champion that even full-vision `e3` only
beats 45.8% of the time.

### Specialist distillation (how per-deck gains are kept)

RL fine-tuning on a focus deck reliably lifts that deck's win rate —
and further RL on other decks erodes it again (sequential focus legs
measured zero-sum). But **recorded games survive**: train a short-lived
specialist, record its play, and merge specialists into the general
model by supervised BC on the combined data. Distillation preserves what
RL forgets, because supervised learning has no reason to trade one
deck's lines against another's. This mechanism (proven by the champion
merge, `general_v2.pt`) is the current phase-13 route to beating `e3`
pool-wide: `e3` is already beaten per-matchup (55.7% on the mirror);
what's missing is coverage across 625 pairings, not capability.

### The skill-gap metric

Absolute per-deck win rates confound deck strength with piloting skill
(`e3` itself only wins 15% with the pool's weakest deck). The actionable
diagnosis is the **skill gap**: our win rate with a deck minus e3's own
rate piloting the same deck under the same protocol. Focused training
targets the largest gaps, not the lowest absolute rates — the first
focus-leg attempt targeted absolute rates and learned nothing, because
those decks were simply weak.

## 8. Evaluation methodology (the part that bites)

- **Greedy evals** (argmax policy) measure deployment strength; sampled
  play runs several points lower and is what PFSP logs show.
- **Variance is brutal**: the same checkpoint legitimately reads 36–53%
  vs `e3` on different 100–200-episode seeds. A 49% single-seed reading
  in the text-feature ablation collapsed to 34.6% under 600 verification
  episodes. Rule: **never claim a result from one seed**; pool ≥ 2 seeds
  × 200 episodes and report Wilson 95% intervals.
- **Head-to-head protocol** (`rl/head_to_head.py`): both agents in the
  same games, both seatings pooled to cancel seat asymmetry, greedy by
  default; legacy-layout checkpoints are adapted losslessly.
- **Hill-climbing protocol**: after each training burst, a pooled eval
  decides keep-or-revert against the incumbent. Bursts from the same
  parent varied 37.2–41.1%, i.e. burst noise ≈ CI width — which is why
  mid-run snapshot harvesting (keep the best of several checkpoints per
  run, not just the final) is the planned refinement.

## 9. Where the numbers stand (July 2026)

| Agent | vs e1 | vs e2 | vs e3 | Note |
|---|---|---|---|---|
| `general_m2.pt` (pool matchups) | ~79.5% | 48.5% | **41.1%** | current best general |
| `general_m2` (held-out decks, zero-shot) | ~74% | ~36% | 24.7% | attributes carry the transfer |
| `general_v2.pt` | 77.1% | 39.6% | 33.0% | + beats mirror champion 54.2% |
| best per-matchup (mirror exploit ckpt) | — | — | 55.7% | proves e3 is beatable per-matchup |
| `e3`'s own seat-0 average with pool decks | — | — | 47.5% | "beat e3" = out-pilot an oracle |

Open problems, updated after the phase-13 campaign: keyword-flag
attributes were the one lever that moved a number (+5pp at BC level) and
should ride the next full BC->PFSP line; specialist distillation needs
specialists with real per-deck deltas (the oversampling burst had already
absorbed them); determinized one-ply search with the engine's hand-crafted
value function measured *negative* (e1-grade judgment diluting a stronger
policy) — honest search needs a learned policy-view value head; snapshot
keep-better is implemented and awaits a long run.

## 10. Glossary

| Term | Meaning here |
|---|---|
| e<N> | expectiminimax engine bot, depth N, full state visibility |
| BC | behavior cloning: supervised learning on recorded decisions |
| PTIE | perfect-training-imperfect-execution: oracle critic, honest policy |
| PFSP | prioritized fictitious self-play: opponents sampled ∝ (1−winrate)^p |
| roster / arm | the set of selectable seat-1 opponents / one member of it |
| frozen opponent | a fixed checkpoint arm (exploit target or sparring partner) |
| exploiter | policy trained specifically to beat one frozen opponent |
| specialist | short-lived per-deck fine-tune whose games feed distillation |
| skill gap | our per-deck win rate minus e3's own rate with the same deck |
| oracle view | full-state observation (both hands/decks visible) |
| policy view | legal-information observation (opponent hand/deck hidden) |
| vocab / global id | per-match card slots (≤40) / engine-wide card index |
| keep-better | accept a training burst only if pooled eval beats incumbent |
