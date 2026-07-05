# RL Literature Review — card-game agents and where deckgym's fit

Second-round survey (July 2026) after the deck-general agent (phase 10).
What the field does, what we already borrow, and what to try next.
Round-one sources (ygo-agent, ByteRL, DouZero, PerfectDou) are covered in
`docs/rl-agent-plan.md`.

## 1. Generalizing across decks / card pools

- **ygo-agent** (sbl1996, Yu-Gi-Oh): one policy across many decks. Cards
  are tokens; each card = id embedding fused with feature vectors; id
  embeddings are *initialized from LLM text embeddings of the card's
  effect text*, then fine-tuned. Legal-action-list scoring, PPO league
  self-play. Our `GeneralAgent` is this pattern minus the text-embedding
  init (our attribute table plays that role).
- **Cardsformer** (Hearthstone): grounds card *descriptions* (language)
  into a prediction model + policy, explicitly to generalize to unseen
  cards. Supports our finding that attribute grounding is what transfers
  (our zero-shot e1 74.3% vs in-pool 79.5%).
- **Deck-building under changing card pools** (Dockhorn et al. and
  follow-ups): representations built from numerical/nominal/text card
  features generalize to unseen cards, pure identity encodings do not.

Implication: the natural upgrade for `CardEncoder` is initializing card
identity embeddings from sentence-embedding vectors of each card's
attack/ability *effect text* (all in database.json) — zero engine work,
directly follows ygo-agent.

## 2. Opponent regimes: self-play, PFSP, leagues, exploiters

- **AlphaStar league** (Nature 2019): three populations — main agents
  (PFSP), *main exploiters* (train only vs the current main agent),
  *league exploiters* (PFSP vs the whole league). Prevents strategy
  collapse and patches specific weaknesses.
- **Minimax Exploiter** (2023): data-efficient exploiter design for
  production settings; exploiters find weaknesses much faster than
  another PFSP arm does.
- **Robust opponent-aware league training** (NeurIPS 2023, StarCraft II):
  league refinements beyond AlphaStar's original recipe.
- **Learning to Beat ByteRL** (2024): state-of-the-art CCG self-play
  agents (ByteRL, Legends of Code & Magic) are *exploitable* — targeted
  counter-strategies reach 60-70% winrates. Self-play alone leaves
  systematic holes; adversarial opponents during training are the fix.
- **DeepNash / R-NaD** (Stratego): regularized Nash dynamics rather than
  a league — principled equilibrium-seeking without belief-state search.
- **DouZero+** (2022): explicit opponent modeling (predicting the
  opponent's hand) as an auxiliary task; our `--aux-belief` is this idea
  (neutral in mirror ablations, untested in the general setting).
- **Self-play survey** (arXiv 2408.01072) for the broader taxonomy.
- **Generals.io superhuman agent** (2026): careful engineering of a
  single-machine self-play pipeline (env throughput, replay, curricula)
  beats scale — encouraging for deckgym's CPU-bound setting.

Implication: our roster (latest + snapshots + engine bots under PFSP) is
a mini-league without exploiters. The cheapest high-value addition is a
*main exploiter* arm: periodically fork the current agent, train it only
against the frozen main agent for ~50k steps, add it to the pool. This
directly targets the "BC plateau + flat PFSP" we hit.

## 3. Search at play time (imperfect information)

- **ReBeL** (FAIR 2020): RL + search over *public belief states*;
  superhuman heads-up poker. Sound but heavy: needs belief-space value
  functions and per-move search.
- **Student of Games** (Schmid et al. 2023): unifies AlphaZero-style
  search with imperfect-information sound search (growing-tree CFR);
  one algorithm for chess, go, poker, Scotland Yard.
- **IS-MCTS / determinization**: sample opponent hands consistent with
  observations, run MCTS per determinization, vote. Not equilibrium-sound
  (strategy fusion) but simple and strong in practice for card games.
- deckgym context: hidden information here is only deck order + opponent
  hand; the engine bots (expectiminimax) already cheat by seeing
  everything. A determinized 1-2 ply search on top of the trained policy
  (policy priors + value leaf evals, like the phase-6 "AZ-style PUCT"
  plan) is the pragmatic version; ReBeL/SoG-grade machinery is likely
  overkill for a 20-card, 3-point game.

## 4. Pokémon TCG specifically (very active in 2025-2026)

- **Official Pokémon TCG AI Battle Challenge** (Kaggle, June-Aug 2026):
  The Pokémon Company is running a $300k+ AI competition with a provided
  simulator SDK for RL training — same genre of engine as deckgym-core.
  deckgym's agents and infra are directly relevant preparation.
- **PTCG-Bench** (2026): LLM agents on the full 60-card PTCG. Ten LLM
  backbones; none of five self-evolution mechanisms (Reflexion, ExpeL,
  memory, prompt/skill evolution) improved consistently; harness design
  mattered as much as model choice. No RL baselines — an open comparison
  deckgym could supply.
- **TCGJax** (2026): auto-generated JAX Pokémon TCG *Pocket* environment;
  env overhead <4% of training time at 200M-param policies on GPU.
  Confirms the ceiling for throughput if deckgym ever wants GPU-native
  vectorization; also a potential cross-check simulator.
- **PokéAgent Challenge** (NeurIPS 2025): battling + speedrunning tracks,
  RL + LLM hybrid focus.

## 5. Ranked next steps for deckgym (from this review)

1. **Text-embedding card init** (§1): embed database.json effect text
   with any sentence encoder, project into `CardEncoder`; expected to
   close part of the zero-shot gap (e2 48.2% in-pool vs 36.2% held-out).
2. **Main-exploiter arm in self-play** (§2): directly attacks the PFSP
   plateau and the exploitability risk ByteRL demonstrated.
3. **Determinized shallow search over the policy** (§3): reuse the
   engine's forecasting to add 1-2 ply lookahead with policy priors at
   eval time; cheapest route past e3 without new training.
4. **Opponent-hand auxiliary loss in the general setting** (§2, DouZero+):
   was neutral in the mirror; hidden-information value is likely higher
   across 625 matchups.
5. **Throughput** (§4): batch env stepping currently saturates 4 CPU
   cores at ~100 SPS vs search bots; a GPU box (or TCGJax-style JAX port)
   is the scaling lever, not more hyperparameter search.

## Sources

- ygo-agent — https://github.com/sbl1996/ygo-agent
- Cardsformer — https://www.researchgate.net/publication/374299909
- AlphaStar league — https://www.nature.com/articles/s41586-019-1724-z
- Minimax Exploiter — https://arxiv.org/abs/2311.17190
- Opponent-aware league training — https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf
- Learning to Beat ByteRL — https://arxiv.org/abs/2404.16689
- Self-play survey — https://arxiv.org/abs/2408.01072
- AlphaExploitem (poker, K-best league) — https://arxiv.org/abs/2605.09150
- Generals.io superhuman self-play — https://arxiv.org/abs/2606.23348
- ReBeL — https://arxiv.org/abs/2007.13544
- PTCG-Bench — https://arxiv.org/abs/2605.29653
- TCGJax / auto-generated RL envs — https://arxiv.org/abs/2603.12145
- Pokémon TCG AI Battle Challenge — https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- PokéAgent Challenge — https://pokeagent.github.io/
