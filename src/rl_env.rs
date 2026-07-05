//! Reinforcement-learning environment on top of the game engine.
//!
//! `RlEnvCore` exposes the game as a single-agent decision process: the agent
//! sits in seat 0, the opponent (any built-in `Player`) plays seat 1 inside
//! `step`, and forced single-action states are auto-applied. Observations and
//! per-legal-action feature vectors are flat `f32` slices so they can be
//! shipped to Python as numpy arrays without per-object conversion.
//!
//! The design follows the "legal-action scorer" pattern (ygo-agent, DouZero):
//! the policy receives the feature vector of every legal action and emits one
//! logit per action, so no global action enumeration is needed.

use std::collections::HashMap;
use std::sync::LazyLock;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use strum::IntoEnumIterator;

use crate::actions::{apply_action, Action, SimpleAction};
use crate::card_ids::CardId;
use crate::database::get_card_by_enum;
use crate::deck::Deck;
use crate::models::{Card, EnergyType, TrainerType};
use crate::players::{create_players, Player, PlayerCode};
use crate::state::GameOutcome;
use crate::State;

/// Hard cap on legal actions encoded per decision. The engine rarely exceeds
/// a few dozen; exceeding the cap is a hard error so it is never silent.
pub const MAX_ACTIONS: usize = 128;

/// Fixed size of the per-match card vocabulary sections in observations and
/// action features. Two 20-card decks contain at most 40 distinct cards, so
/// any matchup fits; unused slots stay zero. A fixed size makes the
/// observation/action dims independent of the decks being played, which is
/// what lets a single network play any deck against any deck.
pub const VOCAB_SIZE: usize = 40;

const NUM_ENERGY_TYPES: usize = 10;
const NUM_STATUS: usize = 5;
const NUM_ACTION_CLASSES: usize = 18;
const NUM_SLOTS: usize = 4; // active + 3 bench
const MAX_POINTS: f32 = 3.0;
const MAX_TURNS: f32 = 30.0;
const MAX_COPIES: f32 = 2.0;

/// Stable global index for every card in the engine: the position of its
/// `CardId` in enum-definition order. Index `num_global_cards()` is reserved
/// as the padding id for empty vocabulary slots.
static GLOBAL_CARD_INDEX: LazyLock<HashMap<CardId, u32>> = LazyLock::new(|| {
    CardId::iter()
        .enumerate()
        .map(|(i, card_id)| (card_id, i as u32))
        .collect()
});

pub fn num_global_cards() -> usize {
    GLOBAL_CARD_INDEX.len()
}

fn global_card_index(card: &Card) -> u32 {
    GLOBAL_CARD_INDEX
        .get(&card.get_card_id())
        .copied()
        .unwrap_or(num_global_cards() as u32)
}

/// Static per-card attribute features (see `write_card_attrs` for layout).
/// These give the network semantics for cards it has rarely (or never) seen:
/// an unfamiliar card is still "a 120-HP stage-1 water Pokemon with a 60-dmg
/// two-energy attack". Identity embeddings capture the rest.
pub const CARD_ATTR_DIM: usize = 45;
const MAX_ATTACKS: usize = 3;

fn write_card_attrs(card: &Card, out: &mut [f32]) {
    debug_assert_eq!(out.len(), CARD_ATTR_DIM);
    out.fill(0.0);
    match card {
        Card::Pokemon(pokemon) => {
            out[0] = 1.0;
            out[6 + energy_index(pokemon.energy_type)] = 1.0;
            out[16] = pokemon.stage as f32 / 2.0;
            out[17] = pokemon.hp as f32 / 300.0;
            out[19] = pokemon.retreat_cost.len() as f32 / 4.0;
            if let Some(weakness) = pokemon.weakness {
                out[20 + energy_index(weakness)] = 1.0;
            }
        }
        Card::Trainer(trainer) => {
            let type_idx = match trainer.trainer_card_type {
                TrainerType::Supporter => 0,
                TrainerType::Item => 1,
                TrainerType::Tool => 2,
                TrainerType::Fossil => 3,
                TrainerType::Stadium => 4,
            };
            out[1 + type_idx] = 1.0;
        }
    }
    out[18] = card.is_ex() as u8 as f32;
    out[30] = card.get_ability().is_some() as u8 as f32;
    out[31] = card.is_basic() as u8 as f32;
    let attacks = card.get_attacks();
    out[32] = attacks.len() as f32 / MAX_ATTACKS as f32;
    for (k, attack) in attacks.iter().take(MAX_ATTACKS).enumerate() {
        let j = 33 + 4 * k;
        out[j] = 1.0;
        out[j + 1] = attack.fixed_damage as f32 / 200.0;
        out[j + 2] = attack.energy_required.len() as f32 / 4.0;
        out[j + 3] = attack.effect.is_some() as u8 as f32;
    }
}

/// Flat `(num_global_cards() + 1) x CARD_ATTR_DIM` attribute table in global
/// index order; the final row (the padding id) is all zeros.
pub fn card_attr_table() -> Vec<f32> {
    let n = num_global_cards();
    let mut table = vec![0.0f32; (n + 1) * CARD_ATTR_DIM];
    for card_id in CardId::iter() {
        let card = get_card_by_enum(card_id);
        let idx = global_card_index(&card) as usize;
        write_card_attrs(
            &card,
            &mut table[idx * CARD_ATTR_DIM..(idx + 1) * CARD_ATTR_DIM],
        );
    }
    table
}

/// Card id strings in global index order (for debugging/tooling).
pub fn global_card_ids() -> Vec<String> {
    CardId::iter()
        .map(|card_id| get_card_by_enum(card_id).get_id())
        .collect()
}

fn energy_index(energy: EnergyType) -> usize {
    match energy {
        EnergyType::Grass => 0,
        EnergyType::Fire => 1,
        EnergyType::Water => 2,
        EnergyType::Lightning => 3,
        EnergyType::Psychic => 4,
        EnergyType::Fighting => 5,
        EnergyType::Darkness => 6,
        EnergyType::Metal => 7,
        EnergyType::Dragon => 8,
        EnergyType::Colorless => 9,
    }
}

/// Coarse class of a `SimpleAction`, used as a one-hot feature.
fn action_class(action: &SimpleAction) -> usize {
    match action {
        SimpleAction::DrawCard { .. } => 0,
        SimpleAction::Play { .. } => 1,
        SimpleAction::Place(..) => 2,
        SimpleAction::Evolve { .. } => 3,
        SimpleAction::UseAbility { .. } => 4,
        SimpleAction::Attack(..) => 5,
        SimpleAction::UseCopiedAttack { .. } => 6,
        SimpleAction::Retreat(..) => 7,
        SimpleAction::EndTurn => 8,
        SimpleAction::Attach { .. } => 9,
        SimpleAction::MoveEnergy { .. } => 10,
        SimpleAction::AttachTool { .. } => 11,
        SimpleAction::Heal { .. }
        | SimpleAction::HealAndDiscardEnergy { .. }
        | SimpleAction::HealAllEeveeEvolutions => 12,
        SimpleAction::ApplyDamage { .. }
        | SimpleAction::ScheduleDelayedSpotDamage { .. }
        | SimpleAction::MoveAllDamage { .. }
        | SimpleAction::ApplyEeveeBagDamageBoost
        | SimpleAction::DiscardRandomOpponentActiveEnergy => 13,
        SimpleAction::Activate { .. } => 14,
        SimpleAction::CommunicatePokemon { .. }
        | SimpleAction::ShufflePokemonIntoDeck { .. }
        | SimpleAction::ShuffleOwnCardsIntoDeck { .. }
        | SimpleAction::ShuffleOpponentSupporter { .. }
        | SimpleAction::DiscardOpponentSupporter { .. }
        | SimpleAction::DiscardOwnCards { .. }
        | SimpleAction::AttachFromDiscard { .. }
        | SimpleAction::SadaAttach { .. } => 15,
        SimpleAction::DiscardFossil { .. }
        | SimpleAction::UseStadium
        | SimpleAction::ReturnPokemonToHand { .. }
        | SimpleAction::ShuffleInPlayPokemonIntoDeck { .. }
        | SimpleAction::DiscardToolFromPokemon { .. }
        | SimpleAction::DiscardActiveStadium => 16,
        SimpleAction::Noop => 17,
    }
}

/// Card referenced by an action, if any (for the card one-hot feature).
fn action_card(action: &SimpleAction) -> Option<Card> {
    match action {
        SimpleAction::Play { trainer_card } => Some(Card::Trainer(trainer_card.clone())),
        SimpleAction::Place(card, _) => Some(card.clone()),
        SimpleAction::Evolve { evolution, .. } => Some(evolution.clone()),
        SimpleAction::AttachTool { tool_card, .. } => Some(tool_card.clone()),
        SimpleAction::CommunicatePokemon { hand_pokemon } => Some(hand_pokemon.clone()),
        SimpleAction::ShufflePokemonIntoDeck { hand_pokemon } => hand_pokemon.first().cloned(),
        SimpleAction::ShuffleOwnCardsIntoDeck { cards } => cards.first().cloned(),
        SimpleAction::ShuffleOpponentSupporter { supporter_card }
        | SimpleAction::DiscardOpponentSupporter { supporter_card } => Some(supporter_card.clone()),
        SimpleAction::DiscardOwnCards { cards } => cards.first().cloned(),
        _ => None,
    }
}

/// (in_play_idx, target_player) referenced by an action, if any. The player
/// is `None` for actions that implicitly target the actor's own side.
fn action_target(action: &SimpleAction) -> Option<(usize, Option<usize>)> {
    match action {
        SimpleAction::Place(_, idx) => Some((*idx, None)),
        SimpleAction::Evolve { in_play_idx, .. }
        | SimpleAction::UseAbility { in_play_idx }
        | SimpleAction::AttachTool { in_play_idx, .. }
        | SimpleAction::Heal { in_play_idx, .. }
        | SimpleAction::HealAndDiscardEnergy { in_play_idx, .. }
        | SimpleAction::AttachFromDiscard { in_play_idx, .. }
        | SimpleAction::DiscardFossil { in_play_idx }
        | SimpleAction::ReturnPokemonToHand { in_play_idx }
        | SimpleAction::ShuffleInPlayPokemonIntoDeck { in_play_idx } => Some((*in_play_idx, None)),
        SimpleAction::Retreat(idx) => Some((*idx, None)),
        SimpleAction::Activate { in_play_idx, .. } => Some((*in_play_idx, None)),
        SimpleAction::MoveEnergy { to_in_play_idx, .. } => Some((*to_in_play_idx, None)),
        SimpleAction::Attach { attachments, .. } => {
            attachments.first().map(|(_, _, idx)| (*idx, None))
        }
        SimpleAction::ApplyDamage { targets, .. } => targets
            .first()
            .map(|(_, target_player, idx)| (*idx, Some(*target_player))),
        SimpleAction::ScheduleDelayedSpotDamage {
            target_player,
            target_in_play_idx,
            ..
        } => Some((*target_in_play_idx, Some(*target_player))),
        SimpleAction::DiscardToolFromPokemon {
            player,
            in_play_idx,
        } => Some((*in_play_idx, Some(*player))),
        _ => None,
    }
}

fn action_energy(action: &SimpleAction) -> Option<EnergyType> {
    match action {
        SimpleAction::Attach { attachments, .. } => {
            attachments.first().map(|(_, energy, _)| *energy)
        }
        SimpleAction::MoveEnergy { energy_type, .. } => Some(*energy_type),
        SimpleAction::SadaAttach { assignments } => assignments.first().map(|(e, _)| *e),
        _ => None,
    }
}

/// (amount, heal, n_cards) scalar features of an action.
fn action_scalars(action: &SimpleAction) -> (f32, f32, f32) {
    match action {
        SimpleAction::DrawCard { amount } => (*amount as f32, 0.0, 0.0),
        SimpleAction::Attach { attachments, .. } => {
            let total: u32 = attachments.iter().map(|(amount, _, _)| *amount).sum();
            (total as f32, 0.0, 0.0)
        }
        SimpleAction::MoveEnergy { amount, .. } => (*amount as f32, 0.0, 0.0),
        SimpleAction::Heal { amount, .. } => (0.0, *amount as f32, 0.0),
        SimpleAction::HealAndDiscardEnergy { heal_amount, .. } => (0.0, *heal_amount as f32, 0.0),
        SimpleAction::ApplyDamage { targets, .. } => {
            let total: u32 = targets.iter().map(|(damage, _, _)| *damage).sum();
            (total as f32, 0.0, 0.0)
        }
        SimpleAction::ScheduleDelayedSpotDamage { amount, .. } => (*amount as f32, 0.0, 0.0),
        SimpleAction::ShufflePokemonIntoDeck { hand_pokemon } => {
            (0.0, 0.0, hand_pokemon.len() as f32)
        }
        SimpleAction::ShuffleOwnCardsIntoDeck { cards }
        | SimpleAction::DiscardOwnCards { cards } => (0.0, 0.0, cards.len() as f32),
        SimpleAction::AttachFromDiscard {
            num_random_energies,
            ..
        } => (*num_random_energies as f32, 0.0, 0.0),
        _ => (0.0, 0.0, 0.0),
    }
}

/// Vocabulary of the card IDs reachable in a match between two decks. Local
/// indices are match-specific; `global_ids` maps each local slot to the
/// engine-wide card index so a network can look up per-card embeddings.
#[derive(Debug, Clone)]
pub struct CardVocab {
    ids: Vec<String>,
    index: HashMap<String, usize>,
    global_ids: Vec<u32>,
}

impl CardVocab {
    pub fn from_decks(deck_a: &Deck, deck_b: &Deck) -> Self {
        let mut cards: Vec<&Card> = deck_a.cards.iter().chain(deck_b.cards.iter()).collect();
        cards.sort_by_key(|c| c.get_id());
        cards.dedup_by_key(|c| c.get_id());
        assert!(
            cards.len() <= VOCAB_SIZE,
            "matchup has {} distinct cards, exceeding VOCAB_SIZE={VOCAB_SIZE}",
            cards.len()
        );
        let ids: Vec<String> = cards.iter().map(|c| c.get_id()).collect();
        let index = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        let global_ids = cards.iter().map(|c| global_card_index(c)).collect();
        CardVocab {
            ids,
            index,
            global_ids,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn index_of(&self, card: &Card) -> Option<usize> {
        self.index.get(&card.get_id()).copied()
    }

    /// Global card index per local slot, padded to `VOCAB_SIZE` with the
    /// reserved padding id (`num_global_cards()`).
    fn write_global_ids(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), VOCAB_SIZE);
        let pad = num_global_cards() as f32;
        for (slot, value) in out.iter_mut().enumerate() {
            *value = self.global_ids.get(slot).map_or(pad, |&id| id as f32);
        }
    }
}

/// Single-agent RL view of one game.
///
/// With an opponent bot (`opponent: Some(..)`), the learner is seat 0 and the
/// bot plays seat 1 inside `step`. With `opponent: None` (self-play mode) the
/// env pauses at *every* non-forced decision and reports whose turn it is via
/// `pending_actor`; the caller supplies both seats' actions. Rewards and
/// outcomes are always from seat 0's perspective.
pub struct RlEnvCore {
    /// Per-seat deck pools; each episode samples one deck per seat, so a
    /// single env can train across every pairing of the pools.
    deck_pool_a: Vec<Deck>,
    deck_pool_b: Vec<Deck>,
    /// Engine bot for seat 1 (rebuilt per episode with the sampled decks);
    /// `None` = self-play mode, the caller supplies both seats' actions.
    opponent_code: Option<PlayerCode>,
    deck_a: Deck,
    deck_b: Deck,
    opponent: Option<Box<dyn Player>>,
    vocab: CardVocab,
    rng: StdRng,
    state: State,
    pending_actions: Vec<Action>,
    pending_actor: usize,
    /// Potential-based shaping coefficient on the point differential (0 = pure win/loss).
    shaping_coef: f32,
    prev_potential: f32,
}

pub const AGENT: usize = 0;
pub const OPPONENT: usize = 1;

/// Step outcome: shaped reward, episode-done flag, and the raw game outcome
/// from the agent's perspective (+1 win / -1 loss / 0 tie) when done.
pub struct StepResult {
    pub reward: f32,
    pub done: bool,
    pub outcome: i8,
}

impl RlEnvCore {
    /// Single fixed matchup (the original behavior).
    pub fn new(
        deck_a: Deck,
        deck_b: Deck,
        opponent_code: Option<PlayerCode>,
        seed: u64,
        shaping_coef: f32,
    ) -> Self {
        Self::new_with_pools(
            vec![deck_a],
            vec![deck_b],
            opponent_code,
            seed,
            shaping_coef,
        )
    }

    /// Multi-deck env: each episode samples one deck per seat from that
    /// seat's pool, so training covers every pairing of the pools.
    pub fn new_with_pools(
        deck_pool_a: Vec<Deck>,
        deck_pool_b: Vec<Deck>,
        opponent_code: Option<PlayerCode>,
        seed: u64,
        shaping_coef: f32,
    ) -> Self {
        assert!(
            !deck_pool_a.is_empty() && !deck_pool_b.is_empty(),
            "deck pools must be non-empty"
        );
        let deck_a = deck_pool_a[0].clone();
        let deck_b = deck_pool_b[0].clone();
        let vocab = CardVocab::from_decks(&deck_a, &deck_b);
        let mut env = RlEnvCore {
            deck_pool_a,
            deck_pool_b,
            opponent_code,
            deck_a,
            deck_b,
            opponent: None,
            vocab,
            rng: StdRng::seed_from_u64(seed),
            state: State::default(),
            pending_actions: Vec::new(),
            pending_actor: AGENT,
            shaping_coef,
            prev_potential: 0.0,
        };
        env.reset_internal();
        env
    }

    pub fn vocab(&self) -> &CardVocab {
        &self.vocab
    }

    pub fn obs_dim(&self) -> usize {
        // globals + 8 board slots + 6 card-count sections (hands/discards/
        // decks) + the vocabulary's global card ids (for embedding lookup).
        58 + 2 * NUM_SLOTS * (23 + VOCAB_SIZE) + 6 * VOCAB_SIZE + VOCAB_SIZE
    }

    pub fn action_feat_dim(&self) -> usize {
        NUM_ACTION_CLASSES + VOCAB_SIZE + NUM_SLOTS + 1 + 4 + NUM_ENERGY_TYPES + 3
    }

    pub fn num_actions(&self) -> usize {
        self.pending_actions.len()
    }

    /// Seat that must act next (always `AGENT` when an opponent bot is set).
    pub fn pending_actor(&self) -> usize {
        self.pending_actor
    }

    /// Ask an external bot to choose among the pending actions, returning
    /// the chosen index (does not step the env). Used in self-play mode to
    /// mix engine bots into the opponent roster without installing them.
    pub fn decide_with(&mut self, player: &mut dyn Player) -> usize {
        let action = player.decision_fn(&mut self.rng, &self.state, &self.pending_actions);
        self.pending_actions
            .iter()
            .position(|a| *a == action)
            .expect("bot returned an action that is not in the legal list")
    }

    pub fn legal_action_strings(&self) -> Vec<String> {
        self.pending_actions
            .iter()
            .map(|a| a.action.to_string())
            .collect()
    }

    /// Determinized one-ply action values from the pending actor's
    /// hidden-information perspective: for each legal action, average the
    /// engine's baseline value function over `determinizations` samples of
    /// the hidden information (the opponent's hand/deck split and both deck
    /// orders are re-dealt uniformly from what the actor could know; chance
    /// inside the action resolves once per determinization). Raw engine
    /// value scale (wins ~1e5, points ~1e4) — normalize downstream. This is
    /// test-time search: it reads only information a real player has.
    pub fn action_values(&self, determinizations: usize, seed: u64) -> Vec<f32> {
        use rand::seq::SliceRandom;
        let actor = self.pending_actor;
        let opponent = 1 - actor;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut values = vec![0.0f64; self.pending_actions.len()];
        for _ in 0..determinizations {
            let mut base = self.state.clone();
            // The actor knows its own deck's composition but not its order.
            base.decks[actor].cards.shuffle(&mut rng);
            // The opponent's hand/deck split is hidden; re-deal it from the
            // combined unseen pool (composition is public via the decklist).
            let hand_size = base.hands[opponent].len();
            let mut pile: Vec<Card> = base.hands[opponent].drain(..).collect();
            pile.append(&mut base.decks[opponent].cards);
            pile.shuffle(&mut rng);
            base.decks[opponent].cards = pile.split_off(hand_size);
            base.hands[opponent] = pile;
            for (i, action) in self.pending_actions.iter().enumerate() {
                // Some engine-generated actions reference specific cards in
                // the opponent's (hidden) hand; after re-dealing, applying
                // them can panic. Score the unmodified determinization for
                // those rare actions instead of crashing.
                let mut next = base.clone();
                let seed = rng.gen::<u64>();
                let applied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut local_rng = StdRng::seed_from_u64(seed);
                    apply_action(&mut local_rng, &mut next, action);
                }))
                .is_ok();
                if !applied {
                    next = base.clone();
                }
                values[i] += match next.winner {
                    Some(GameOutcome::Win(p)) if p == actor => 200_000.0,
                    Some(GameOutcome::Win(_)) => -200_000.0,
                    Some(GameOutcome::Tie) => 0.0,
                    None => crate::players::baseline_value_function(&next, actor),
                };
            }
        }
        values
            .into_iter()
            .map(|v| (v / determinizations as f64) as f32)
            .collect()
    }

    pub fn reset(&mut self) {
        self.reset_internal();
    }

    fn reset_internal(&mut self) {
        // A freshly dealt game can, very rarely, roll into a terminal state
        // (engine deadlock scored as a tie) before anyone gets a real
        // decision; re-deal until the episode is live.
        loop {
            if self.deck_pool_a.len() > 1 {
                let idx = self.rng.gen_range(0..self.deck_pool_a.len());
                self.deck_a = self.deck_pool_a[idx].clone();
            }
            if self.deck_pool_b.len() > 1 {
                let idx = self.rng.gen_range(0..self.deck_pool_b.len());
                self.deck_b = self.deck_pool_b[idx].clone();
            }
            self.vocab = CardVocab::from_decks(&self.deck_a, &self.deck_b);
            // The bot holds a deck; rebuild it for the sampled matchup.
            self.opponent = self.opponent_code.clone().map(|code| {
                create_players(
                    self.deck_a.clone(),
                    self.deck_b.clone(),
                    vec![code.clone(), code],
                )
                .into_iter()
                .nth(1)
                .unwrap()
            });
            self.state = State::initialize(&self.deck_a, &self.deck_b, &mut self.rng);
            self.prev_potential = 0.0;
            self.advance_until_agent_decision();
            if !self.state.is_game_over() && !self.pending_actions.is_empty() {
                return;
            }
        }
    }

    /// Ask a freshly built engine bot (for the current matchup) to choose
    /// among the pending actions, returning the chosen index without
    /// stepping. Used to mix engine bots into self-play opponent rosters
    /// and to collect demonstration games.
    pub fn decide_with_code(&mut self, code: &PlayerCode) -> usize {
        let mut bot = create_players(
            self.deck_a.clone(),
            self.deck_b.clone(),
            vec![code.clone(), code.clone()],
        )
        .into_iter()
        .nth(1)
        .unwrap();
        self.decide_with(bot.as_mut())
    }

    /// Apply the `idx`-th legal action for the pending actor, then roll the
    /// game forward (bot moves and forced actions) until someone must decide
    /// again or the game ends. Resets automatically when the episode ends, so
    /// the env is always left at a decision point of a live game. Rewards and
    /// outcomes are from seat 0's perspective regardless of who acted.
    pub fn step(&mut self, idx: usize) -> StepResult {
        assert!(
            idx < self.pending_actions.len(),
            "action index {idx} out of range ({} legal actions)",
            self.pending_actions.len()
        );
        let action = self.pending_actions[idx].clone();
        apply_action(&mut self.rng, &mut self.state, &action);
        self.advance_until_agent_decision();

        let (done, outcome) = match self.state.winner {
            Some(GameOutcome::Win(player)) => (true, if player == AGENT { 1 } else { -1 }),
            Some(GameOutcome::Tie) => (true, 0),
            None => (false, 0),
        };

        // Potential-based shaping on the point differential: r += Φ(s') - Φ(s),
        // with Φ(terminal) = 0 so the shaped return telescopes to the true one.
        let potential = if done {
            0.0
        } else {
            self.shaping_coef
                * (self.state.points[AGENT] as f32 - self.state.points[OPPONENT] as f32)
                / MAX_POINTS
        };
        let reward = if done { outcome as f32 } else { 0.0 } + potential - self.prev_potential;
        self.prev_potential = potential;

        if done {
            self.reset_internal();
        }
        StepResult {
            reward,
            done,
            outcome,
        }
    }

    fn advance_until_agent_decision(&mut self) {
        loop {
            if self.state.is_game_over() {
                self.pending_actions.clear();
                return;
            }
            let (actor, actions) = self.state.generate_possible_actions();
            if actions.is_empty() {
                // Engine deadlock: a non-terminal state with no legal moves
                // (rare card-interaction edge). Score it as a tie instead of
                // leaving the env parked at a zero-action decision point.
                self.state.winner = Some(GameOutcome::Tie);
                continue;
            }
            if actions.len() == 1 {
                apply_action(&mut self.rng, &mut self.state, &actions[0]);
            } else if actor == OPPONENT && self.opponent.is_some() {
                let action = self.opponent.as_mut().unwrap().decision_fn(
                    &mut self.rng,
                    &self.state,
                    &actions,
                );
                apply_action(&mut self.rng, &mut self.state, &action);
            } else {
                assert!(
                    actions.len() <= MAX_ACTIONS,
                    "{} legal actions exceed MAX_ACTIONS={MAX_ACTIONS}",
                    actions.len()
                );
                self.pending_actions = actions;
                self.pending_actor = actor;
                return;
            }
        }
    }

    /// Write the observation from `perspective`'s point of view into `out`
    /// (must be `obs_dim()` long): "my" sections come first, "theirs" second.
    ///
    /// With `include_hidden = false` the sections a real player cannot see —
    /// the other seat's hand contents and deck composition — are left zeroed
    /// (hand/deck *sizes* stay visible in the globals). The oracle variant
    /// (`include_hidden = true`) is intended for a training-time critic.
    pub fn write_observation(&self, out: &mut [f32], perspective: usize, include_hidden: bool) {
        debug_assert_eq!(out.len(), self.obs_dim());
        out.fill(0.0);
        let state = &self.state;
        let me = perspective;
        let them = 1 - perspective;
        let v = VOCAB_SIZE;
        let mut i = 0;

        // Globals (58)
        out[i] = state.points[me] as f32 / MAX_POINTS;
        out[i + 1] = state.points[them] as f32 / MAX_POINTS;
        out[i + 2] = state.turn_count as f32 / MAX_TURNS;
        out[i + 3] = (state.turn_count <= 2) as u8 as f32;
        i += 4;
        for player in [me, them] {
            let zone = &state.energy_zone[player];
            if let Some(energy) = zone.current {
                out[i + energy_index(energy)] = 1.0;
                out[i + NUM_ENERGY_TYPES] = 1.0;
            }
            i += NUM_ENERGY_TYPES + 1;
            if let Some(energy) = zone.next {
                out[i + energy_index(energy)] = 1.0;
                out[i + NUM_ENERGY_TYPES] = 1.0;
            }
            i += NUM_ENERGY_TYPES + 1;
        }
        for player in [me, them] {
            out[i] = state.hands[player].len() as f32 / 10.0;
            out[i + 1] = state.decks[player].cards.len() as f32 / 20.0;
            out[i + 2] = state.discard_piles[player].len() as f32 / 20.0;
            i += 3;
        }
        out[i] = state.has_played_support as u8 as f32;
        out[i + 1] = state.has_retreated as u8 as f32;
        i += 2;
        out[i] = state.active_stadium.is_some() as u8 as f32;
        out[i + 1] = (state.active_stadium_owner == Some(me)) as u8 as f32;
        i += 2;

        // Board slots: my 4 then theirs, (23 + V) each
        for player in [me, them] {
            for slot in 0..NUM_SLOTS {
                if let Some(played) = &state.in_play_pokemon[player][slot] {
                    out[i] = 1.0;
                    if let Some(card_idx) = self.vocab.index_of(&played.card) {
                        out[i + 1 + card_idx] = 1.0;
                    }
                    let mut j = i + 1 + v;
                    let total_hp = played.get_effective_total_hp();
                    out[j] = played.get_remaining_hp() as f32 / total_hp.max(1) as f32;
                    out[j + 1] = total_hp as f32 / 300.0;
                    j += 2;
                    for energy in &played.attached_energy {
                        out[j + energy_index(*energy)] += 0.25;
                    }
                    j += NUM_ENERGY_TYPES;
                    out[j] = played.is_poisoned() as u8 as f32;
                    out[j + 1] = played.is_paralyzed() as u8 as f32;
                    out[j + 2] = played.is_asleep() as u8 as f32;
                    out[j + 3] = played.is_burned() as u8 as f32;
                    out[j + 4] = played.is_confused() as u8 as f32;
                    j += NUM_STATUS;
                    if let Card::Pokemon(pokemon) = &played.card {
                        out[j] = pokemon.stage as f32 / 2.0;
                    }
                    out[j + 1] = played.card.is_ex() as u8 as f32;
                    out[j + 2] = played.played_this_turn as u8 as f32;
                    out[j + 3] = played.ability_used as u8 as f32;
                    out[j + 4] = played.attached_tool.is_some() as u8 as f32;
                }
                i += 23 + v;
            }
        }

        // Card counts per vocab entry: hands, discards, decks (6 × V).
        // The other seat's hand and deck composition are hidden information.
        for player in [me, them] {
            if player == me || include_hidden {
                count_cards(&state.hands[player], &self.vocab, &mut out[i..i + v]);
            }
            i += v;
        }
        for player in [me, them] {
            count_cards(
                &state.discard_piles[player],
                &self.vocab,
                &mut out[i..i + v],
            );
            i += v;
        }
        for player in [me, them] {
            if player == me || include_hidden {
                count_cards(&state.decks[player].cards, &self.vocab, &mut out[i..i + v]);
            }
            i += v;
        }

        // Global card ids of the vocabulary slots (as f32 — exact for ids
        // below 2^24). This tells an embedding-based network *which* cards
        // the local one-hot/count sections refer to. Both decklists are
        // considered public knowledge (like the metagame), but the
        // opponent's hand and remaining deck contents stay hidden above.
        self.vocab.write_global_ids(&mut out[i..i + VOCAB_SIZE]);
        i += VOCAB_SIZE;
        debug_assert_eq!(i, self.obs_dim());
    }

    /// Write padded action features into `out` (must be `MAX_ACTIONS * action_feat_dim()` long).
    pub fn write_action_features(&self, out: &mut [f32]) {
        let dim = self.action_feat_dim();
        debug_assert_eq!(out.len(), MAX_ACTIONS * dim);
        out.fill(0.0);
        let v = VOCAB_SIZE;
        for (slot, action) in self.pending_actions.iter().enumerate() {
            let simple = &action.action;
            let row = &mut out[slot * dim..(slot + 1) * dim];
            row[action_class(simple)] = 1.0;
            let mut j = NUM_ACTION_CLASSES;
            if let Some(card) = action_card(simple) {
                if let Some(card_idx) = self.vocab.index_of(&card) {
                    row[j + card_idx] = 1.0;
                }
            }
            j += v;
            if let Some((idx, target_player)) = action_target(simple) {
                if idx < NUM_SLOTS {
                    row[j + idx] = 1.0;
                }
                let other_side = target_player.is_some_and(|p| p != self.pending_actor);
                row[j + NUM_SLOTS] = other_side as u8 as f32;
            }
            j += NUM_SLOTS + 1;
            if let SimpleAction::Attack(attack_idx) = simple {
                if *attack_idx < 4 {
                    row[j + attack_idx] = 1.0;
                }
            }
            if let SimpleAction::UseCopiedAttack { attack_index, .. } = simple {
                if *attack_index < 4 {
                    row[j + attack_index] = 1.0;
                }
            }
            j += 4;
            if let Some(energy) = action_energy(simple) {
                row[j + energy_index(energy)] = 1.0;
            }
            j += NUM_ENERGY_TYPES;
            let (amount, heal, n_cards) = action_scalars(simple);
            row[j] = amount / 4.0;
            row[j + 1] = heal / 100.0;
            row[j + 2] = n_cards / 3.0;
        }
    }
}

fn count_cards(cards: &[Card], vocab: &CardVocab, out: &mut [f32]) {
    for card in cards {
        if let Some(idx) = vocab.index_of(card) {
            out[idx] += 1.0 / MAX_COPIES;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_env(seed: u64) -> RlEnvCore {
        let deck = Deck::from_file("example_decks/venusaur-exeggutor.txt").unwrap();
        RlEnvCore::new(deck.clone(), deck, Some(PlayerCode::R), seed, 0.0)
    }

    #[test]
    fn test_env_runs_random_episodes() {
        let mut env = make_env(7);
        let obs_dim = env.obs_dim();
        let feat_dim = env.action_feat_dim();
        let mut obs = vec![0.0; obs_dim];
        let mut feats = vec![0.0; MAX_ACTIONS * feat_dim];
        let mut rng = StdRng::seed_from_u64(42);
        let mut episodes = 0;
        let mut total_reward = 0.0;
        let mut steps = 0;
        while episodes < 20 {
            assert!(env.num_actions() >= 2, "env must pause on real decisions");
            env.write_observation(&mut obs, AGENT, false);
            env.write_action_features(&mut feats);
            assert!(obs.iter().all(|x| x.is_finite()));
            // The hidden (policy) view must reveal no more than the oracle view.
            let mut oracle = vec![0.0; obs_dim];
            env.write_observation(&mut oracle, AGENT, true);
            assert!(obs.iter().zip(&oracle).all(|(p, o)| *p == 0.0 || p == o));
            let idx = rng.gen_range(0..env.num_actions());
            let result = env.step(idx);
            total_reward += result.reward;
            steps += 1;
            if result.done {
                episodes += 1;
                assert!((-1..=1).contains(&result.outcome));
            }
            assert!(steps < 20_000, "episodes should terminate");
        }
        // Pure win/loss rewards: total must equal the sum of outcomes (each in [-1, 1]).
        assert!(total_reward.abs() <= episodes as f32);
    }

    #[test]
    fn test_selfplay_mode_pauses_at_both_seats() {
        let deck = Deck::from_file("example_decks/venusaur-exeggutor.txt").unwrap();
        let mut env = RlEnvCore::new(deck.clone(), deck, None, 5, 0.0);
        let mut rng = StdRng::seed_from_u64(17);
        let mut seen_seats = [false, false];
        let mut episodes = 0;
        let mut steps = 0;
        let mut obs = vec![0.0; env.obs_dim()];
        while episodes < 10 {
            let actor = env.pending_actor();
            seen_seats[actor] = true;
            // Perspective view must be valid for whoever acts.
            env.write_observation(&mut obs, actor, false);
            assert!(obs.iter().all(|x| x.is_finite()));
            let idx = rng.gen_range(0..env.num_actions());
            let result = env.step(idx);
            if result.done {
                episodes += 1;
            }
            steps += 1;
            assert!(steps < 30_000, "episodes should terminate");
        }
        assert!(
            seen_seats[AGENT] && seen_seats[OPPONENT],
            "self-play mode must pause for both seats"
        );
    }

    #[test]
    fn test_multi_deck_pool_sampling() {
        let decks: Vec<Deck> = [
            "example_decks/venusaur-exeggutor.txt",
            "example_decks/weezing-arbok.txt",
            "example_decks/mewtwoex.txt",
        ]
        .iter()
        .map(|p| Deck::from_file(p).unwrap())
        .collect();
        let mut env = RlEnvCore::new_with_pools(decks.clone(), decks, Some(PlayerCode::R), 3, 0.0);
        let obs_dim = env.obs_dim();
        let mut obs = vec![0.0; obs_dim];
        let mut rng = StdRng::seed_from_u64(1);
        let mut vocab_signatures = std::collections::HashSet::new();
        let mut episodes = 0;
        let mut steps = 0;
        while episodes < 12 {
            env.write_observation(&mut obs, AGENT, false);
            // Dims are fixed regardless of the sampled matchup; the id
            // section identifies the cards, padded with num_global_cards().
            let ids = &obs[obs_dim - VOCAB_SIZE..];
            assert!(ids
                .iter()
                .all(|&id| id >= 0.0 && id <= num_global_cards() as f32));
            vocab_signatures.insert(ids.iter().map(|&id| id as u32).collect::<Vec<_>>());
            let idx = rng.gen_range(0..env.num_actions());
            if env.step(idx).done {
                episodes += 1;
            }
            steps += 1;
            assert!(steps < 20_000, "episodes should terminate");
        }
        // 3x3 deck pairings: several distinct vocabularies must appear.
        assert!(
            vocab_signatures.len() >= 3,
            "expected multiple matchup vocabularies, saw {}",
            vocab_signatures.len()
        );
    }

    #[test]
    fn test_card_attr_table_shape() {
        let table = card_attr_table();
        let n = num_global_cards();
        assert_eq!(table.len(), (n + 1) * CARD_ATTR_DIM);
        // Padding row is all zeros.
        assert!(table[n * CARD_ATTR_DIM..].iter().all(|&x| x == 0.0));
        // Every real row is finite and at least one field is set.
        assert!(table.iter().all(|x| x.is_finite()));
        assert_eq!(global_card_ids().len(), n);
    }

    #[test]
    fn test_shaping_telescopes() {
        // With shaping, an episode's total reward must still equal the outcome.
        let deck = Deck::from_file("example_decks/venusaur-exeggutor.txt").unwrap();
        let mut env = RlEnvCore::new(deck.clone(), deck, Some(PlayerCode::R), 11, 0.5);
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..5 {
            let mut episode_reward = 0.0;
            loop {
                let idx = rng.gen_range(0..env.num_actions());
                let result = env.step(idx);
                episode_reward += result.reward;
                if result.done {
                    assert!(
                        (episode_reward - result.outcome as f32).abs() < 1e-5,
                        "shaped return {episode_reward} != outcome {}",
                        result.outcome
                    );
                    break;
                }
            }
        }
    }

    #[test]
    fn test_deterministic_given_seed() {
        let mut env_a = make_env(123);
        let mut env_b = make_env(123);
        let mut obs_a = vec![0.0; env_a.obs_dim()];
        let mut obs_b = vec![0.0; env_b.obs_dim()];
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..200 {
            assert_eq!(env_a.num_actions(), env_b.num_actions());
            env_a.write_observation(&mut obs_a, AGENT, true);
            env_b.write_observation(&mut obs_b, AGENT, true);
            assert_eq!(obs_a, obs_b);
            let idx = rng.gen_range(0..env_a.num_actions());
            let result_a = env_a.step(idx);
            let result_b = env_b.step(idx);
            assert_eq!(result_a.done, result_b.done);
            assert_eq!(result_a.outcome, result_b.outcome);
        }
    }
}
