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

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::actions::{apply_action, Action, SimpleAction};
use crate::deck::Deck;
use crate::models::{Card, EnergyType};
use crate::players::Player;
use crate::state::GameOutcome;
use crate::State;

/// Hard cap on legal actions encoded per decision. The engine rarely exceeds
/// a few dozen; exceeding the cap is a hard error so it is never silent.
pub const MAX_ACTIONS: usize = 128;

const NUM_ENERGY_TYPES: usize = 10;
const NUM_STATUS: usize = 5;
const NUM_ACTION_CLASSES: usize = 18;
const NUM_SLOTS: usize = 4; // active + 3 bench
const MAX_POINTS: f32 = 3.0;
const MAX_TURNS: f32 = 30.0;
const MAX_COPIES: f32 = 2.0;

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

/// (in_play_idx, targets_opponent) referenced by an action, if any.
fn action_target(action: &SimpleAction) -> Option<(usize, bool)> {
    match action {
        SimpleAction::Place(_, idx) => Some((*idx, false)),
        SimpleAction::Evolve { in_play_idx, .. }
        | SimpleAction::UseAbility { in_play_idx }
        | SimpleAction::AttachTool { in_play_idx, .. }
        | SimpleAction::Heal { in_play_idx, .. }
        | SimpleAction::HealAndDiscardEnergy { in_play_idx, .. }
        | SimpleAction::AttachFromDiscard { in_play_idx, .. }
        | SimpleAction::DiscardFossil { in_play_idx }
        | SimpleAction::ReturnPokemonToHand { in_play_idx }
        | SimpleAction::ShuffleInPlayPokemonIntoDeck { in_play_idx } => {
            Some((*in_play_idx, false))
        }
        SimpleAction::Retreat(idx) => Some((*idx, false)),
        SimpleAction::Activate { in_play_idx, .. } => Some((*in_play_idx, false)),
        SimpleAction::MoveEnergy { to_in_play_idx, .. } => Some((*to_in_play_idx, false)),
        SimpleAction::Attach { attachments, .. } => {
            attachments.first().map(|(_, _, idx)| (*idx, false))
        }
        SimpleAction::ApplyDamage { targets, .. } => targets
            .first()
            .map(|(_, target_player, idx)| (*idx, *target_player == 1)),
        SimpleAction::ScheduleDelayedSpotDamage {
            target_player,
            target_in_play_idx,
            ..
        } => Some((*target_in_play_idx, *target_player == 1)),
        SimpleAction::DiscardToolFromPokemon { player, in_play_idx } => {
            Some((*in_play_idx, *player == 1))
        }
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

/// Vocabulary of the card IDs reachable in a match between two decks.
#[derive(Debug, Clone)]
pub struct CardVocab {
    ids: Vec<String>,
    index: HashMap<String, usize>,
}

impl CardVocab {
    pub fn from_decks(deck_a: &Deck, deck_b: &Deck) -> Self {
        let mut ids: Vec<String> = deck_a
            .cards
            .iter()
            .chain(deck_b.cards.iter())
            .map(|c| c.get_id())
            .collect();
        ids.sort();
        ids.dedup();
        let index = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        CardVocab { ids, index }
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
}

/// Single-agent RL view of one game. Agent is always seat 0.
pub struct RlEnvCore {
    deck_a: Deck,
    deck_b: Deck,
    opponent: Box<dyn Player>,
    vocab: CardVocab,
    rng: StdRng,
    state: State,
    pending_actions: Vec<Action>,
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
    pub fn new(
        deck_a: Deck,
        deck_b: Deck,
        opponent: Box<dyn Player>,
        seed: u64,
        shaping_coef: f32,
    ) -> Self {
        let vocab = CardVocab::from_decks(&deck_a, &deck_b);
        let mut env = RlEnvCore {
            deck_a,
            deck_b,
            opponent,
            vocab,
            rng: StdRng::seed_from_u64(seed),
            state: State::default(),
            pending_actions: Vec::new(),
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
        let v = self.vocab.len();
        // globals + 8 board slots + 6 card-count sections (hands/discards/decks)
        58 + 2 * NUM_SLOTS * (23 + v) + 6 * v
    }

    pub fn action_feat_dim(&self) -> usize {
        NUM_ACTION_CLASSES + self.vocab.len() + NUM_SLOTS + 1 + 4 + NUM_ENERGY_TYPES + 3
    }

    pub fn num_actions(&self) -> usize {
        self.pending_actions.len()
    }

    pub fn legal_action_strings(&self) -> Vec<String> {
        self.pending_actions
            .iter()
            .map(|a| a.action.to_string())
            .collect()
    }

    pub fn reset(&mut self) {
        self.reset_internal();
    }

    fn reset_internal(&mut self) {
        self.state = State::initialize(&self.deck_a, &self.deck_b, &mut self.rng);
        self.prev_potential = 0.0;
        self.advance_until_agent_decision();
        debug_assert!(!self.state.is_game_over());
    }

    /// Apply the `idx`-th legal action, then roll the game forward (opponent
    /// moves and forced actions) until the agent must decide again or the
    /// game ends. Resets automatically when the episode ends, so the env is
    /// always left at a decision point of a live game.
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
            if actions.len() == 1 {
                apply_action(&mut self.rng, &mut self.state, &actions[0]);
            } else if actor == OPPONENT {
                let action = self
                    .opponent
                    .decision_fn(&mut self.rng, &self.state, &actions);
                apply_action(&mut self.rng, &mut self.state, &action);
            } else {
                assert!(
                    actions.len() <= MAX_ACTIONS,
                    "{} legal actions exceed MAX_ACTIONS={MAX_ACTIONS}",
                    actions.len()
                );
                self.pending_actions = actions;
                return;
            }
        }
    }

    /// Write the observation into `out` (must be `obs_dim()` long).
    pub fn write_observation(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.obs_dim());
        out.fill(0.0);
        let state = &self.state;
        let v = self.vocab.len();
        let mut i = 0;

        // Globals (58)
        out[i] = state.points[AGENT] as f32 / MAX_POINTS;
        out[i + 1] = state.points[OPPONENT] as f32 / MAX_POINTS;
        out[i + 2] = state.turn_count as f32 / MAX_TURNS;
        out[i + 3] = (state.turn_count <= 2) as u8 as f32;
        i += 4;
        for player in [AGENT, OPPONENT] {
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
        for player in [AGENT, OPPONENT] {
            out[i] = state.hands[player].len() as f32 / 10.0;
            out[i + 1] = state.decks[player].cards.len() as f32 / 20.0;
            out[i + 2] = state.discard_piles[player].len() as f32 / 20.0;
            i += 3;
        }
        out[i] = state.has_played_support as u8 as f32;
        out[i + 1] = state.has_retreated as u8 as f32;
        i += 2;
        out[i] = state.active_stadium.is_some() as u8 as f32;
        out[i + 1] = (state.active_stadium_owner == Some(AGENT)) as u8 as f32;
        i += 2;

        // Board slots: agent's 4 then opponent's 4, (23 + V) each
        for player in [AGENT, OPPONENT] {
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

        // Card counts per vocab entry: hands, discards, decks (6 × V)
        for player in [AGENT, OPPONENT] {
            count_cards(&state.hands[player], &self.vocab, &mut out[i..i + v]);
            i += v;
        }
        for player in [AGENT, OPPONENT] {
            count_cards(&state.discard_piles[player], &self.vocab, &mut out[i..i + v]);
            i += v;
        }
        for player in [AGENT, OPPONENT] {
            count_cards(&state.decks[player].cards, &self.vocab, &mut out[i..i + v]);
            i += v;
        }
        debug_assert_eq!(i, self.obs_dim());
    }

    /// Write padded action features into `out` (must be `MAX_ACTIONS * action_feat_dim()` long).
    pub fn write_action_features(&self, out: &mut [f32]) {
        let dim = self.action_feat_dim();
        debug_assert_eq!(out.len(), MAX_ACTIONS * dim);
        out.fill(0.0);
        let v = self.vocab.len();
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
            if let Some((idx, opponent_side)) = action_target(simple) {
                if idx < NUM_SLOTS {
                    row[j + idx] = 1.0;
                }
                row[j + NUM_SLOTS] = opponent_side as u8 as f32;
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
    use crate::players::RandomPlayer;
    use rand::Rng;

    fn make_env(seed: u64) -> RlEnvCore {
        let deck = Deck::from_file("example_decks/venusaur-exeggutor.txt").unwrap();
        let opponent = Box::new(RandomPlayer { deck: deck.clone() });
        RlEnvCore::new(deck.clone(), deck, opponent, seed, 0.0)
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
            env.write_observation(&mut obs);
            env.write_action_features(&mut feats);
            assert!(obs.iter().all(|x| x.is_finite()));
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
    fn test_shaping_telescopes() {
        // With shaping, an episode's total reward must still equal the outcome.
        let deck = Deck::from_file("example_decks/venusaur-exeggutor.txt").unwrap();
        let opponent = Box::new(RandomPlayer { deck: deck.clone() });
        let mut env = RlEnvCore::new(deck.clone(), deck, opponent, 11, 0.5);
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
            env_a.write_observation(&mut obs_a);
            env_b.write_observation(&mut obs_b);
            assert_eq!(obs_a, obs_b);
            let idx = rng.gen_range(0..env_a.num_actions());
            let result_a = env_a.step(idx);
            let result_b = env_b.step(idx);
            assert_eq!(result_a.done, result_b.done);
            assert_eq!(result_a.outcome, result_b.outcome);
        }
    }
}
