#![allow(clippy::useless_conversion)]

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::{
    deck::Deck,
    game::Game,
    models::{Ability, Attack, Card, EnergyType, PlayedCard},
    players::{create_players, fill_code_array, parse_player_code, PlayerCode},
    rl_env::{RlEnvCore, MAX_ACTIONS},
    state::{GameOutcome, State},
};

/// Python wrapper for EnergyType
#[pyclass]
#[derive(Clone, Copy)]
pub struct PyEnergyType {
    energy_type: EnergyType,
}

#[pymethods]
impl PyEnergyType {
    fn __repr__(&self) -> String {
        format!("{:?}", self.energy_type)
    }

    fn __str__(&self) -> String {
        format!("{}", self.energy_type)
    }

    #[getter]
    fn name(&self) -> String {
        format!("{:?}", self.energy_type)
    }
}

impl From<EnergyType> for PyEnergyType {
    fn from(energy_type: EnergyType) -> Self {
        PyEnergyType { energy_type }
    }
}

/// Python wrapper for Attack
#[pyclass]
#[derive(Clone)]
pub struct PyAttack {
    attack: Attack,
}

#[pymethods]
impl PyAttack {
    #[getter]
    fn title(&self) -> String {
        self.attack.title.clone()
    }

    #[getter]
    fn fixed_damage(&self) -> u32 {
        self.attack.fixed_damage
    }

    #[getter]
    fn effect(&self) -> Option<String> {
        self.attack.effect.clone()
    }

    #[getter]
    fn energy_required(&self) -> Vec<PyEnergyType> {
        self.attack
            .energy_required
            .iter()
            .map(|&e| e.into())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Attack(title='{}', damage={}, effect={:?})",
            self.attack.title, self.attack.fixed_damage, self.attack.effect
        )
    }
}

impl From<Attack> for PyAttack {
    fn from(attack: Attack) -> Self {
        PyAttack { attack }
    }
}

/// Python wrapper for Ability
#[pyclass]
#[derive(Clone)]
pub struct PyAbility {
    #[pyo3(get)]
    pub title: String,
    #[pyo3(get)]
    pub effect: String,
}

#[pymethods]
impl PyAbility {
    fn __repr__(&self) -> String {
        format!("Ability(title='{}', effect='{}')", self.title, self.effect)
    }
}

impl From<Ability> for PyAbility {
    fn from(ability: Ability) -> Self {
        PyAbility {
            title: ability.title,
            effect: ability.effect,
        }
    }
}

/// Python wrapper for Card
#[pyclass]
#[derive(Clone)]
pub struct PyCard {
    card: Card,
}

#[pymethods]
impl PyCard {
    #[getter]
    fn id(&self) -> String {
        self.card.get_id()
    }

    #[getter]
    fn name(&self) -> String {
        self.card.get_name()
    }

    #[getter]
    fn is_pokemon(&self) -> bool {
        matches!(self.card, Card::Pokemon(_))
    }

    #[getter]
    fn is_trainer(&self) -> bool {
        matches!(self.card, Card::Trainer(_))
    }

    #[getter]
    fn is_basic(&self) -> bool {
        self.card.is_basic()
    }

    #[getter]
    fn is_ex(&self) -> bool {
        self.card.is_ex()
    }

    #[getter]
    fn energy_type(&self) -> Option<PyEnergyType> {
        self.card.get_type().map(|t| t.into())
    }

    #[getter]
    fn attacks(&self) -> Vec<PyAttack> {
        match &self.card {
            Card::Pokemon(_) => self
                .card
                .get_attacks()
                .iter()
                .map(|a| a.clone().into())
                .collect(),
            _ => Vec::new(),
        }
    }

    #[getter]
    fn ability(&self) -> Option<PyAbility> {
        self.card.get_ability().map(|a| a.into())
    }

    #[getter]
    fn weakness(&self) -> Option<PyEnergyType> {
        match &self.card {
            Card::Pokemon(pokemon_card) => pokemon_card.weakness.map(|w| w.into()),
            _ => None,
        }
    }

    #[getter]
    fn retreat_cost(&self) -> usize {
        match &self.card {
            Card::Pokemon(pokemon_card) => pokemon_card.retreat_cost.len(),
            _ => 0,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Card(id='{}', name='{}')",
            self.card.get_id(),
            self.card.get_name()
        )
    }
}

impl From<Card> for PyCard {
    fn from(card: Card) -> Self {
        PyCard { card }
    }
}

/// Python wrapper for PlayedCard
#[pyclass]
#[derive(Clone)]
pub struct PyPlayedCard {
    played_card: PlayedCard,
}

#[pymethods]
impl PyPlayedCard {
    #[getter]
    fn card(&self) -> PyCard {
        self.played_card.card.clone().into()
    }

    #[getter]
    fn remaining_hp(&self) -> u32 {
        self.played_card.get_remaining_hp()
    }

    #[getter]
    fn total_hp(&self) -> u32 {
        self.played_card.get_effective_total_hp()
    }

    #[getter]
    fn attached_energy(&self) -> Vec<PyEnergyType> {
        self.played_card
            .attached_energy
            .iter()
            .map(|&e| e.into())
            .collect()
    }

    #[getter]
    fn played_this_turn(&self) -> bool {
        self.played_card.played_this_turn
    }

    #[getter]
    fn ability_used(&self) -> bool {
        self.played_card.ability_used
    }

    #[getter]
    fn poisoned(&self) -> bool {
        self.played_card.is_poisoned()
    }

    #[getter]
    fn paralyzed(&self) -> bool {
        self.played_card.is_paralyzed()
    }

    #[getter]
    fn asleep(&self) -> bool {
        self.played_card.is_asleep()
    }

    #[getter]
    fn is_damaged(&self) -> bool {
        self.played_card.is_damaged()
    }

    #[getter]
    fn has_tool_attached(&self) -> bool {
        self.played_card.has_tool_attached()
    }

    #[getter]
    fn name(&self) -> String {
        self.played_card.get_name()
    }

    #[getter]
    fn energy_type(&self) -> Option<PyEnergyType> {
        self.played_card.get_energy_type().map(|t| t.into())
    }

    #[getter]
    fn attacks(&self) -> Vec<PyAttack> {
        self.played_card
            .get_attacks()
            .iter()
            .map(|a| a.clone().into())
            .collect()
    }

    #[getter]
    fn ability(&self) -> Option<PyAbility> {
        self.played_card.card.get_ability().map(|a| a.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "PlayedCard(name='{}', hp={}/{}, energy={})",
            self.played_card.get_name(),
            self.played_card.get_remaining_hp(),
            self.played_card.get_effective_total_hp(),
            self.played_card.attached_energy.len()
        )
    }
}

impl From<PlayedCard> for PyPlayedCard {
    fn from(played_card: PlayedCard) -> Self {
        PyPlayedCard { played_card }
    }
}

/// Python wrapper for GameOutcome
#[pyclass]
#[derive(Clone)]
pub struct PyGameOutcome {
    #[pyo3(get)]
    pub winner: Option<usize>,
    #[pyo3(get)]
    pub is_tie: bool,
}

impl From<GameOutcome> for PyGameOutcome {
    fn from(outcome: GameOutcome) -> Self {
        match outcome {
            GameOutcome::Win(player) => PyGameOutcome {
                winner: Some(player),
                is_tie: false,
            },
            GameOutcome::Tie => PyGameOutcome {
                winner: None,
                is_tie: true,
            },
        }
    }
}

#[pymethods]
impl PyGameOutcome {
    fn __repr__(&self) -> String {
        if self.is_tie {
            "GameOutcome::Tie".to_string()
        } else if let Some(winner) = self.winner {
            format!("GameOutcome::Win({})", winner)
        } else {
            panic!("Invalid state: PyGameOutcome has neither a winner nor a tie.");
        }
    }
}

/// Python wrapper for Deck
#[pyclass]
pub struct PyDeck {
    deck: Deck,
}

#[pymethods]
impl PyDeck {
    #[new]
    pub fn new(deck_path: &str) -> PyResult<Self> {
        let deck = Deck::from_file(deck_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load deck: {}", e))
        })?;
        Ok(PyDeck { deck })
    }

    fn __repr__(&self) -> String {
        format!("PyDeck(cards={})", self.deck.cards.len())
    }

    #[getter]
    fn card_count(&self) -> usize {
        self.deck.cards.len()
    }
}

/// Python wrapper for State
#[pyclass]
pub struct PyState {
    state: State,
}

#[pymethods]
impl PyState {
    #[getter]
    fn turn_count(&self) -> u8 {
        self.state.turn_count
    }

    #[getter]
    fn current_player(&self) -> usize {
        self.state.current_player
    }

    #[getter]
    fn points(&self) -> [u8; 2] {
        self.state.points
    }

    #[getter]
    fn winner(&self) -> Option<PyGameOutcome> {
        self.state.winner.map(|outcome| outcome.into())
    }

    #[getter]
    fn current_energy(&self) -> Option<PyEnergyType> {
        self.state.energy_zone[self.state.current_player]
            .current
            .map(|e| e.into())
    }

    /// The energy that will rotate into `current` at the start of the active player's next
    /// turn (visible preview).
    #[getter]
    fn next_energy(&self) -> Option<PyEnergyType> {
        self.state.energy_zone[self.state.current_player]
            .next
            .map(|e| e.into())
    }

    /// The current-slot energy for `player` (0 or 1).
    fn energy_zone_current(&self, player: usize) -> PyResult<Option<PyEnergyType>> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.energy_zone[player].current.map(|e| e.into()))
    }

    /// The next-slot energy for `player` (0 or 1).
    fn energy_zone_next(&self, player: usize) -> PyResult<Option<PyEnergyType>> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.energy_zone[player].next.map(|e| e.into()))
    }

    #[getter]
    fn has_played_support(&self) -> bool {
        self.state.has_played_support
    }

    #[getter]
    fn has_retreated(&self) -> bool {
        self.state.has_retreated
    }

    fn is_game_over(&self) -> bool {
        self.state.is_game_over()
    }

    /// Get the hand of a specific player (0 or 1)
    fn get_hand(&self, player: usize) -> PyResult<Vec<PyCard>> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.hands[player]
            .iter()
            .map(|card| card.clone().into())
            .collect())
    }

    /// Get the number of cards remaining in a player's deck
    fn get_deck_size(&self, player: usize) -> PyResult<usize> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.decks[player].cards.len())
    }

    /// Get the discard pile of a specific player
    fn get_discard_pile(&self, player: usize) -> PyResult<Vec<PyCard>> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.discard_piles[player]
            .iter()
            .map(|card| card.clone().into())
            .collect())
    }

    /// Get all in-play pokemon for a specific player
    fn get_in_play_pokemon(&self, player: usize) -> PyResult<Vec<Option<PyPlayedCard>>> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.in_play_pokemon[player]
            .iter()
            .map(|opt_card| opt_card.as_ref().map(|card| card.clone().into()))
            .collect())
    }

    /// Get the active pokemon for a specific player (index 0)
    fn get_active_pokemon(&self, player: usize) -> PyResult<Option<PyPlayedCard>> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.in_play_pokemon[player][0]
            .as_ref()
            .map(|card| card.clone().into()))
    }

    /// Get the bench pokemon for a specific player (indices 1-3)
    fn get_bench_pokemon(&self, player: usize) -> PyResult<Vec<Option<PyPlayedCard>>> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.in_play_pokemon[player][1..4]
            .iter()
            .map(|opt_card| opt_card.as_ref().map(|card| card.clone().into()))
            .collect())
    }

    /// Get a specific pokemon by player and position
    fn get_pokemon_at_position(
        &self,
        player: usize,
        position: usize,
    ) -> PyResult<Option<PyPlayedCard>> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        if position > 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Position index must be 0-3 (0=active, 1-3=bench)",
            ));
        }
        Ok(self.state.in_play_pokemon[player][position]
            .as_ref()
            .map(|card| card.clone().into()))
    }

    /// Get the remaining HP of a pokemon at a specific position
    fn get_remaining_hp(&self, player: usize, position: usize) -> PyResult<u32> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        if position > 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Position index must be 0-3 (0=active, 1-3=bench)",
            ));
        }
        if let Some(pokemon) = &self.state.in_play_pokemon[player][position] {
            Ok(pokemon.get_remaining_hp())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No pokemon at this position",
            ))
        }
    }

    /// Count the number of in-play pokemon of a specific energy type for a player
    fn count_in_play_of_type(&self, player: usize, energy_type: PyEnergyType) -> PyResult<usize> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self
            .state
            .num_in_play_of_type(player, energy_type.energy_type))
    }

    /// Get a list of (position, pokemon) tuples for all in-play pokemon of a player
    fn enumerate_in_play_pokemon(&self, player: usize) -> PyResult<Vec<(usize, PyPlayedCard)>> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self
            .state
            .enumerate_in_play_pokemon(player)
            .map(|(pos, card)| (pos, card.clone().into()))
            .collect())
    }

    /// Get a list of (position, pokemon) tuples for all bench pokemon of a player
    fn enumerate_bench_pokemon(&self, player: usize) -> PyResult<Vec<(usize, PyPlayedCard)>> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self
            .state
            .enumerate_bench_pokemon(player)
            .map(|(pos, card)| (pos, card.clone().into()))
            .collect())
    }

    /// Get the debug string representation of the state
    fn debug_string(&self) -> String {
        self.state.debug_string()
    }

    /// Get hand size for a player
    fn get_hand_size(&self, player: usize) -> PyResult<usize> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.hands[player].len())
    }

    /// Get discard pile size for a player
    fn get_discard_pile_size(&self, player: usize) -> PyResult<usize> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.discard_piles[player].len())
    }

    /// Count the number of pokemon in play for a player
    fn count_in_play_pokemon(&self, player: usize) -> PyResult<usize> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.in_play_pokemon[player]
            .iter()
            .filter(|pokemon| pokemon.is_some())
            .count())
    }

    /// Check if a player has an active pokemon
    fn has_active_pokemon(&self, player: usize) -> PyResult<bool> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.in_play_pokemon[player][0].is_some())
    }

    /// Count the number of bench pokemon for a player
    fn count_bench_pokemon(&self, player: usize) -> PyResult<usize> {
        if player > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Player index must be 0 or 1",
            ));
        }
        Ok(self.state.in_play_pokemon[player][1..4]
            .iter()
            .filter(|pokemon| pokemon.is_some())
            .count())
    }

    fn __repr__(&self) -> String {
        format!(
            "PyState(turn={}, player={}, points={:?}, game_over={})",
            self.state.turn_count,
            self.state.current_player,
            self.state.points,
            self.state.is_game_over()
        )
    }
}

/// Python wrapper for Game
#[pyclass(unsendable)]
pub struct PyGame {
    game: Game<'static>,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (deck_a_path, deck_b_path, players=None, seed=None))]
    pub fn new(
        deck_a_path: &str,
        deck_b_path: &str,
        players: Option<Vec<String>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let deck_a = Deck::from_file(deck_a_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load deck A: {}", e))
        })?;
        let deck_b = Deck::from_file(deck_b_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load deck B: {}", e))
        })?;

        let player_codes = if let Some(player_strs) = players {
            let mut codes = Vec::new();
            for player_str in player_strs {
                let code = parse_player_code(&player_str)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                codes.push(code);
            }
            Some(codes)
        } else {
            None
        };

        let cli_players = fill_code_array(player_codes);
        let rust_players = create_players(deck_a, deck_b, cli_players);
        let game_seed = seed.unwrap_or_else(rand::random::<u64>);
        let game = Game::new(rust_players, game_seed);

        Ok(PyGame { game })
    }

    fn play(&mut self) -> Option<PyGameOutcome> {
        self.game.play().map(|outcome| outcome.into())
    }

    fn get_state(&self) -> PyState {
        PyState {
            state: self.game.get_state_clone(),
        }
    }

    fn play_tick(&mut self) -> String {
        let action = self.game.play_tick();
        format!("{:?}", action.action)
    }

    fn __repr__(&self) -> String {
        let state = self.game.get_state_clone();
        format!(
            "PyGame(turn={}, current_player={}, game_over={})",
            state.turn_count,
            state.current_player,
            state.is_game_over()
        )
    }
}

/// Simulation results
#[pyclass]
pub struct PySimulationResults {
    #[pyo3(get)]
    pub total_games: u32,
    #[pyo3(get)]
    pub player_a_wins: u32,
    #[pyo3(get)]
    pub player_b_wins: u32,
    #[pyo3(get)]
    pub ties: u32,
    #[pyo3(get)]
    pub player_a_win_rate: f32,
    #[pyo3(get)]
    pub player_b_win_rate: f32,
    #[pyo3(get)]
    pub tie_rate: f32,
}

#[pymethods]
impl PySimulationResults {
    fn __repr__(&self) -> String {
        format!(
            "SimulationResults(games={}, A_wins={} ({:.1}%), B_wins={} ({:.1}%), ties={} ({:.1}%))",
            self.total_games,
            self.player_a_wins,
            self.player_a_win_rate * 100.0,
            self.player_b_wins,
            self.player_b_win_rate * 100.0,
            self.ties,
            self.tie_rate * 100.0
        )
    }
}

/// Run multiple game simulations
#[pyfunction]
#[pyo3(signature = (deck_a_path, deck_b_path, players=None, num_simulations=100, seed=None))]
pub fn py_simulate(
    deck_a_path: &str,
    deck_b_path: &str,
    players: Option<Vec<String>>,
    num_simulations: u32,
    seed: Option<u64>,
) -> PyResult<PySimulationResults> {
    let deck_a = Deck::from_file(deck_a_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load deck A: {}", e))
    })?;
    let deck_b = Deck::from_file(deck_b_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load deck B: {}", e))
    })?;

    let player_codes = if let Some(player_strs) = players {
        let mut codes = Vec::new();
        for player_str in player_strs {
            let code = parse_player_code(&player_str)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            codes.push(code);
        }
        Some(codes)
    } else {
        None
    };

    let cli_players = fill_code_array(player_codes);

    // Run simulations
    let mut wins_per_deck = [0u32, 0u32, 0u32]; // [player_a, player_b, ties]

    for _ in 0..num_simulations {
        let players = create_players(deck_a.clone(), deck_b.clone(), cli_players.clone());
        let game_seed = seed.unwrap_or_else(rand::random::<u64>);
        let mut game = Game::new(players, game_seed);
        let outcome = game.play();

        match outcome {
            Some(GameOutcome::Win(winner)) => {
                if winner < 2 {
                    wins_per_deck[winner] += 1;
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid winner index: {}",
                        winner
                    )));
                }
            }
            Some(GameOutcome::Tie) | None => {
                wins_per_deck[2] += 1;
            }
        }
    }

    Ok(PySimulationResults {
        total_games: num_simulations,
        player_a_wins: wins_per_deck[0],
        player_b_wins: wins_per_deck[1],
        ties: wins_per_deck[2],
        player_a_win_rate: wins_per_deck[0] as f32 / num_simulations as f32,
        player_b_win_rate: wins_per_deck[1] as f32 / num_simulations as f32,
        tie_rate: wins_per_deck[2] as f32 / num_simulations as f32,
    })
}

/// Vectorized RL environment over `RlEnvCore`.
///
/// With a bot opponent code, the agent always sits in seat 0 and the bot plays
/// seat 1 inside `step`. With `opponent="self"` (self-play mode) the envs
/// pause at *both* seats' decisions: `seats()` reports who must act per env,
/// observations are written from the pending actor's perspective, and
/// `step_some` advances a chosen subset of envs. Rewards/outcomes are always
/// from seat 0's perspective. Episodes auto-reset, so every returned
/// observation belongs to a live game at a decision point. Arrays are
/// returned flat; reshape Python-side to `(num_envs, obs_dim)` and
/// `(num_envs, max_actions, action_feat_dim)`.
#[pyclass(unsendable)]
pub struct PyRlVecEnv {
    envs: Vec<RlEnvCore>,
}

type StepArrays<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<u32>>,
);

impl PyRlVecEnv {
    fn arrays<'py>(&self, py: Python<'py>) -> StepArrays<'py> {
        let obs_dim = self.envs[0].obs_dim();
        let feat_dim = self.envs[0].action_feat_dim();
        let n = self.envs.len();
        let mut obs = vec![0.0f32; n * obs_dim];
        let mut oracle_obs = vec![0.0f32; n * obs_dim];
        let mut feats = vec![0.0f32; n * MAX_ACTIONS * feat_dim];
        let mut n_actions = vec![0u32; n];
        for (i, env) in self.envs.iter().enumerate() {
            let actor = env.pending_actor();
            env.write_observation(&mut obs[i * obs_dim..(i + 1) * obs_dim], actor, false);
            env.write_observation(&mut oracle_obs[i * obs_dim..(i + 1) * obs_dim], actor, true);
            env.write_action_features(
                &mut feats[i * MAX_ACTIONS * feat_dim..(i + 1) * MAX_ACTIONS * feat_dim],
            );
            n_actions[i] = env.num_actions() as u32;
        }
        (
            obs.into_pyarray_bound(py),
            oracle_obs.into_pyarray_bound(py),
            feats.into_pyarray_bound(py),
            n_actions.into_pyarray_bound(py),
        )
    }
}

#[pymethods]
impl PyRlVecEnv {
    #[new]
    #[pyo3(signature = (deck_a_path, deck_b_path, opponent="r", num_envs=1, seed=0, shaping_coef=0.0))]
    pub fn new(
        deck_a_path: &str,
        deck_b_path: &str,
        opponent: &str,
        num_envs: usize,
        seed: u64,
        shaping_coef: f32,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_envs must be >= 1",
            ));
        }
        let code = if opponent == "self" {
            None
        } else {
            let code = parse_player_code(opponent)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            if code == PlayerCode::H {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Human opponent is not supported in the RL environment",
                ));
            }
            Some(code)
        };
        let deck_a = Deck::from_file(deck_a_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load deck A: {}", e))
        })?;
        let deck_b = Deck::from_file(deck_b_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load deck B: {}", e))
        })?;

        let envs = (0..num_envs)
            .map(|i| {
                let opponent_player = code.clone().map(|code| {
                    let players =
                        create_players(deck_a.clone(), deck_b.clone(), vec![code.clone(), code]);
                    players.into_iter().nth(1).unwrap()
                });
                RlEnvCore::new(
                    deck_a.clone(),
                    deck_b.clone(),
                    opponent_player,
                    seed.wrapping_add(i as u64),
                    shaping_coef,
                )
            })
            .collect();
        Ok(PyRlVecEnv { envs })
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn obs_dim(&self) -> usize {
        self.envs[0].obs_dim()
    }

    fn action_feat_dim(&self) -> usize {
        self.envs[0].action_feat_dim()
    }

    fn max_actions(&self) -> usize {
        MAX_ACTIONS
    }

    /// Reset all environments.
    /// Returns (obs, oracle_obs, action_feats, n_actions), flat.
    /// `obs` hides the opponent's hand/deck composition (what a real player
    /// sees); `oracle_obs` is the full state for a training-time critic.
    fn reset<'py>(&mut self, py: Python<'py>) -> StepArrays<'py> {
        for env in &mut self.envs {
            env.reset();
        }
        self.arrays(py)
    }

    /// Step every environment with the chosen legal-action indices.
    ///
    /// Returns (obs, oracle_obs, action_feats, n_actions, rewards, dones,
    /// outcomes). `outcomes[i]` is +1/-1/0 from the agent's perspective and
    /// only meaningful where `dones[i]` is true. Done envs auto-reset; their
    /// returned observation is the first decision of the next episode.
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<i8>>,
    )> {
        if actions.len() != self.envs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "expected {} actions, got {}",
                self.envs.len(),
                actions.len()
            )));
        }
        for (env, &action) in self.envs.iter().zip(actions.iter()) {
            if action >= env.num_actions() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "action index {} out of range ({} legal actions)",
                    action,
                    env.num_actions()
                )));
            }
        }
        let n = self.envs.len();
        let mut rewards = vec![0.0f32; n];
        let mut dones = vec![false; n];
        let mut outcomes = vec![0i8; n];
        let results = self
            .envs
            .par_iter_mut()
            .zip(actions)
            .map(|(env, action)| env.step(action))
            .collect::<Vec<_>>();
        for (i, result) in results.into_iter().enumerate() {
            rewards[i] = result.reward;
            dones[i] = result.done;
            outcomes[i] = result.outcome;
        }
        let (obs, oracle_obs, feats, n_actions) = self.arrays(py);
        Ok((
            obs,
            oracle_obs,
            feats,
            n_actions,
            rewards.into_pyarray_bound(py),
            dones.into_pyarray_bound(py),
            outcomes.into_pyarray_bound(py),
        ))
    }

    /// Seat that must act next in each env (always 0 with a bot opponent).
    fn seats<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.envs
            .iter()
            .map(|env| env.pending_actor() as u32)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py)
    }

    /// Step only the envs in `indices` (self-play mode). Returns
    /// (rewards, dones, outcomes) aligned with `indices`, seat-0 perspective.
    /// Call `reset`/`arrays` semantics as usual: fresh observations for all
    /// envs come from the next `observe` call.
    #[allow(clippy::type_complexity)]
    fn step_some<'py>(
        &mut self,
        py: Python<'py>,
        indices: Vec<usize>,
        actions: Vec<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<i8>>,
    )> {
        if indices.len() != actions.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "indices and actions must have the same length",
            ));
        }
        let mut rewards = vec![0.0f32; indices.len()];
        let mut dones = vec![false; indices.len()];
        let mut outcomes = vec![0i8; indices.len()];
        for (k, (&i, &action)) in indices.iter().zip(actions.iter()).enumerate() {
            let env = self.envs.get_mut(i).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>("env index out of range")
            })?;
            if action >= env.num_actions() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "action index {} out of range ({} legal actions)",
                    action,
                    env.num_actions()
                )));
            }
            let result = env.step(action);
            rewards[k] = result.reward;
            dones[k] = result.done;
            outcomes[k] = result.outcome;
        }
        Ok((
            rewards.into_pyarray_bound(py),
            dones.into_pyarray_bound(py),
            outcomes.into_pyarray_bound(py),
        ))
    }

    /// Current observations/features without stepping (pending-actor
    /// perspective). Same tuple as `reset`.
    fn observe<'py>(&self, py: Python<'py>) -> StepArrays<'py> {
        self.arrays(py)
    }

    /// Human-readable legal actions of one environment (for debugging).
    fn legal_action_strings(&self, env_idx: usize) -> PyResult<Vec<String>> {
        self.envs
            .get(env_idx)
            .map(|env| env.legal_action_strings())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("env_idx out of range"))
    }
}

/// Get available player types
#[pyfunction]
pub fn get_player_types() -> HashMap<String, String> {
    let mut types = HashMap::new();
    types.insert("r".to_string(), "Random Player".to_string());
    types.insert("aa".to_string(), "Attach-Attack Player".to_string());
    types.insert("et".to_string(), "End Turn Player".to_string());
    types.insert("h".to_string(), "Human Player".to_string());
    types.insert("w".to_string(), "Weighted Random Player".to_string());
    types.insert("m".to_string(), "MCTS Player".to_string());
    types.insert("v".to_string(), "Value Function Player".to_string());
    types.insert("e".to_string(), "Expectiminimax Player".to_string());
    types
}

/// Python module definition
pub fn deckgym(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnergyType>()?;
    m.add_class::<PyAttack>()?;
    m.add_class::<PyAbility>()?;
    m.add_class::<PyCard>()?;
    m.add_class::<PyPlayedCard>()?;
    m.add_class::<PyDeck>()?;
    m.add_class::<PyGame>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyGameOutcome>()?;
    m.add_class::<PySimulationResults>()?;
    m.add_class::<PyRlVecEnv>()?;
    m.add_function(wrap_pyfunction!(py_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(get_player_types, m)?)?;
    Ok(())
}
