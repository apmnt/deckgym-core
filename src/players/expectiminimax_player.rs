use log::{trace, LevelFilter};
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::fmt::Debug;
use std::fmt::Write;
use std::time::{Duration, Instant};
use std::vec;

use crate::actions::{forecast_action, Action};
use crate::{Deck, State};

use super::Player;

// Type alias for value functions
// Takes a state and player index, returns a score
// Using Box<dyn Fn> to allow closures with captured variables
pub type ValueFunction = Box<dyn Fn(&State, usize) -> f64 + Send>;

lazy_static::lazy_static! {
    static ref EXPECTI_PROFILE_ENABLED: bool =
        std::env::var_os("DECKGYM_PROFILE_EXPECTIMINIMAX").is_some();
    static ref EXPECTI_PROFILE_EVERY: u64 = std::env::var("DECKGYM_PROFILE_EXPECTIMINIMAX_EVERY")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1000);
}

thread_local! {
    static EXPECTI_PROFILE: RefCell<ExpectiMiniMaxProfile> =
        RefCell::new(ExpectiMiniMaxProfile::default());
}

#[derive(Default)]
struct ExpectiMiniMaxProfile {
    decisions: u64,
    root_actions: u64,
    action_evals: u64,
    state_nodes: u64,
    terminal_evals: u64,
    chance_branches: u64,
    generated_actions: u64,
    decision_ns: u128,
    forecast_ns: u128,
    branch_apply_ns: u128,
    generate_actions_ns: u128,
    value_ns: u128,
}

fn profile_enabled() -> bool {
    *EXPECTI_PROFILE_ENABLED
}

fn profile_record(update: impl FnOnce(&mut ExpectiMiniMaxProfile)) {
    if !profile_enabled() {
        return;
    }
    EXPECTI_PROFILE.with(|profile| update(&mut profile.borrow_mut()));
}

fn profile_duration(duration: Duration) -> u128 {
    duration.as_nanos()
}

fn maybe_report_profile() {
    if !profile_enabled() {
        return;
    }
    EXPECTI_PROFILE.with(|profile| {
        let profile = profile.borrow();
        if profile.decisions == 0 || profile.decisions % *EXPECTI_PROFILE_EVERY != 0 {
            return;
        }
        let total_ms = profile.decision_ns as f64 / 1_000_000.0;
        let pct = |ns: u128| {
            if profile.decision_ns == 0 {
                0.0
            } else {
                100.0 * ns as f64 / profile.decision_ns as f64
            }
        };
        eprintln!(
            concat!(
                "[expectiminimax-profile] decisions={} total_ms={:.1} ",
                "avg_decision_ms={:.3} root_actions/decision={:.1} ",
                "action_evals={} state_nodes={} terminal_evals={} ",
                "chance_branches={} generated_actions={} ",
                "forecast_ms={:.1}({:.1}%) branch_apply_ms={:.1}({:.1}%) ",
                "generate_actions_ms={:.1}({:.1}%) value_ms={:.1}({:.1}%)"
            ),
            profile.decisions,
            total_ms,
            total_ms / profile.decisions as f64,
            profile.root_actions as f64 / profile.decisions as f64,
            profile.action_evals,
            profile.state_nodes,
            profile.terminal_evals,
            profile.chance_branches,
            profile.generated_actions,
            profile.forecast_ns as f64 / 1_000_000.0,
            pct(profile.forecast_ns),
            profile.branch_apply_ns as f64 / 1_000_000.0,
            pct(profile.branch_apply_ns),
            profile.generate_actions_ns as f64 / 1_000_000.0,
            pct(profile.generate_actions_ns),
            profile.value_ns as f64 / 1_000_000.0,
            pct(profile.value_ns),
        );
    });
}

struct DebugStateNode {
    acting_player: usize,
    children: Vec<DebugActionNode>,
    proba: f64,
    value: f64,
}

struct DebugActionNode {
    action: Action,
    children: Vec<DebugStateNode>,
    value: f64,
}

pub struct ExpectiMiniMaxPlayer {
    pub deck: Deck,
    pub max_depth: usize, // max_depth = 1 it should be value function player
    pub write_debug_trees: bool,
    pub value_function: ValueFunction,
}

impl Player for ExpectiMiniMaxPlayer {
    fn decision_fn(
        &mut self,
        rng: &mut StdRng,
        state: &State,
        possible_actions: &[Action],
    ) -> Action {
        let decision_start = profile_enabled().then(Instant::now);
        let myself = possible_actions[0].actor;

        // Get value for each possible action
        let original_level = log::max_level();
        log::set_max_level(LevelFilter::Info); // Temporarily silence debug and trace logs
        let mut scores: Vec<f64> = Vec::with_capacity(possible_actions.len());

        let maybe_root = if self.write_debug_trees {
            let mut root = DebugStateNode {
                acting_player: myself,
                children: vec![],
                proba: 1.0,
                value: 0.0,
            };
            for action in possible_actions.iter() {
                let (score, action_node) = expected_value_function(
                    rng,
                    state,
                    action,
                    self.max_depth - 1,
                    myself,
                    &self.value_function,
                );
                scores.push(score);
                root.children.push(action_node);
            }
            Some(root)
        } else {
            for action in possible_actions.iter() {
                scores.push(expected_value_score(
                    rng,
                    state,
                    action,
                    self.max_depth - 1,
                    myself,
                    &self.value_function,
                ));
            }
            None
        };
        log::set_max_level(original_level); // Restore the original logging level

        trace!("Scores: {scores:?}");
        // Select the one with best score
        let (best_idx, best_score) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, score)| (idx, *score))
            .unwrap();

        // Output Tree in Dot format for visualization if enabled
        if let Some(mut root) = maybe_root {
            root.value = best_score;
            let folder = "expectiminimax_trees";
            std::fs::create_dir_all(folder).unwrap();

            // Find next available filename to avoid overwriting
            let mut counter = 0;
            let filename = loop {
                let candidate = format!(
                    "{}/expectiminimax_tree_turn{}_p{}_{}.dot",
                    folder, state.turn_count, myself, counter
                );
                if !std::path::Path::new(&candidate).exists() {
                    break candidate;
                }
                counter += 1;
            };
            save_tree_as_dot(&root, state, filename).unwrap();
        }

        if let Some(start) = decision_start {
            profile_record(|profile| {
                profile.decisions += 1;
                profile.root_actions += possible_actions.len() as u64;
                profile.decision_ns += profile_duration(start.elapsed());
            });
            maybe_report_profile();
        }

        // You can now use both best_idx and best_score as needed
        possible_actions[best_idx].clone()
    }

    fn get_deck(&self) -> Deck {
        self.deck.clone()
    }
}

fn expected_value_score(
    rng: &mut StdRng,
    state: &State,
    action: &Action,
    depth: usize,
    myself: usize,
    value_function: &ValueFunction,
) -> f64 {
    let forecast_start = profile_enabled().then(Instant::now);
    let (probabilities, mutations) = forecast_action(state, action).into_branches();
    if let Some(start) = forecast_start {
        profile_record(|profile| {
            profile.action_evals += 1;
            profile.chance_branches += probabilities.len() as u64;
            profile.forecast_ns += profile_duration(start.elapsed());
        });
    }

    probabilities
        .iter()
        .zip(mutations)
        .map(|(prob, mutation)| {
            let apply_start = profile_enabled().then(Instant::now);
            let mut state_copy = state.clone();
            mutation(rng, &mut state_copy, action);
            if let Some(start) = apply_start {
                profile_record(|profile| {
                    profile.branch_apply_ns += profile_duration(start.elapsed());
                });
            }
            prob * expectiminimax_score(rng, &state_copy, depth, myself, value_function)
        })
        .sum()
}

fn expectiminimax_score(
    rng: &mut StdRng,
    state: &State,
    depth: usize,
    myself: usize,
    value_function: &ValueFunction,
) -> f64 {
    profile_record(|profile| {
        profile.state_nodes += 1;
    });

    if state.is_game_over() || depth == 0 || state.current_player != myself {
        let value_start = profile_enabled().then(Instant::now);
        let score = value_function(state, myself);
        if let Some(start) = value_start {
            profile_record(|profile| {
                profile.terminal_evals += 1;
                profile.value_ns += profile_duration(start.elapsed());
            });
        }
        return score;
    }

    let generate_start = profile_enabled().then(Instant::now);
    let (actor, actions) = state.generate_possible_actions();
    if let Some(start) = generate_start {
        profile_record(|profile| {
            profile.generated_actions += actions.len() as u64;
            profile.generate_actions_ns += profile_duration(start.elapsed());
        });
    }

    if actor == myself {
        actions
            .iter()
            .map(|action| {
                expected_value_score(rng, state, action, depth - 1, myself, value_function)
            })
            .fold(f64::NEG_INFINITY, f64::max)
    } else {
        actions
            .iter()
            .map(|action| {
                expected_value_score(rng, state, action, depth - 1, myself, value_function)
            })
            .fold(f64::INFINITY, f64::min)
    }
}

fn expected_value_function(
    rng: &mut StdRng,
    state: &State,
    action: &Action,
    depth: usize,
    myself: usize,
    value_function: &ValueFunction,
) -> (f64, DebugActionNode) {
    let indent = "\t".repeat(10 - depth.min(10));
    trace!("{indent}E({myself}) depth left: {depth} action: {action:?}");

    let (probabilities, mutations) = forecast_action(state, action).into_branches();
    let mut outcomes: Vec<State> = vec![];
    for mutation in mutations {
        let mut state_copy = state.clone();
        mutation(rng, &mut state_copy, action);
        outcomes.push(state_copy);
    }

    // Mantain node
    let mut scores = vec![];
    let mut action_node = DebugActionNode {
        action: action.clone(),
        children: vec![],
        value: 0.0,
    };
    for (prob, outcome) in probabilities.iter().zip(outcomes.iter()) {
        let (score, mut state_node) = expectiminimax(rng, outcome, depth, myself, value_function);
        scores.push(score);
        state_node.proba = *prob;
        action_node.children.push(state_node);
    }

    let score = probabilities
        .iter()
        .zip(scores.iter())
        .map(|(p, s)| p * s)
        .sum::<f64>();

    action_node.value = score;
    trace!("{indent}E({myself}) action: {action:?} score: {score}");
    (score, action_node)
}

fn expectiminimax(
    rng: &mut StdRng,
    state: &State,
    depth: usize,
    myself: usize,
    value_function: &ValueFunction,
) -> (f64, DebugStateNode) {
    if state.is_game_over() || depth == 0 || state.current_player != myself {
        let score = value_function(state, myself);
        let state_node = DebugStateNode {
            acting_player: state.current_player,
            children: vec![],
            proba: 1.0,
            value: score,
        };
        return (score, state_node);
    }

    let (actor, actions) = state.generate_possible_actions();
    if actor == myself {
        // We are in maximing mode.
        let mut scores: Vec<f64> = Vec::with_capacity(actions.len());
        let mut children = vec![];
        for action in actions.iter() {
            let (score, action_node) =
                expected_value_function(rng, state, action, depth - 1, myself, value_function);
            scores.push(score);
            children.push(action_node);
        }
        let best_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let state_node = DebugStateNode {
            acting_player: actor,
            children,
            proba: 0.0, // this will get set by parent
            value: best_score,
        };
        (best_score, state_node)
    } else {
        // TODO: If minimizing, we can't just generate_possible_actions since
        //  not everything is public information. So we would have to have
        //  our own version of it that only returns the actions that are
        let mut scores: Vec<f64> = Vec::with_capacity(actions.len());
        let mut children: Vec<DebugActionNode> = Vec::new();
        for action in actions.iter() {
            let (score, action_node) =
                expected_value_function(rng, state, action, depth - 1, myself, value_function);
            scores.push(score);
            children.push(action_node);
        }
        let best_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let state_node = DebugStateNode {
            acting_player: actor,
            children,
            proba: 0.0, // this will get set by parent
            value: best_score,
        };
        (best_score, state_node)
    }
}

impl Debug for ExpectiMiniMaxPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExpectiMiniMaxPlayer")
    }
}

fn save_tree_as_dot(
    root: &DebugStateNode,
    root_state: &State,
    filename: String,
) -> std::io::Result<()> {
    let dot_representation = generate_dot(root, root_state);
    std::fs::write(filename, dot_representation)
}

fn generate_dot(root: &DebugStateNode, root_state: &State) -> String {
    let mut dot = String::new();
    writeln!(dot, "digraph GameTree {{").unwrap();
    writeln!(dot, "    rankdir=TB;").unwrap();
    writeln!(dot, "    node [shape=box];").unwrap();

    // Add info node with root state debug string
    let debug_str = root_state
        .debug_string()
        .replace('"', "'")
        .replace('\n', "\\l")
        + "\\l";
    writeln!(
        dot,
        "    info [label=\"{}\", shape=box, style=filled, fillcolor=lightyellow, align=left];",
        debug_str
    )
    .unwrap();

    let mut state_counter = 0;
    let mut action_counter = 0;

    generate_dot_recursive(
        root,
        &mut dot,
        &mut state_counter,
        &mut action_counter,
        0,
        root.acting_player,
    );

    writeln!(dot, "}}").unwrap();
    dot
}

fn generate_dot_recursive(
    state: &DebugStateNode,
    dot: &mut String,
    state_counter: &mut usize,
    action_counter: &mut usize,
    current_state_id: usize,
    myself: usize,
) {
    // Define the state node with color based on acting player
    let color = if state.acting_player == myself {
        "lightgreen"
    } else {
        "lightcoral"
    };
    writeln!(
        dot,
        "    s{} [label=\"State\\nPlayer: {}\\nProba: {:.3}\\nValue: {:.3}\", style=filled, fillcolor={}];",
        current_state_id,
        state.acting_player,
        state.proba,
        state.value,
        color
    ).unwrap();

    // Process each action child
    for action_node in &state.children {
        *action_counter += 1;
        let action_id = *action_counter;

        // Define the action node (neutral color)
        writeln!(
            dot,
            "    a{} [label=\"P{} {}\\n{:?}\\nValue: {:.3}\", shape=ellipse, style=filled, fillcolor=lightgrey];",
            action_id,
            action_node.action.actor,
            action_node.action.is_stack,
            action_node.action.action,
            action_node.value
        ).unwrap();

        // Edge from state to action
        writeln!(dot, "    s{} -> a{};", current_state_id, action_id).unwrap();

        // Process each state child of this action
        for child_state in &action_node.children {
            *state_counter += 1;
            let child_state_id = *state_counter;

            // Edge from action to child state
            writeln!(dot, "    a{} -> s{};", action_id, child_state_id).unwrap();

            // Recursively process the child state
            generate_dot_recursive(
                child_state,
                dot,
                state_counter,
                action_counter,
                child_state_id,
                myself,
            );
        }
    }
}
