"""RL-layer regression tests: checkpoint round-trips, hidden-information
masking, and a CPU trainer smoke run.

Run from the repo root:  .venv/bin/python -m pytest rl/tests -q
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import make_agent  # noqa: E402
from env_wrapper import VecEnv  # noqa: E402
from torch_runtime import agent_from_checkpoint, save_agent_state  # noqa: E402

DECK = "example_decks/venusaur-exeggutor.txt"


def test_checkpoint_roundtrip_preserves_config_and_outputs(tmp_path):
    obs_dim, act_dim = 410, 64
    agent = make_agent(obs_dim, act_dim, arch="res", hidden=128, blocks=1, heads=8,
                       memory=True, belief=True)
    path = tmp_path / "agent.pt"
    save_agent_state(agent, path)
    loaded = agent_from_checkpoint(path, obs_dim, act_dim, map_location="cpu")
    assert loaded.config["heads"] == 8
    assert loaded.config["memory"] is True
    assert loaded.config["belief"] is True

    obs = torch.randn(2, obs_dim)
    feats = torch.randn(2, 5, act_dim)
    mask = torch.ones(2, 5, dtype=torch.bool)
    h = agent.initial_state(2, "cpu")
    logits_a, _ = agent.policy_logits(obs, feats, mask, h)
    logits_b, _ = loaded.policy_logits(obs, feats, mask, h)
    assert torch.allclose(logits_a, logits_b)


def test_legacy_raw_state_dict_still_loads(tmp_path):
    obs_dim, act_dim = 410, 64
    agent = make_agent(obs_dim, act_dim, arch="res", hidden=128, blocks=2, heads=4)
    path = tmp_path / "legacy.pt"
    torch.save(agent.state_dict(), path)  # raw, config-less format
    loaded = agent_from_checkpoint(path, obs_dim, act_dim, map_location="cpu")
    assert type(loaded).__name__ == "ResAttnAgent"


def test_policy_view_hides_opponent_hand_and_deck():
    env = VecEnv(DECK, DECK, opponent="r", num_envs=4, seed=7)
    obs, oracle, _, _ = env.reset()
    assert obs.shape == oracle.shape
    # The policy view must reveal no more than the oracle view, and the
    # hidden sections must actually be hidden in at least one env.
    assert np.all((obs == oracle) | (obs == 0.0))
    assert np.any(obs != oracle)


def test_obs_carries_vocab_ids():
    import deckgym

    env = VecEnv(DECK, DECK, opponent="r", num_envs=2, seed=3)
    obs, oracle, _, _ = env.reset()
    num_cards = deckgym.num_global_cards()
    ids = obs[:, -40:]
    # Ids are valid global indices (or the padding id) and identical in
    # both views: decklists are treated as public, hands/decks are not.
    assert np.all((ids >= 0) & (ids <= num_cards))
    assert np.all(ids == oracle[:, -40:])
    # A mirror match has at most 20 distinct cards: padding must appear.
    assert np.any(ids == num_cards)


def test_general_agent_roundtrip_and_masking(tmp_path):
    import deckgym

    env = VecEnv(DECK, DECK, opponent="r", num_envs=4, seed=11)
    agent = make_agent(env.obs_dim, env.act_feat_dim, arch="gen", hidden=64, blocks=1)
    obs, oracle, feats, mask = env.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    oracle_t = torch.as_tensor(oracle, dtype=torch.float32)
    feats_t = torch.as_tensor(feats, dtype=torch.float32)
    mask_t = torch.as_tensor(mask)
    logits, value = agent(obs_t, oracle_t, feats_t, mask_t)
    assert torch.isfinite(logits[mask_t]).all()
    assert (logits[~mask_t] < -1e30).all()
    assert torch.isfinite(value).all()

    path = tmp_path / "gen.pt"
    save_agent_state(agent, path)
    loaded = agent_from_checkpoint(path, env.obs_dim, env.act_feat_dim, map_location="cpu")
    assert type(loaded).__name__ == "GeneralAgent"
    assert loaded.config["num_cards"] == deckgym.num_global_cards()
    logits_b, _ = loaded(obs_t, oracle_t, feats_t, mask_t)
    assert torch.allclose(logits, logits_b)
    # The attribute table must survive the round-trip (it is a buffer).
    assert torch.equal(agent.cards.attr_table, loaded.cards.attr_table)


def test_multi_deck_env_fixed_dims():
    spec = (
        "example_decks/venusaur-exeggutor.txt,"
        "example_decks/weezing-arbok.txt,"
        "example_decks/mewtwoex.txt"
    )
    env = VecEnv(spec, spec, opponent="r", num_envs=4, seed=5)
    obs, _, feats, mask = env.reset()
    single = VecEnv(DECK, DECK, opponent="r", num_envs=1, seed=5)
    assert env.obs_dim == single.obs_dim
    assert env.act_feat_dim == single.act_feat_dim
    # Step a few hundred decisions across resets: dims stay fixed and at
    # least two different vocabularies (matchups) must show up.
    rng = np.random.default_rng(0)
    signatures = set()
    for _ in range(300):
        signatures.add(tuple(obs[0, -40:].astype(int)))
        actions = [rng.integers(0, mask[i].sum()) for i in range(env.num_envs)]
        obs, _, feats, mask, _, _, _ = env.step(np.array(actions))
    assert len(signatures) >= 2


def test_ppo_smoke_cpu(tmp_path):
    rl_dir = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(rl_dir / "train_ppo.py"),
        "--opponent", "r",
        "--total-steps", "256",
        "--num-envs", "4",
        "--num-steps", "32",
        "--num-minibatches", "2",
        "--arch", "res", "--hidden", "64", "--blocks", "1",
        "--device", "cpu",
        "--run-name", "pytest_smoke",
    ]
    subprocess.run(cmd, check=True, cwd=rl_dir.parent, timeout=600)
    run_dir = rl_dir.parent / "runs" / "pytest_smoke"
    assert (run_dir / "final.pt").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
