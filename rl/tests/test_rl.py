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
