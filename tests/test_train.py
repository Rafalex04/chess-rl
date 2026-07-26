import numpy as np
import pytest
import yaml

from rlchess.policy import ChessPolicyNet
from rlchess.train import train


class _StubLogger:
    """Records what train() reports, without touching disk."""

    def __init__(self):
        self.batches = []
        self.checkpoints = []

    def log_batch(self, step, games, stats):
        self.batches.append((step, stats))

    def save_checkpoint(self, step, policy):
        self.checkpoints.append(step)

    def save_games(self, step, games):
        pass


@pytest.fixture
def tiny_config():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    config["env"]["max_moves"] = 20
    config["policy"]["num_filters"] = 8
    config["policy"]["num_res_blocks"] = 1
    config["policy"]["value_hidden_dim"] = 8
    config["ppo"]["games_per_batch"] = 2
    config["ppo"]["minibatch_size"] = 32
    config["ppo"]["epochs"] = 2
    config["train"]["num_updates"] = 2
    config["train"]["device"] = "cpu"
    config["logging"]["checkpoint_interval"] = 1
    config["logging"]["sample_games_per_checkpoint"] = 1
    return config


def test_tiny_train_run_completes_without_logger(tiny_config):
    policy = train(tiny_config)
    assert isinstance(policy, ChessPolicyNet)


def test_tiny_train_run_logs_finite_stats(tiny_config):
    logger = _StubLogger()
    train(tiny_config, logger=logger)

    assert len(logger.batches) == tiny_config["train"]["num_updates"]
    for step, stats in logger.batches:
        for key, value in stats.items():
            assert np.isfinite(value), f"step {step} stat {key} not finite: {value}"
    # checkpoint_interval=1 -> a checkpoint every step
    assert logger.checkpoints == list(range(tiny_config["train"]["num_updates"]))
