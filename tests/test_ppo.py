import copy

import chess
import numpy as np
import pytest
import torch
import yaml

from rlchess.env import ACTION_SPACE_SIZE, NUM_PLANES
from rlchess.policy import ChessPolicyNet
from rlchess.ppo import (
    ColorTrajectory,
    _propagate_terminal_reward,
    build_ppo_batch,
    collect_self_play_games,
    compute_gae,
    ppo_update,
)


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
    return config


def test_propagate_terminal_reward_losing_side():
    # White's move ended the game (checkmate); env already gave White's last
    # step the correct terminal reward (+1). Black's last step only got its
    # at-the-time shaping reward (0.1) since the game wasn't over yet from
    # Black's perspective when they moved.
    white_traj = ColorTrajectory(rewards=[0.0, 0.1, 1.0])
    black_traj = ColorTrajectory(rewards=[0.0, 0.1])
    trajectories = {chess.WHITE: white_traj, chess.BLACK: black_traj}

    _propagate_terminal_reward(trajectories, last_mover_color=chess.WHITE, result="1-0")

    assert white_traj.rewards == [0.0, 0.1, 1.0]  # untouched (env already got this right)
    assert black_traj.rewards == [0.0, -1.0]  # overwritten: black lost


def test_propagate_terminal_reward_draw():
    white_traj = ColorTrajectory(rewards=[0.0, 0.0])
    black_traj = ColorTrajectory(rewards=[0.0, 0.0])
    trajectories = {chess.WHITE: white_traj, chess.BLACK: black_traj}

    _propagate_terminal_reward(trajectories, last_mover_color=chess.BLACK, result="1/2-1/2")

    assert white_traj.rewards == [0.0, 0.0]  # overwritten with 0.0, no visible change
    assert black_traj.rewards == [0.0, 0.0]  # untouched


def test_compute_gae_matches_hand_computed():
    rewards = [1.0, 0.0]
    values = [0.0, 0.0]
    gamma = 0.5
    gae_lambda = 1.0

    advantages, returns = compute_gae(rewards, values, gamma, gae_lambda)

    # t=1: delta = 0 + 0.5*0 - 0 = 0 -> advantage[1] = 0
    # t=0: delta = 1 + 0.5*0 - 0 = 1 -> advantage[0] = 1 + 0.5*1*0 = 1
    assert advantages == pytest.approx([1.0, 0.0])
    assert returns == pytest.approx([1.0, 0.0])


def test_collect_self_play_games_produces_valid_records(tiny_config):
    policy = ChessPolicyNet.from_config(tiny_config)
    device = torch.device("cpu")

    games = collect_self_play_games(tiny_config, policy, num_games=2, device=device)

    assert len(games) == 2
    for game in games:
        assert game.result in ("1-0", "0-1", "1/2-1/2")
        white_len = len(game.trajectories[chess.WHITE].rewards)
        black_len = len(game.trajectories[chess.BLACK].rewards)
        assert len(game.moves_san) == white_len + black_len

        # Replaying the recorded SAN sequence must succeed end to end.
        board = chess.Board()
        for san in game.moves_san:
            board.push_san(san)


def test_ppo_update_finite_and_changes_weights(tiny_config):
    policy = ChessPolicyNet.from_config(tiny_config)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(policy.parameters(), lr=tiny_config["ppo"]["lr"])

    games = collect_self_play_games(tiny_config, policy, num_games=2, device=device)
    batch = build_ppo_batch(
        games, tiny_config["ppo"]["gamma"], tiny_config["ppo"]["gae_lambda"], device
    )

    params_before = [p.clone() for p in policy.parameters()]
    stats = ppo_update(policy, optimizer, batch, tiny_config)

    for key, value in stats.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"

    changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, policy.parameters())
    )
    assert changed
