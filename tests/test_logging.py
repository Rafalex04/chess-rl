import json

import chess
import chess.pgn
import pytest
import torch
import yaml

from rlchess.logging import RunLogger
from rlchess.policy import ChessPolicyNet
from rlchess.ppo import GameRecord


@pytest.fixture
def config(tmp_path):
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    config["logging"]["run_dir"] = str(tmp_path / "runs")
    config["logging"]["run_id"] = "test-run"
    config["logging"]["rolling_window"] = 3
    config["policy"]["num_filters"] = 8
    config["policy"]["num_res_blocks"] = 1
    config["policy"]["value_hidden_dim"] = 8
    return config


def _game(result: str, moves_san: list[str]) -> GameRecord:
    return GameRecord(trajectories={}, result=result, moves_san=moves_san)


def test_run_directory_structure(config):
    logger = RunLogger(config)
    assert (logger.run_dir / "config.yaml").exists()
    assert (logger.run_dir / "checkpoints").is_dir()
    assert (logger.run_dir / "games").is_dir()

    with open(logger.run_dir / "config.yaml") as f:
        dumped = yaml.safe_load(f)
    assert dumped["policy"]["num_filters"] == 8


def test_metrics_jsonl_rolling_window(config):
    logger = RunLogger(config)

    # rolling_window=3: feed white, white, black, draw across two batches.
    logger.log_batch(0, [_game("1-0", ["e4"]), _game("1-0", ["e4"])], {"loss": 0.5})
    record = logger.log_batch(1, [_game("0-1", ["e4"]), _game("1/2-1/2", ["e4"])], {"loss": 0.4})

    assert record["games_played"] == 4
    assert record["white_wins"] == 2
    assert record["black_wins"] == 1
    assert record["draws"] == 1
    # Rolling window (last 3 outcomes): white, black, draw -> 1/3 each.
    assert record["rolling_white_winrate"] == pytest.approx(1 / 3)
    assert record["rolling_black_winrate"] == pytest.approx(1 / 3)
    assert record["rolling_draw_rate"] == pytest.approx(1 / 3)
    assert record["rolling_accuracy_cp"] is None
    assert record["loss"] == 0.4

    lines = logger.metrics_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # must be valid JSON


def test_checkpoint_round_trip(config):
    logger = RunLogger(config)
    policy = ChessPolicyNet.from_config(config)

    path = logger.save_checkpoint(7, policy)
    assert path.exists()

    loaded_policy = ChessPolicyNet.from_config(config)
    loaded_policy.load_state_dict(torch.load(path))

    for p1, p2 in zip(policy.parameters(), loaded_policy.parameters()):
        assert torch.equal(p1, p2)


def test_pgn_valid_and_reparseable(config):
    logger = RunLogger(config)
    games = [
        _game("1-0", ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]),
        _game("1/2-1/2", ["Nf3", "Nf6", "g3", "g6"]),
    ]

    game_dir = logger.save_games(3, games)
    pgn_files = sorted(game_dir.glob("*.pgn"))
    assert len(pgn_files) == 2

    for pgn_file, expected in zip(pgn_files, games):
        with open(pgn_file) as f:
            parsed = chess.pgn.read_game(f)
        assert parsed.headers["Result"] == expected.result

        board = chess.Board()
        for san in expected.moves_san:
            board.push_san(san)
        expected_fen = board.fen()

        replay_board = parsed.board()
        move_count = 0
        for move in parsed.mainline_moves():
            replay_board.push(move)
            move_count += 1
        assert move_count == len(expected.moves_san)
        assert replay_board.fen() == expected_fen
