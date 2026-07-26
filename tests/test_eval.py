import shutil

import chess
import pytest
import yaml

from rlchess.eval import evaluate_pgn_file, evaluate_run, get_stockfish_path, is_stockfish_available
from rlchess.logging import RunLogger
from rlchess.ppo import GameRecord

STOCKFISH_PATH = shutil.which("stockfish")


@pytest.fixture
def config(tmp_path):
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    config["logging"]["run_dir"] = str(tmp_path / "runs")
    config["logging"]["run_id"] = "eval-test-run"
    config["eval"]["rolling_window"] = 2
    return config


def test_get_stockfish_path_env_var_takes_precedence(config, monkeypatch):
    config["eval"]["stockfish_path"] = "/from/config"
    monkeypatch.setenv("STOCKFISH_PATH", "/from/env")
    assert get_stockfish_path(config) == "/from/env"


def test_get_stockfish_path_falls_back_to_config(config, monkeypatch):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    config["eval"]["stockfish_path"] = "/from/config"
    assert get_stockfish_path(config) == "/from/config"


def test_is_stockfish_available_false_when_missing(config, monkeypatch):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    config["eval"]["stockfish_path"] = "/definitely/not/a/real/binary"
    assert is_stockfish_available(config) is False


def test_evaluate_run_noop_without_stockfish(config, monkeypatch, tmp_path):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    config["eval"]["stockfish_path"] = None

    logger = RunLogger(config)
    logger.save_games(
        0, [GameRecord(trajectories={}, result="1-0", moves_san=["e4", "e5"])]
    )

    result = evaluate_run(logger.run_dir, config)

    assert result is None
    assert not (logger.run_dir / "accuracy.jsonl").exists()


@pytest.mark.skipif(STOCKFISH_PATH is None, reason="stockfish binary not installed")
def test_evaluate_pgn_file_real_stockfish(config, monkeypatch):
    monkeypatch.setenv("STOCKFISH_PATH", STOCKFISH_PATH)
    logger = RunLogger(config)
    game = GameRecord(
        trajectories={}, result="1-0", moves_san=["e4", "e5", "Nf3", "Nc6", "Bb5"]
    )
    game_dir = logger.save_games(0, [game])

    import random

    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    try:
        mean_loss = evaluate_pgn_file(
            engine, next(game_dir.glob("*.pgn")), sample_rate=1.0, depth=6,
            cp_clip=1000, rng=random.Random(0),
        )
    finally:
        engine.quit()

    assert mean_loss is not None
    assert 0.0 <= mean_loss <= 1000.0


@pytest.mark.skipif(STOCKFISH_PATH is None, reason="stockfish binary not installed")
def test_evaluate_run_produces_rolling_accuracy(config, monkeypatch):
    monkeypatch.setenv("STOCKFISH_PATH", STOCKFISH_PATH)
    config["eval"]["sample_rate"] = 1.0
    config["eval"]["depth"] = 6

    logger = RunLogger(config)
    games = [
        GameRecord(trajectories={}, result="1-0", moves_san=["e4", "e5", "Nf3"]),
        GameRecord(trajectories={}, result="0-1", moves_san=["d4", "d5", "Nf3"]),
    ]
    logger.save_games(0, games)
    logger.save_games(1, games)

    accuracy_path = evaluate_run(logger.run_dir, config)

    assert accuracy_path is not None
    lines = accuracy_path.read_text().strip().splitlines()
    assert len(lines) == 2
    import json

    for line in lines:
        record = json.loads(line)
        assert record["step"] in (0, 1)
        assert record["rolling_accuracy_cp"] >= 0.0
