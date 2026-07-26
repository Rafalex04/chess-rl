import shutil
import subprocess

import chess
import pytest
import yaml
from PIL import Image

from rlchess.logging import RunLogger
from rlchess.ppo import GameRecord
from rlchess.render_video import (
    render_board_frame,
    render_metric_timelapse_frames,
    render_slow_motion_frames,
    render_video,
    select_slow_motion_checkpoints,
)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@pytest.fixture
def config(tmp_path):
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    config["logging"]["run_dir"] = str(tmp_path / "runs")
    config["logging"]["run_id"] = "video-test-run"
    config["video"]["fps"] = 5
    config["video"]["board_size"] = 80
    config["video"]["metric_timelapse_seconds"] = 1
    config["video"]["slow_motion_seconds_per_move"] = 0.2
    return config


def test_render_board_frame_shape_and_not_blank():
    board = chess.Board()
    img = render_board_frame(board, size=80, caption="step 0")
    assert img.mode == "RGB"
    assert img.width == 80
    # Some non-trivial height added for the caption strip.
    assert img.height > 80
    # Board should not be a single flat color (pieces + checkerboard present).
    colors = img.getcolors(maxcolors=1_000_000)
    assert len(colors) > 2


def test_select_slow_motion_checkpoints_nearest_match():
    available = [0, 12, 25, 37, 50]
    # Fractions chosen to land exactly on available steps, avoiding ties.
    selected = select_slow_motion_checkpoints(available, [0.0, 0.5, 1.0])
    assert selected == [0, 25, 50]


def test_select_slow_motion_checkpoints_empty():
    assert select_slow_motion_checkpoints([], [0.1, 0.5]) == []


def test_render_slow_motion_frames_count(tmp_path):
    logger_config = {
        "logging": {"run_dir": str(tmp_path / "runs"), "run_id": "r", "rolling_window": 50},
        "policy": {"num_filters": 8},
    }
    logger = RunLogger(logger_config)
    game = GameRecord(trajectories={}, result="1-0", moves_san=["e4", "e5", "Nf3"])
    game_dir = logger.save_games(0, [game])
    pgn_path = next(game_dir.glob("*.pgn"))

    fps, seconds_per_move = 5, 0.2
    frames = render_slow_motion_frames(pgn_path, board_size=80, fps=fps, seconds_per_move=seconds_per_move, caption_prefix="test")

    frames_per_move = max(1, round(fps * seconds_per_move))
    # 3 moves + the initial position = 4 board states.
    assert len(frames) == frames_per_move * 4
    assert all(isinstance(f, Image.Image) for f in frames)


def test_render_metric_timelapse_frames_count_and_handles_missing_accuracy():
    metrics = [
        {"step": i, "rolling_white_winrate": 0.4, "rolling_black_winrate": 0.3, "rolling_draw_rate": 0.3}
        for i in range(5)
    ]
    frames = render_metric_timelapse_frames(metrics, accuracy=[], num_frames=6, size=(80, 80))
    assert len(frames) == 6
    assert all(isinstance(f, Image.Image) for f in frames)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
def test_render_video_end_to_end(config):
    logger = RunLogger(config)
    games = [GameRecord(trajectories={}, result="1-0", moves_san=["e4", "e5", "Nf3", "Nc6"])]
    logger.log_batch(0, games, {"loss": 0.5})
    logger.save_games(0, games)

    output_path = render_video(logger.run_dir, config, slow_at=[0.0])

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
        capture_output=True, text=True, check=True,
    )
    duration = float(probe.stdout.strip())
    assert duration > 0
