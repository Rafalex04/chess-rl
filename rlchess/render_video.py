"""Timelapse video: metric curves growing fast, pausing into slow motion at
a handful of checkpoints to play through a real sampled game.

Design choice (resolves planning/SPEC.md sec 10's open item on the video
renderer): boards are drawn directly with **Pillow** (piece = a filled/
outlined circle + a K/Q/R/B/N/P letter), not `python-chess` SVG->cairosvg.
cairosvg pulls in a system libcairo dependency that isn't guaranteed present
on every machine this project runs on (local dev box vs. a Kaggle GPU
notebook); Pillow + a couple of near-universal system fonts (falling back to
Pillow's bundled default font if even those are missing) has no such
portability risk.

Frames are written to a temp directory as PNGs and encoded with `ffmpeg`
(matches README's `render_video.py` description: "board SVG/PNG -> frames ->
MP4 via ffmpeg").
"""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
from pathlib import Path

import chess
import chess.pgn
import matplotlib

matplotlib.use("Agg")  # headless: no display available on a training box
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

_PIECE_LETTERS = {
    chess.KING: "K",
    chess.QUEEN: "Q",
    chess.ROOK: "R",
    chess.BISHOP: "B",
    chess.KNIGHT: "N",
    chess.PAWN: "P",
}

_LIGHT_SQUARE = (240, 217, 181)
_DARK_SQUARE = (181, 136, 99)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_board_frame(board: chess.Board, size: int = 480, caption: str | None = None) -> Image.Image:
    """Board drawn from White's side (absolute orientation, matching env.py's
    encoding -- no per-turn flipping)."""
    square_size = size // 8
    board_px = square_size * 8
    caption_height = int(square_size * 0.4) if caption else 0

    img = Image.new("RGB", (board_px, board_px + caption_height), "black")
    draw = ImageDraw.Draw(img)
    piece_font = _load_font(int(square_size * 0.5))

    for rank in range(8):
        for file in range(8):
            x0 = file * square_size
            y0 = (7 - rank) * square_size
            color = _LIGHT_SQUARE if (rank + file) % 2 == 0 else _DARK_SQUARE
            draw.rectangle([x0, y0, x0 + square_size, y0 + square_size], fill=color)

            piece = board.piece_at(chess.square(file, rank))
            if piece is None:
                continue
            letter = _PIECE_LETTERS[piece.piece_type]
            fill = "white" if piece.color == chess.WHITE else "black"
            outline = "black" if piece.color == chess.WHITE else "white"
            cx, cy = x0 + square_size / 2, y0 + square_size / 2
            radius = square_size * 0.38
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=fill, outline=outline, width=2,
            )
            bbox = draw.textbbox((0, 0), letter, font=piece_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]),
                letter, fill=outline, font=piece_font,
            )

    if caption:
        caption_font = _load_font(int(caption_height * 0.5))
        draw.text((8, board_px + caption_height * 0.15), caption, fill="white", font=caption_font)

    return img


def render_slow_motion_frames(
    pgn_path: Path, board_size: int, fps: int, seconds_per_move: float, caption_prefix: str
) -> list[Image.Image]:
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)

    frames_per_move = max(1, round(fps * seconds_per_move))
    board = game.board()
    frames = [render_board_frame(board, board_size, f"{caption_prefix} | move 0")] * frames_per_move

    for i, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        frame = render_board_frame(board, board_size, f"{caption_prefix} | move {i}")
        frames.extend([frame] * frames_per_move)

    return frames


def select_slow_motion_checkpoints(available_steps: list[int], fractions: list[float]) -> list[int]:
    if not available_steps:
        return []
    max_step = max(available_steps)
    selected: list[int] = []
    for frac in fractions:
        target = frac * max_step
        nearest = min(available_steps, key=lambda s: abs(s - target))
        if nearest not in selected:
            selected.append(nearest)
    return selected


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _nearest_accuracy(accuracy: list[dict], step: int) -> float | None:
    candidates = [a for a in accuracy if a["step"] <= step]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a["step"])["rolling_accuracy_cp"]


def render_metric_timelapse_frames(
    metrics: list[dict], accuracy: list[dict], num_frames: int, size: tuple[int, int]
) -> list[Image.Image]:
    if not metrics:
        return []

    steps = [m["step"] for m in metrics]
    white_rates = [m["rolling_white_winrate"] for m in metrics]
    black_rates = [m["rolling_black_winrate"] for m in metrics]
    draw_rates = [m["rolling_draw_rate"] for m in metrics]
    acc_steps = [a["step"] for a in accuracy]
    acc_values = [a["rolling_accuracy_cp"] for a in accuracy]

    dpi = 100
    fig_w, fig_h = size[0] / dpi, size[1] / dpi
    n = len(metrics)
    # A single-step run would otherwise give matplotlib an empty (min==max)
    # xlim range.
    x_max = steps[-1] if steps[-1] > steps[0] else steps[0] + 1

    frames = []
    for i in range(num_frames):
        cutoff = max(1, round((i + 1) / num_frames * n))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=dpi)
        ax1.plot(steps[:cutoff], white_rates[:cutoff], label="White", color="#4C72B0")
        ax1.plot(steps[:cutoff], black_rates[:cutoff], label="Black", color="#C44E52")
        ax1.plot(steps[:cutoff], draw_rates[:cutoff], label="Draw", color="#8C8C8C")
        ax1.set_xlim(steps[0], x_max)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Rolling win rate")
        ax1.legend(loc="upper right", fontsize=8)

        if acc_values:
            acc_cutoff = max(1, round((i + 1) / num_frames * len(acc_values)))
            ax2.plot(acc_steps[:acc_cutoff], acc_values[:acc_cutoff], color="#55A868")
            ax2.set_xlim(steps[0], x_max)
        else:
            ax2.text(
                0.5, 0.5, "Accuracy unavailable (no Stockfish)",
                ha="center", va="center", transform=ax2.transAxes,
            )
        ax2.set_ylabel("Rolling CP loss")
        ax2.set_xlabel("Training step")

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    return frames


def _encode_frames_to_mp4(frame_dir: Path, fps: int, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def render_video(run_dir: Path, config: dict, slow_at: list[float] | None = None) -> Path:
    video_cfg = config["video"]
    fps = video_cfg["fps"]
    board_size = video_cfg["board_size"]
    slow_at = slow_at if slow_at is not None else video_cfg["slow_at"]

    metrics = _load_jsonl(run_dir / "metrics.jsonl")
    if not metrics:
        raise ValueError(f"No metrics.jsonl found under {run_dir}")
    accuracy = _load_jsonl(run_dir / "accuracy.jsonl")

    games_dir = run_dir / "games"
    available_steps = (
        sorted(int(p.name.removeprefix("ckpt_")) for p in games_dir.glob("ckpt_*"))
        if games_dir.exists()
        else []
    )
    selected_steps = select_slow_motion_checkpoints(available_steps, slow_at)

    num_metric_frames = max(1, round(video_cfg["metric_timelapse_seconds"] * fps))
    output_path = run_dir / "timelapse.mp4"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        frame_idx = 0

        for frame in render_metric_timelapse_frames(
            metrics, accuracy, num_metric_frames, (board_size, board_size)
        ):
            frame.save(tmp_path / f"frame_{frame_idx:06d}.png")
            frame_idx += 1

        for step in selected_steps:
            pgn_files = sorted((games_dir / f"ckpt_{step}").glob("*.pgn"))
            if not pgn_files:
                continue
            acc_value = _nearest_accuracy(accuracy, step)
            caption = f"step {step}" + (f" | acc {acc_value:.0f}cp" if acc_value is not None else "")
            slow_frames = render_slow_motion_frames(
                pgn_files[0], board_size, fps, video_cfg["slow_motion_seconds_per_move"], caption
            )
            for frame in slow_frames:
                frame.save(tmp_path / f"frame_{frame_idx:06d}.png")
                frame_idx += 1

        if frame_idx == 0:
            raise ValueError("No frames generated -- nothing to render")

        _encode_frames_to_mp4(tmp_path, fps, output_path)

    return output_path


def _main() -> None:
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Path to runs/<run-id>/")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--slow-at", type=float, nargs="+", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_path = render_video(Path(args.run), config, slow_at=args.slow_at)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    _main()
