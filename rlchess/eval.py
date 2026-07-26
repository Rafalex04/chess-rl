"""Stockfish-based accuracy: sampled centipawn loss over logged PGN, rolling
mean over the last ~50 games.

Design choice (resolves planning/SPEC.md sec 10's open item): runs
**offline over the PGN files `logging.py` already writes** at each
checkpoint, rather than inline during training. Keeps the self-play/PPO
loop's throughput and reliability independent of Stockfish being installed,
slow, or absent -- exactly what CLAUDE.md's "the pipeline must still run
without it" guard asks for. Writes a separate `runs/<id>/accuracy.jsonl`
(one line per evaluated checkpoint) rather than rewriting `metrics.jsonl` in
place, since JSONL logs are meant to be append-only, not patched.

centipawn loss = eval(best move) - eval(chosen move), both from the mover's
point of view, clipped to [0, cp_clip] so a single blunder doesn't blow up
the average.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import deque
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def get_stockfish_path(config: dict) -> str | None:
    """STOCKFISH_PATH env var takes precedence over config["eval"]["stockfish_path"]."""
    return os.environ.get("STOCKFISH_PATH") or config.get("eval", {}).get("stockfish_path")


def is_stockfish_available(config: dict) -> bool:
    path = get_stockfish_path(config)
    return bool(path) and shutil.which(path) is not None


def _score_to_cp(score: chess.engine.PovScore, mover_color: bool) -> float:
    # Mate scores collapse to a large-but-finite value so arithmetic on them
    # (subtraction, clipping) stays well-defined.
    return float(score.pov(mover_color).score(mate_score=100_000))


def evaluate_pgn_file(
    engine: chess.engine.SimpleEngine,
    pgn_path: Path,
    sample_rate: float,
    depth: int,
    cp_clip: float,
    rng: random.Random,
) -> float | None:
    """Mean centipawn loss over sampled moves in one game, or None if none sampled."""
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)
    if game is None:
        return None

    losses = []
    board = game.board()
    for move in game.mainline_moves():
        if rng.random() < sample_rate:
            mover_color = board.turn
            info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
            score_before = _score_to_cp(info_before["score"], mover_color)

            board_after = board.copy()
            board_after.push(move)
            info_after = engine.analyse(board_after, chess.engine.Limit(depth=depth))
            score_after = _score_to_cp(info_after["score"], mover_color)

            cp_loss = max(0.0, min(cp_clip, score_before - score_after))
            losses.append(cp_loss)
        board.push(move)

    if not losses:
        return None
    return sum(losses) / len(losses)


def evaluate_run(run_dir: Path, config: dict, seed: int = 0) -> Path | None:
    """Walk runs/<id>/games/ckpt_<step>/*.pgn in step order, sampling centipawn
    loss per game and writing a rolling-mean accuracy curve to
    runs/<id>/accuracy.jsonl. Returns None (no-op) if Stockfish is unavailable.
    """
    if not is_stockfish_available(config):
        return None

    eval_cfg = config["eval"]
    stockfish_path = get_stockfish_path(config)
    rng = random.Random(seed)

    games_dir = run_dir / "games"
    checkpoint_dirs = sorted(
        games_dir.glob("ckpt_*"), key=lambda p: int(p.name.removeprefix("ckpt_"))
    )

    accuracy_path = run_dir / "accuracy.jsonl"
    rolling = deque(maxlen=eval_cfg["rolling_window"])

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        with open(accuracy_path, "w") as out:
            for ckpt_dir in checkpoint_dirs:
                step = int(ckpt_dir.name.removeprefix("ckpt_"))
                for pgn_path in sorted(ckpt_dir.glob("*.pgn")):
                    mean_loss = evaluate_pgn_file(
                        engine,
                        pgn_path,
                        eval_cfg["sample_rate"],
                        eval_cfg["depth"],
                        eval_cfg["cp_clip"],
                        rng,
                    )
                    if mean_loss is not None:
                        rolling.append(mean_loss)

                if rolling:
                    record = {
                        "step": step,
                        "rolling_accuracy_cp": sum(rolling) / len(rolling),
                    }
                    out.write(json.dumps(record) + "\n")
    finally:
        engine.quit()

    return accuracy_path


def _main() -> None:
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Path to runs/<run-id>/")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if not is_stockfish_available(config):
        print("Stockfish not available (STOCKFISH_PATH unset/not found) -- skipping eval.")
        return

    accuracy_path = evaluate_run(Path(args.run), config)
    print(f"Wrote {accuracy_path}")


if __name__ == "__main__":
    _main()
