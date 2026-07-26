"""Run directory: config.yaml, metrics.jsonl, checkpoints, PGN samples.

Each run writes to `runs/<run-id>/` (see planning/SPEC.md sec 5). Nothing
here is Stockfish-dependent -- `rolling_accuracy_cp` stays `None` in every
metrics line until `rlchess/eval.py` exists and is wired in.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import chess
import chess.pgn
import torch
import yaml

from rlchess.policy import ChessPolicyNet
from rlchess.ppo import GameRecord

_RESULT_TO_OUTCOME = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}


class RunLogger:
    def __init__(self, config: dict):
        logging_cfg = config["logging"]
        run_id = logging_cfg["run_id"] or time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(logging_cfg["run_dir"]) / run_id
        self.rolling_window = logging_cfg["rolling_window"]

        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "games").mkdir(parents=True, exist_ok=True)

        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)

        self.metrics_path = self.run_dir / "metrics.jsonl"

        self._outcomes: deque[str] = deque(maxlen=self.rolling_window)
        self._games_played = 0
        self._cum_white_wins = 0
        self._cum_black_wins = 0
        self._cum_draws = 0

    def log_batch(self, step: int, games: list[GameRecord], ppo_stats: dict) -> dict:
        for game in games:
            outcome = _RESULT_TO_OUTCOME[game.result]
            self._games_played += 1
            if outcome == "white":
                self._cum_white_wins += 1
            elif outcome == "black":
                self._cum_black_wins += 1
            else:
                self._cum_draws += 1
            self._outcomes.append(outcome)

        n_roll = len(self._outcomes)
        record = {
            "step": step,
            "games_played": self._games_played,
            "white_wins": self._cum_white_wins,
            "black_wins": self._cum_black_wins,
            "draws": self._cum_draws,
            "rolling_white_winrate": self._outcomes.count("white") / n_roll,
            "rolling_black_winrate": self._outcomes.count("black") / n_roll,
            "rolling_draw_rate": self._outcomes.count("draw") / n_roll,
            "rolling_accuracy_cp": None,
            **ppo_stats,
        }
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def save_checkpoint(self, step: int, policy: ChessPolicyNet) -> Path:
        path = self.run_dir / "checkpoints" / f"ckpt_{step}.pt"
        torch.save(policy.state_dict(), path)
        return path

    def save_games(self, step: int, games: list[GameRecord]) -> Path:
        game_dir = self.run_dir / "games" / f"ckpt_{step}"
        game_dir.mkdir(parents=True, exist_ok=True)
        for i, game in enumerate(games):
            pgn_game = chess.pgn.Game()
            pgn_game.headers["Event"] = "RL Chess Self-Play"
            pgn_game.headers["White"] = "policy"
            pgn_game.headers["Black"] = "policy"
            pgn_game.headers["Result"] = game.result

            node = pgn_game
            board = chess.Board()
            for san in game.moves_san:
                move = board.parse_san(san)
                node = node.add_variation(move)
                board.push(move)

            with open(game_dir / f"game_{i}.pgn", "w") as f:
                print(pgn_game, file=f)
        return game_dir
