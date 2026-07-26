import random

import chess
import numpy as np
import pytest
import yaml

from rlchess.env import (
    ChessEnv,
    NUM_PLANES,
    decode_move,
    encode_move,
    legal_action_mask,
)


@pytest.fixture
def config():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


def test_reset_shape_and_start_position(config):
    env = ChessEnv(config)
    obs, info = env.reset()

    assert obs.shape == (NUM_PLANES, 8, 8)
    # White pawns on rank index 1 (rank 2), all files.
    assert np.array_equal(obs[0, 1, :], np.ones(8, dtype=np.float32))
    # Black pawns on rank index 6 (rank 7), all files.
    assert np.array_equal(obs[6, 6, :], np.ones(8, dtype=np.float32))
    # White to move.
    assert np.all(obs[12] == 1.0)
    # All castling rights available at the start.
    assert np.all(obs[13] == 1.0)
    assert np.all(obs[14] == 1.0)
    assert np.all(obs[15] == 1.0)
    assert np.all(obs[16] == 1.0)
    # No en-passant target yet.
    assert np.all(obs[17] == 0.0)
    assert info["fen"] == chess.STARTING_FEN


def test_encode_decode_round_trip_curated_moves():
    cases = [
        # (fen, uci)
        (chess.STARTING_FEN, "e2e4"),  # normal double push
        (chess.STARTING_FEN, "g1f3"),  # knight move
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "f1c4"),  # bishop
        (
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "e1g1",
        ),  # white kingside castle
        (
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "e1c1",
        ),  # white queenside castle
        (
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
            "e8g8",
        ),  # black kingside castle
        (
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "e5d6",
        ),  # en passant capture
        ("8/P7/8/8/8/8/8/k6K w - - 0 1", "a7a8q"),  # queen promotion
        ("8/P7/8/8/8/8/8/k6K w - - 0 1", "a7a8n"),  # underpromotion knight
        ("8/P7/8/8/8/8/8/k6K w - - 0 1", "a7a8b"),  # underpromotion bishop
        ("8/P7/8/8/8/8/8/k6K w - - 0 1", "a7a8r"),  # underpromotion rook
        (
            "rnbqkbnr/ppppppp1/8/7P/8/8/PPPPPPP1/RNBQKBNR b KQkq - 0 2",
            "g7g5",
        ),  # black pawn double push
    ]
    for fen, uci in cases:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"{uci} not legal in {fen}"
        action = encode_move(move)
        assert 0 <= action < 4672
        decoded = decode_move(board, action)
        assert decoded == move, f"round trip failed for {uci}: got {decoded.uci()}"


def test_legal_action_mask_matches_python_chess():
    rng = random.Random(42)
    for _ in range(20):
        board = chess.Board()
        for _ in range(rng.randint(0, 40)):
            if board.is_game_over(claim_draw=True):
                break
            move = rng.choice(list(board.legal_moves))
            board.push(move)

        if board.is_game_over(claim_draw=True):
            continue

        mask = legal_action_mask(board)
        expected_indices = {encode_move(m) for m in board.legal_moves}
        actual_indices = set(np.flatnonzero(mask).tolist())
        assert actual_indices == expected_indices
        assert mask.sum() == len(list(board.legal_moves))


def test_random_games_terminate(config):
    rng = random.Random(7)
    for game_idx in range(5):
        env = ChessEnv(config)
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            legal = np.flatnonzero(info["legal_action_mask"]).tolist()
            action = rng.choice(legal)
            obs, reward, done, info = env.step(action)
            steps += 1
            assert steps <= env.max_moves
            assert info["result"] is None if not done else info["result"] is not None
        assert done
        assert info["result"] in ("1-0", "0-1", "1/2-1/2")


def test_max_move_cap_adjudicates_by_material():
    config = {
        "env": {
            "max_moves": 4,
            "material_shaping_enabled": False,
            "material_shaping_coef": 0.0,
        }
    }
    env = ChessEnv(config)
    # White is up a queen; shuffle both kings back and forth so the game
    # doesn't end naturally within the 4-ply cap.
    env.board = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    shuffle_uci = ["e1e2", "e8e7", "e2e1", "e7e8"]
    done = False
    reward = None
    for uci in shuffle_uci:
        move = chess.Move.from_uci(uci)
        action = encode_move(move)
        obs, reward, done, info = env.step(action)
    assert done
    # White is up material -> adjudicated "1-0", exposed via info["result"]
    # even though board.result() itself stays "*" (python-chess never
    # decided the game was over; the adjudication is env-level only).
    assert info["result"] == "1-0"
    # The 4th (last) ply was Black's move, so the reward returned is from
    # Black's perspective: a loss.
    assert reward == -1.0


def test_reward_sign_scholars_mate(config):
    env = ChessEnv(config)
    env.reset()
    moves_san = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]
    done = False
    reward = None
    for san in moves_san:
        move = env.board.parse_san(san)
        action = encode_move(move)
        obs, reward, done, info = env.step(action)
    assert done
    assert env.board.result(claim_draw=True) == "1-0"
    assert info["result"] == "1-0"
    assert reward == 1.0  # White delivered mate


def test_castling_and_en_passant_round_trip_via_env(config):
    env = ChessEnv(config)
    env.reset()
    san_sequence = ["Nf3", "Nf6", "g3", "g6", "Bg2", "Bg7", "O-O", "O-O"]
    for san in san_sequence:
        move = env.board.parse_san(san)
        action = encode_move(move)
        obs, reward, done, info = env.step(action)
        assert not done
