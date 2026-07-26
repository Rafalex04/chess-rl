"""python-chess wrapper: board encoding, legal-move masking, reward, gym-like API.

Board -> tensor encoding: shape (20, 8, 8), absolute orientation (rank index 0
is rank 1 / White's back rank, for both colours -- no board-flip/canonical-
ization for Black; see planning/SPEC.md for the rationale).

Plane layout:
    0-5   White P, N, B, R, Q, K (one plane each)
    6-11  Black P, N, B, R, Q, K
    12    Side to move (all-1 if White to move, all-0 if Black)
    13    Castling rights: White kingside
    14    Castling rights: White queenside
    15    Castling rights: Black kingside
    16    Castling rights: Black queenside
    17    En-passant target square (one-hot, all-0 if none)
    18    Halfmove clock, normalized (clock / 100), broadcast
    19    Repetition count, normalized (min(count, 2) / 2), broadcast

Action space: AlphaZero's flat 4672-move encoding, from_square (0-63) * 73 +
move_type (0-72):
    0-55   "queen-like" moves: 8 directions (N,NE,E,SE,S,SW,W,NW) x 7
           distances (1-7). Covers rook/bishop/queen/king moves, castling
           (a 2-square horizontal king move), ordinary pawn pushes/captures,
           and queen promotions (a pawn's single-step move onto the back
           rank through this same plane defaults to queen promotion).
    56-63  Knight moves (8 fixed L-shaped deltas).
    64-72  Underpromotions only: 3 directions (capture-left, push,
           capture-right) x 3 pieces (N, B, R). Queen promotion is not here.

`encode_move`/`decode_move` are pure functions; `decode_move` only needs
`board` to read the moving piece's colour/type (for promotion-rank
interpretation), not for any side-relative geometry (encoding is absolute).
"""

from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 20
ACTION_SPACE_SIZE = 64 * 73  # 4672

_PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Directions as (delta_rank, delta_file) unit steps, fixed index order.
_DIRECTIONS = [
    (1, 0),  # N
    (1, 1),  # NE
    (0, 1),  # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1),  # SW
    (0, -1),  # W
    (1, -1),  # NW
]
_DIRECTION_TO_IDX = {d: i for i, d in enumerate(_DIRECTIONS)}

# Knight move deltas as (delta_rank, delta_file), fixed index order.
_KNIGHT_DELTAS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]
_KNIGHT_DELTA_TO_IDX = {d: i for i, d in enumerate(_KNIGHT_DELTAS)}

# Underpromotion sub-encoding: 3 directions (by delta_file) x 3 pieces.
_UNDERPROMO_DELTA_FILE_TO_IDX = {-1: 0, 0: 1, 1: 2}
_IDX_TO_UNDERPROMO_DELTA_FILE = {v: k for k, v in _UNDERPROMO_DELTA_FILE_TO_IDX.items()}
_UNDERPROMO_PIECE_TO_IDX = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
_IDX_TO_UNDERPROMO_PIECE = {v: k for k, v in _UNDERPROMO_PIECE_TO_IDX.items()}


def encode_move(move: chess.Move) -> int:
    """Map a chess.Move to an action index in [0, 4672)."""
    from_sq = move.from_square
    to_sq = move.to_square
    delta_rank = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    delta_file = chess.square_file(to_sq) - chess.square_file(from_sq)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        dir_idx = _UNDERPROMO_DELTA_FILE_TO_IDX[delta_file]
        piece_idx = _UNDERPROMO_PIECE_TO_IDX[move.promotion]
        move_type = 64 + dir_idx * 3 + piece_idx
    elif (delta_rank, delta_file) in _KNIGHT_DELTA_TO_IDX:
        move_type = 56 + _KNIGHT_DELTA_TO_IDX[(delta_rank, delta_file)]
    else:
        distance = max(abs(delta_rank), abs(delta_file))
        sign_rank = 0 if delta_rank == 0 else (1 if delta_rank > 0 else -1)
        sign_file = 0 if delta_file == 0 else (1 if delta_file > 0 else -1)
        dir_idx = _DIRECTION_TO_IDX[(sign_rank, sign_file)]
        move_type = dir_idx * 7 + (distance - 1)

    return from_sq * 73 + move_type


def decode_move(board: chess.Board, action: int) -> chess.Move:
    """Map an action index back to a chess.Move, given the current board.

    `board` is only used to read the moving piece's colour/type, which
    disambiguates promotion rank/piece; the from/to squares are derived
    purely from the action index.
    """
    from_sq, move_type = divmod(action, 73)
    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    piece = board.piece_at(from_sq)

    if move_type >= 64:
        sub = move_type - 64
        dir_idx, piece_idx = divmod(sub, 3)
        delta_file = _IDX_TO_UNDERPROMO_DELTA_FILE[dir_idx]
        promotion = _IDX_TO_UNDERPROMO_PIECE[piece_idx]
        delta_rank = 1 if piece.color == chess.WHITE else -1
        to_sq = chess.square(from_file + delta_file, from_rank + delta_rank)
        return chess.Move(from_sq, to_sq, promotion=promotion)

    if move_type >= 56:
        delta_rank, delta_file = _KNIGHT_DELTAS[move_type - 56]
        to_sq = chess.square(from_file + delta_file, from_rank + delta_rank)
        return chess.Move(from_sq, to_sq)

    dir_idx, dist_idx = divmod(move_type, 7)
    sign_rank, sign_file = _DIRECTIONS[dir_idx]
    distance = dist_idx + 1
    to_rank = from_rank + sign_rank * distance
    to_file = from_file + sign_file * distance
    to_sq = chess.square(to_file, to_rank)

    promotion = None
    if piece is not None and piece.piece_type == chess.PAWN:
        if (piece.color == chess.WHITE and to_rank == 7) or (
            piece.color == chess.BLACK and to_rank == 0
        ):
            promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a board into the (20, 8, 8) plane tensor described above."""
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane = _PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        tensor[plane, chess.square_rank(square), chess.square_file(square)] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    if board.ep_square is not None:
        tensor[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0

    tensor[18, :, :] = min(board.halfmove_clock, 100) / 100.0

    rep_count = 1
    if board.is_repetition(2):
        rep_count = 2
    if board.is_repetition(3):
        rep_count = 3
    tensor[19, :, :] = min(rep_count, 2) / 2.0

    return tensor


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Boolean vector, shape (4672,), True at indices of currently legal moves."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(move)] = True
    return mask


def _material_balance(board: chess.Board, color: bool) -> float:
    """Material balance from `color`'s perspective (color's total - opponent's total)."""
    balance = 0.0
    for piece_type, value in _PIECE_VALUES.items():
        balance += value * len(board.pieces(piece_type, color))
        balance -= value * len(board.pieces(piece_type, not color))
    return balance


def _adjudicate_by_material(board: chess.Board) -> str:
    """Result string used when the max-move cap is hit without natural termination."""
    balance = _material_balance(board, chess.WHITE)
    if balance > 0:
        return "1-0"
    if balance < 0:
        return "0-1"
    return "1/2-1/2"


def _terminal_reward(result: str, mover_color: bool) -> float:
    """Terminal reward from the perspective of the side that just moved."""
    if result == "1/2-1/2":
        return 0.0
    mover_won = (result == "1-0") == (mover_color == chess.WHITE)
    return 1.0 if mover_won else -1.0


class ChessEnv:
    """Gym-like single-agent view onto a self-play chess game.

    One shared policy plays both colours; this env just advances the board
    one ply per step() call, from whichever side's turn it currently is, and
    returns the reward from that side's perspective. Propagating a terminal
    outcome back onto the *other* side's last trajectory entry is a
    train.py/ppo.py bookkeeping concern (see planning/SPEC.md sec 3), not
    something this env does.
    """

    def __init__(self, config: dict):
        env_cfg = config["env"]
        self.max_moves = env_cfg["max_moves"]
        self.material_shaping_enabled = env_cfg["material_shaping_enabled"]
        self.material_shaping_coef = env_cfg["material_shaping_coef"]
        self.board = chess.Board()
        self._ply_count = 0

    def reset(self) -> tuple[np.ndarray, dict]:
        self.board = chess.Board()
        self._ply_count = 0
        obs = board_to_tensor(self.board)
        info = {"fen": self.board.fen(), "legal_action_mask": legal_action_mask(self.board)}
        return obs, info

    def legal_action_mask(self) -> np.ndarray:
        return legal_action_mask(self.board)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        move = decode_move(self.board, action)
        mover_color = self.board.turn
        material_before = _material_balance(self.board, mover_color)

        san = self.board.san(move)
        self.board.push(move)
        self._ply_count += 1

        done = False
        result = None
        # claim_draw=True so 50-move-rule / threefold-repetition draws are
        # caught here rather than always falling through to the move cap.
        if self.board.is_game_over(claim_draw=True):
            done = True
            result = self.board.result(claim_draw=True)
        elif self._ply_count >= self.max_moves:
            done = True
            result = _adjudicate_by_material(self.board)

        if done:
            reward = _terminal_reward(result, mover_color)
        elif self.material_shaping_enabled:
            material_after = _material_balance(self.board, mover_color)
            reward = self.material_shaping_coef * (material_after - material_before)
        else:
            reward = 0.0

        obs = board_to_tensor(self.board)
        mask = (
            np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            if done
            else legal_action_mask(self.board)
        )
        info = {
            "san": san,
            "uci": move.uci(),
            "fen": self.board.fen(),
            "legal_action_mask": mask,
        }
        return obs, reward, done, info
