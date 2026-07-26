# SPEC — RL Chess Self-Play

Technical specification. This is the source of truth for what to build. If
implementation diverges from this, update this file.

## 0. Design decisions (fixed)

- **One shared policy**, self-play. The network plays both colours; both sides'
  moves are training data. "Two models (White/Black)" is a *display* label only.
- **Algorithm: PPO** (policy gradient), no tree search. Expectation is
  moderate play that visibly improves — not engine-strength chess. This is
  accepted; the point is watching learning happen.
- **Rules/PGN via `python-chess`.** No hand-rolled chess engine.
- **Legal-move action masking** at every step (illegal moves are never
  sampled).
- **Accuracy = mean centipawn loss vs Stockfish**, sampled, rolling 50-game avg.
- **Compute: cloud GPU** for training; visualization runs on CPU.
- **Two output surfaces: rendered MP4 + React dashboard**, both fed from the
  same logged run directory.

## 1. Environment (`rlchess/env.py`)

Wrap `python-chess`.

**State encoding (board → tensor):** `(20, 8, 8)`, fixed. Planes 0-5 white
P/N/B/R/Q/K, 6-11 black P/N/B/R/Q/K, 12 side-to-move, 13-16 castling rights
(white-K, white-Q, black-K, black-Q), 17 en-passant target, 18 halfmove clock
(normalized), 19 repetition count (normalized). Exact layout documented in
`rlchess/env.py`'s module docstring.

Indexing is **absolute** (rank index 0 = rank 1 / White's back rank) for both
colours — unlike original AlphaZero, the board is **not flipped/canonicalized**
for Black. Rationale: a flip transform (board + move squares) is a common
source of orientation bugs, and this project explicitly targets "moderate,
not engine-strength" play (sec 9) — simplicity was chosen over the sample-
efficiency benefit of canonicalization.

**Action space:** resolved to the flat **4672-move AlphaZero encoding** (64
from-squares × 73 move types incl. underpromotions). Layout: types 0-55 are
"queen-like" moves (8 directions × 7 distances — this also covers rook/
bishop/king/pawn moves, castling, and queen promotions, since those are all
geometric subsets of a queen's reach from a square); 56-63 are the 8 knight
deltas; 64-72 are underpromotions only (3 directions × N/B/R). Full detail
and lookup tables in `rlchess/env.py`. MUST support a **legal-move mask**: a
boolean vector over the action space marking legal moves in the current
position. Illegal logits are set to **−inf** (the fixed sentinel, applied via
`rlchess/policy.py`'s `apply_action_mask`, shared between act-time inference
and PPO's later logprob recomputation) before softmax.

**Reward:**
- Terminal: +1 win, −1 loss, 0 draw (from the perspective of the side that just
  moved / trajectory owner — be explicit and consistent).
- Shaping (light, tunable, default small weight): change in material balance
  after the move. Keep the shaping coefficient low so it doesn't dominate the
  real objective. Make it toggleable in config.
- Discount γ configurable (default 0.99).

**Step / reset:** standard gym-like API (`reset()`, `step(action) ->
obs, reward, done, info`). `info` carries the SAN/UCI move and the resulting
FEN so games can be logged.

**Termination:** uses `board.is_game_over(claim_draw=True)` (not the
python-chess default `claim_draw=False`) so 50-move-rule and threefold-
repetition draws are caught through normal termination. If the game hasn't
terminated naturally by `env.max_moves` plies (config, default **200**), it
is cut off and adjudicated by material balance: White ahead → `"1-0"`, Black
ahead → `"0-1"`, equal → `"1/2-1/2"`.

## 2. Policy network (`rlchess/policy.py`)

Input: board tensor. Output: (policy logits over action space, scalar value).
- Conv stack / small ResNet trunk → two heads (policy, value).
- Value head used as the PPO critic.
- Keep it small enough to train on a single cloud GPU in reasonable time; size
  is configurable.

## 3. PPO (`rlchess/ppo.py`, `rlchess/train.py`)

**Self-play loop:**
1. Play a batch of games with the current policy (both colours = same net).
   Store trajectories: (state, action, mask, logprob, value, reward) per step,
   for **both** sides.
2. Compute returns + GAE advantages. Terminal reward assigned at game end and
   propagated back; be careful with the sign per colour (each side's trajectory
   is scored from its own perspective).
3. PPO update: clipped surrogate objective, value loss, entropy bonus. Standard
   hyperparameters, all in config.
4. Log, checkpoint, repeat.

**Practical notes / known hard parts (call these out in code comments):**
- Sparse reward + long horizon → training may be slow/unstable. Material shaping
  and entropy bonus help. Consider curriculum (shorter games / draw-adjudication
  by move cap) — add a max-move cap that adjudicates by material if reached.
- Draws are common early. That's expected.
- Mask MUST be applied both when acting and when recomputing logprobs in the
  update, or the ratio is wrong.

## 4. Evaluation — accuracy (`rlchess/eval.py`)

- Uses **Stockfish** via `python-chess`'s UCI interface. Binary path from
  `STOCKFISH_PATH` env var / config.
- For a **sampled** subset of positions (NOT every move — too slow): get
  Stockfish's eval of the position before the move and of the policy's chosen
  move; **centipawn loss = eval(best move) − eval(chosen move)**, clipped at a
  cap (e.g. 1000cp) so blunders don't blow up the average.
- Aggregate per game, then report a **rolling mean over the last ~50 games**
  (window size configurable). This is the "accuracy" curve.
- Sampling rate (fraction of moves evaluated), Stockfish depth/time, and the
  rolling window are all config knobs. Keep eval cheap enough not to bottleneck
  training; it can also run offline over logged PGN if preferred.

## 5. Logging (`rlchess/logging.py`)

Each run writes to `runs/<run-id>/`:
- `config.yaml` — resolved config for the run.
- `metrics.jsonl` — one JSON line per logging step:
  `{step, games_played, white_wins, black_wins, draws,
    rolling_white_winrate, rolling_black_winrate, rolling_draw_rate,
    rolling_accuracy_cp, ppo_loss, entropy, ...}`.
- `checkpoints/ckpt_<step>.pt` — model weights at each checkpoint.
- `games/ckpt_<step>/*.pgn` — a batch of **sampled full games** recorded at each
  checkpoint (this is what the timelapse replays). Include result + move list in
  standard PGN so both the video renderer and the JS board can read them.

Checkpoint interval configurable (by step or games-played count).

## 6. Timelapse video (`rlchess/render_video.py`)

Goal: one accelerated MP4 of the whole run that **pauses into slow motion ~4–5
times** so you can watch real games at different training stages.

- **Metric timelapse portion:** animate the win-count and accuracy curves
  growing over training steps (fast).
- **Slow-motion windows:** at ~4–5 chosen checkpoints (default evenly spaced,
  e.g. 10%, 35%, 60%, 85%, 100% of training — overridable via `--slow-at`),
  cut to a **board view** and play through one sampled game from that checkpoint
  at watchable move speed (e.g. 1–2 moves/sec), showing pieces on a real board.
  Caption with the checkpoint step and current accuracy.
- Render boards from `python-chess` SVG (or a PNG board renderer) → frames →
  MP4 via ffmpeg. Frame rate + slow-motion move duration configurable.
- Output: `runs/<run-id>/timelapse.mp4`.

## 7. Dashboard (`dashboard/`, React)

Reads the same `runs/<run-id>/` outputs (served statically or via a tiny
endpoint).
- **Board replay:** pick any logged game (by checkpoint) and step/auto-play
  through it on a real chessboard. Use `react-chessboard` + `chess.js` (parse
  the PGN client-side).
- **Win chart:** White-wins / Black-wins / draws over training steps (from
  `metrics.jsonl`), with a scrubber to move through training time.
- **Accuracy chart:** rolling centipawn-loss curve over training steps.
- **Timelapse mode:** a scrubber/play button that advances the charts through
  training and lets you jump to the sampled game at each checkpoint — the
  interactive analogue of the MP4.

## 8. Config (`configs/default.yaml`)

All knobs live here: network size, PPO hyperparams (clip, lr, epochs,
batch/minibatch, entropy coef, value coef, γ, GAE λ), reward shaping coef +
toggle, max-move cap, checkpoint interval, sample-games-per-checkpoint,
Stockfish path/depth/sampling-rate, rolling window (default 50), slow-motion
checkpoints for video.

## 9. Non-goals

- Not aiming for engine-strength play.
- No distributed multi-node training required (single cloud GPU is the target;
  multi-GPU optional).
- No opening books, endgame tablebases, or MCTS.

## 10. Open items to confirm during build

- ~~Exact action-space encoding (4672 vs from×to×promo).~~ Resolved: 4672
  AlphaZero encoding (sec 1).
- Whether Stockfish eval runs inline during training or offline over PGN.
- Video renderer: `python-chess` SVG→PNG (cairosvg) vs a dedicated board image
  lib.
