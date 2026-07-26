# RL Chess Self-Play

Train a single neural network to play chess against itself with reinforcement
learning, then replay the whole training run as a timelapse — both a rendered
video and an interactive web dashboard.

Two players sit at the board (labelled **White** and **Black**), but under the
hood there is **one shared policy** that plays both sides. Every move from both
colours is training data for that one network. The "two models" are a display
convention for the visualization, not two independently-weighted agents.

## What it does

1. **Self-play** — the policy plays full games against itself. Moves are masked
   to legal moves only (via `python-chess`).
2. **Learn** — PPO (policy gradient) updates the shared policy from the
   trajectories of both colours. Reward = terminal game result plus light
   material shaping.
3. **Checkpoint** — at fixed intervals the policy is snapshotted, a batch of
   sample games is recorded as PGN, and evaluation metrics are written out.
4. **Visualize** — replay training as:
   - a **rendered MP4** that runs accelerated but slows to real-move-by-move
     speed ~4–5 times (at chosen checkpoints) so you can watch actual games at
     different stages of learning;
   - a **React dashboard** with a board replay of any logged game, win-count
     curves over training, and a rolling accuracy curve.

## Metrics

- **Win counts** — White-wins / Black-wins / draws over training time. With a
  shared policy this tracks how the network learns first-move advantage and how
  the draw rate evolves.
- **Accuracy** — mean **centipawn loss vs Stockfish** on sampled moves,
  reported as a **rolling average over the previous ~50 games**. Lower is
  better (0 = matches the engine's best move). This is the only ground-truth
  quality signal; without an engine there is no notion of move "correctness".

## Stack

- **RL / model:** PyTorch, PPO (custom or a lightweight lib), action masking
- **Chess rules / PGN:** `python-chess`
- **Eval engine:** Stockfish (sampled, not every move — it's slow)
- **Video:** offline renderer (board SVG/PNG → frames → MP4 via ffmpeg)
- **Dashboard:** React + a chessboard component (e.g. `react-chessboard` +
  `chess.js`), reading logged PGN + metrics
- **Compute:** cloud GPU(s) for training; CPU is fine for visualization

## Running it

### 1. Setup

```bash
pip install -r requirements.txt
```

If your system Python refuses with an "externally-managed-environment" error
(PEP 668, common on Debian/Ubuntu):

```bash
pip install --user --break-system-packages -r requirements.txt
```

Stockfish is only needed for `eval.py` — everything else works without it:

```bash
sudo apt-get install -y stockfish
```

If it's not on `PATH`, either set the `STOCKFISH_PATH` env var or fill in
`eval.stockfish_path` in the config; leaving both unset makes `eval.py` no-op
cleanly rather than error.

### 2. Train

```bash
python -m rlchess.train --config configs/default.yaml
```

Creates `runs/<run-id>/` (timestamped, unless `logging.run_id` is set) with
`config.yaml` (frozen copy), `metrics.jsonl` (one line per PPO update),
`checkpoints/ckpt_<step>.pt`, and `games/ckpt_<step>/game_<i>.pgn`.

Common variations, all via editing/copying `configs/default.yaml`:

| Want to... | Change |
|---|---|
| Run a fast local smoke test | Shrink `train.num_updates`, `ppo.games_per_batch`, `policy.num_filters`/`num_res_blocks`, `env.max_moves` (see `tests/test_train.py`'s `tiny_config` fixture for the exact tiny values the test suite uses) |
| Name a run instead of auto-timestamping | Set `logging.run_id: "my-run"` |
| Force CPU or GPU | Set `train.device: cpu` or `cuda` instead of `auto` |
| Use a different config file | `--config path/to/other.yaml` |

There's no resume-from-checkpoint flag — every invocation trains a fresh
policy from scratch.

### 3. Evaluate vs. Stockfish (optional)

```bash
python -m rlchess.eval --run runs/<run-id>
```

Samples logged PGN positions, scores them with Stockfish, and writes
`runs/<run-id>/accuracy.jsonl` (rolling mean centipawn loss per checkpoint).
Can be run any time after checkpoints/games exist — it only reads files.

### 4. Render the timelapse video (optional)

```bash
python -m rlchess.render_video --run runs/<run-id> --slow-at 0.1 0.35 0.6 0.85 1.0
```

Requires `ffmpeg` on `PATH`. `--slow-at` are fractions through training
(0=earliest checkpoint, 1=latest) to render in real move-by-move speed;
everything else is accelerated. Writes `runs/<run-id>/timelapse.mp4`.

### 5. Dashboard (optional)

```bash
cd dashboard
npm install
npm run dev
```

Open `http://localhost:5173`. Reads directly from `../runs/` through a
custom Vite dev-server middleware — no separate backend process. Pick a run,
scrub through checkpoints, and watch the board replay, win-rate chart, and
accuracy chart update together.

### 6. Tests

```bash
python -m pytest tests/ -v
```

Fast (tiny configs throughout, no GPU needed) — run after any code change.

### Training on a cloud GPU (e.g. Kaggle)

Only `train.py`/`eval.py`/`render_video.py` are meant to run there — the
dashboard is a local dev server and isn't meant to be opened from a notebook.

1. Get the code onto the instance, e.g. `!git clone <this-repo-url>` in a
   notebook cell (turn on the notebook's "Internet" setting first).
2. `train.device: auto` already resolves to `cuda` automatically — no config
   change needed for GPU.
3. Install only what's missing; **don't** `pip install torch` from
   `requirements.txt` over a preinstalled GPU-linked PyTorch, or you risk
   breaking the CUDA build:
   ```bash
   !apt-get install -y stockfish
   !pip install chess==1.11.2 PyYAML==6.0.1 Pillow==10.2.0 matplotlib==3.6.3
   ```
4. In a notebook, `!cd some-dir` does **not** persist to the next cell (each
   `!` line is its own subprocess) — use `%cd some-dir` instead, or chain
   with `&&` on one line.
5. For anything longer than a few minutes, use "Save & Run All (Commit)"
   rather than running cells interactively — interactive sessions disconnect
   on idle/tab-close, while a committed run finishes in the background and
   attaches `runs/` as the notebook's output.
6. Afterwards, download `runs/<run-id>/` from the notebook output and drop it
   into this repo's local `runs/` to browse it with the dashboard (step 5
   above).

## Design tradeoffs

A few decisions here trade something away on purpose. The first group was
resolved by asking directly at the start of the project; the second group got
settled mid-build and is recorded for traceability in `planning/SPEC.md` §10.

### Decided up front

**Build order: foundation first, not everything at once.** `env.py` +
`policy.py` were built and fully tested in isolation before PPO, training, or
logging touched them, rather than writing the whole system and debugging at
the end. Slower to reach a runnable demo, but bugs (like the move-cap
adjudication needing its own `info["result"]` field) surface at the layer
that caused them instead of three layers downstream.

**Action space: AlphaZero's flat 4672-move encoding, not a simpler
`from×to×promotion` encoding.** A `from_square × to_square` scheme is easier
to reason about but gives a larger, sparser action space with no clean
separation between "geometrically impossible" and "just illegal right now."
4672 (64 squares × 73 direction/distance/knight/underpromotion types) keeps
the policy head a fixed, denser size and is the standard choice for a
masked-policy setup like this — at the cost of a more intricate
`encode_move`/`decode_move` (see the queen-vs-underpromotion split above).

**Install Stockfish immediately, not deferred until `eval.py` was built.**
Traded a bit of early setup friction (a sudo/PEP-668 detour) for `eval.py`'s
Stockfish-dependent tests running for real from day one, instead of sitting
`@pytest.mark.skipif`'d and untrusted until later.

**Compute target: Kaggle GPU, not a paid/dedicated cloud VM.** Free but
quota-limited and session-capped, with no persistent environment (every
session reinstalls dependencies) and no resume-from-checkpoint support built
— a session that gets cut off restarts training from scratch rather than
continuing. Accepted because a hobby-scale run doesn't need dedicated
infrastructure; `train.num_updates: 400` is sized to fit inside one session
rather than requiring resume support to be built.

**No per-move step penalty in the reward.** Considered a small negative
reward per move to push toward shorter, more decisive games, and declined it
to keep the reward to two components: terminal outcome and light material
shaping (`env.material_shaping_coef: 0.01`). Games may run longer/drawish for
longer during early training as a result, but the reward function stays
simple and doesn't risk teaching the policy to end games artificially early
just to dodge the penalty.

### Resolved during implementation (`planning/SPEC.md` §10)

**Board orientation: absolute, not flipped/canonicalized for Black.**
Original AlphaZero flips the board so the policy always "sees" itself moving
up the board; this project keeps it absolute — no flip-transform bugs to
chase, at the cost of some sample efficiency, consistent with the explicit
non-goal of engine-strength play.

**Stockfish evaluation: offline over logged PGN, not inline during
training.** Decouples training entirely from Stockfish being installed, fast,
or even running — at the cost of accuracy numbers lagging behind training
instead of appearing in the same `metrics.jsonl` line they were produced in.

**Video rendering: Pillow direct drawing, not `python-chess` SVG → cairosvg.**
cairosvg needs a system `libcairo` not guaranteed present on every machine
this runs on (dev laptop vs. Kaggle notebook). Pillow-drawn boards are
visually plainer but add zero system dependencies.

## Repo layout

```
rlchess/            # training + env + eval + video (Python)
  env.py            # python-chess wrapper: legal-move masking, encoding, reward
  policy.py         # network (board tensor -> move logits + value)
  ppo.py            # PPO update
  train.py          # self-play loop, checkpointing, logging
  eval.py           # Stockfish centipawn-loss sampling
  logging.py        # run directory: metrics.jsonl + PGN samples
  render_video.py   # accelerated timelapse with slow-motion windows
configs/            # YAML run configs
dashboard/          # React app: board replay + metric charts
runs/               # outputs (gitignored)
tests/              # pytest suite
planning/
  SPEC.md           # detailed technical spec
  CLAUDE.md         # instructions for Claude Code
```

See **planning/SPEC.md** for the full design and **planning/CLAUDE.md** for
build guidance.
