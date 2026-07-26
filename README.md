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

## Quick start

```bash
# install
pip install -r requirements.txt          # torch, python-chess, etc.
# install Stockfish binary separately and set STOCKFISH_PATH

# train (writes checkpoints, PGN samples, metrics to runs/<run-id>/)
python -m rlchess.train --config configs/default.yaml

# render the timelapse video
python -m rlchess.render_video --run runs/<run-id> --slow-at 0.1 0.35 0.6 0.85 1.0

# launch the dashboard
cd dashboard && npm install && npm run dev
```

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
SPEC.md             # detailed technical spec
CLAUDE.md           # instructions for Claude Code
```

See **SPEC.md** for the full design and **CLAUDE.md** for build guidance.
