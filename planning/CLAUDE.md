# CLAUDE.md — guidance for Claude Code

Read **SPEC.md** first; it is the source of truth. This file is how to work in
this repo.

## The one thing to get right

There is **one shared policy** playing both colours in self-play. Do NOT build
two separate networks. "White" and "Black" are display labels for the
visualization; mechanically it's a single network updated on trajectories from
both sides. If you find yourself instantiating two policies, stop — that's wrong.

## Build order

Build and verify incrementally. Do not write the whole system then debug at the
end.

1. **`env.py`** — `python-chess` wrapper: board→tensor encoding, legal-move
   mask, reward, gym-like API. **Test first:** random legal games run to
   completion, mask only ever allows legal moves, encoding round-trips a few
   known FENs.
2. **`policy.py`** — network (board tensor → masked policy logits + value).
   Test: forward pass shape-checks; masked illegal logits are −inf.
3. **`ppo.py` + `train.py`** — self-play + PPO. **Test on a tiny run first**
   (few games, few steps) and confirm loss/entropy are finite and it doesn't
   crash before scaling up. Verify the mask is applied both when acting and when
   recomputing logprobs in the update.
4. **`logging.py`** — run directory: `metrics.jsonl`, checkpoints, PGN samples.
   Confirm PGNs are valid (re-parseable by `python-chess` and by `chess.js`).
5. **`eval.py`** — Stockfish centipawn-loss sampling + rolling-50 accuracy.
   Guard behind a "Stockfish available" check; the pipeline must still run
   without it (accuracy just unavailable).
6. **`render_video.py`** — metric timelapse + 4–5 slow-motion board windows →
   MP4.
7. **`dashboard/`** — React board replay + charts, reading the run directory.

Get 1–4 working end-to-end (a run that trains, logs, and produces valid PGN)
before touching eval, video, or dashboard.

## Correctness traps — check these

- **Reward sign per colour.** Each side's trajectory is scored from its own
  perspective. A win for White is +1 for White's moves and the loss is −1 for
  Black's moves in that game. Get this consistent or PPO learns garbage.
- **Mask in the PPO ratio.** Apply the legal mask when recomputing logprobs in
  the update, not just when acting. Otherwise the importance ratio is wrong.
- **Draw / infinite games.** Enforce a max-move cap that adjudicates by material.
  Early self-play draws and shuffles a lot; that's expected, but don't let games
  run forever.
- **Centipawn loss = eval(engine best) − eval(chosen move)**, clipped. Lower is
  better. Don't invert it.
- **Rolling window is over games, default 50**, and configurable.

## Conventions

- Config-driven: every hyperparameter and path lives in `configs/*.yaml`, not
  hardcoded. `STOCKFISH_PATH` from env/config.
- All run outputs go under `runs/<run-id>/` (gitignored). Never commit weights,
  PGN dumps, or videos.
- Python: type hints, small focused modules, `python-chess` for anything
  chess-rules-related (never reimplement legality).
- Keep the network small and configurable — target single cloud GPU.
- Comment the known-hard parts (sparse reward, mask handling, reward sign) so
  future edits don't silently break them.

## Expectations to keep honest

- PPO-no-search will play weak-to-moderate chess. The deliverable is *visible
  improvement over training*, not strong play. Don't over-engineer toward
  strength; if strength is later wanted, that's an AlphaZero-lite rewrite, not a
  tweak.
- If a design choice contradicts SPEC.md, update SPEC.md in the same change and
  say so — don't silently diverge.

## Definition of done (v1)

A single `train.py` run on a cloud GPU produces a `runs/<id>/` with metrics,
checkpoints, and valid sampled PGNs; `render_video.py` turns it into an
accelerated MP4 with ~5 slow-motion game windows; the React dashboard replays
any logged game and shows the win-count and rolling-accuracy curves over
training.
