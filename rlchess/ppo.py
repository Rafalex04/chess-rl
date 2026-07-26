"""Self-play data collection + PPO update.

One shared policy plays both colours (planning/CLAUDE.md's "the one thing to
get right"). Per game, moves are recorded into two per-colour trajectories
(white's decisions, black's decisions), since each colour is its own
decision process for the purposes of returns/advantages -- GAE is computed
over each colour's own move sequence, not the interleaved ply order.

Known-hard part #1 (reward sign): `ChessEnv.step` returns the terminal
reward correctly for whichever colour's move ended the game, but the *other*
colour's last recorded move only got a shaping/zero reward at the time (the
game wasn't over yet from their perspective). That colour's last reward must
be overwritten with the correct terminal value once the game result is known
-- see `_propagate_terminal_reward`.

Known-hard part #2 (mask in the ratio): the legal-move mask used when an
action was sampled must be reapplied when recomputing logprobs during the
PPO update, or the importance ratio is wrong. Both `collect_self_play_games`
and `ppo_update` route through `ChessPolicyNet.forward(obs, mask=...)`,
which uses the same `apply_action_mask` either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from rlchess.env import ChessEnv, terminal_reward
from rlchess.policy import ChessPolicyNet


@dataclass
class ColorTrajectory:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


@dataclass
class GameRecord:
    trajectories: dict[bool, ColorTrajectory]
    result: str
    moves_san: list[str]


def _propagate_terminal_reward(
    trajectories: dict[bool, ColorTrajectory], last_mover_color: bool, result: str
) -> None:
    """Fix up the non-terminal-mover's last reward to reflect the game outcome.

    The colour whose move ended the game already has the correct terminal
    reward (env.py's own logic). The other colour's last step is
    *overwritten* (not added to) with their terminal reward, mirroring how
    env.py itself replaces rather than adds on the natural terminal step.
    """
    other_color = not last_mover_color
    other_traj = trajectories[other_color]
    if not other_traj.rewards:
        return
    other_traj.rewards[-1] = terminal_reward(result, other_color)


def compute_gae(
    rewards: list[float], values: list[float], gamma: float, gae_lambda: float
) -> tuple[list[float], list[float]]:
    """Standard GAE over a single trajectory. Episodic: bootstraps 0 after
    the last step (the max-move cap is itself a terminal event with its own
    adjudicated reward, not a truncation-without-reward case).
    """
    length = len(rewards)
    advantages = [0.0] * length
    last_gae = 0.0
    for t in reversed(range(length)):
        next_value = values[t + 1] if t + 1 < length else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae
    returns = [advantages[t] + values[t] for t in range(length)]
    return advantages, returns


def collect_self_play_games(
    config: dict, policy: ChessPolicyNet, num_games: int, device: torch.device
) -> list[GameRecord]:
    """`config` is the full resolved config (same one ChessEnv/ChessPolicyNet
    take), not just the `env:` sub-section."""
    records = []
    for _ in range(num_games):
        env = ChessEnv(config)
        obs, info = env.reset()
        trajectories = {chess.WHITE: ColorTrajectory(), chess.BLACK: ColorTrajectory()}
        moves_san: list[str] = []
        done = False
        last_mover_color = None

        while not done:
            mover_color = env.board.turn
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_np = info["legal_action_mask"]
            mask_t = torch.tensor(mask_np, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, value = policy(obs_t, mask=mask_t)
                dist = torch.distributions.Categorical(logits=logits)
                action_t = dist.sample()
                logprob_t = dist.log_prob(action_t)

            action = int(action_t.item())
            traj = trajectories[mover_color]
            traj.obs.append(obs)
            traj.actions.append(action)
            traj.masks.append(mask_np)
            traj.logprobs.append(float(logprob_t.item()))
            traj.values.append(float(value.item()))

            obs, reward, done, info = env.step(action)
            traj.rewards.append(reward)
            moves_san.append(info["san"])
            last_mover_color = mover_color

        _propagate_terminal_reward(trajectories, last_mover_color, info["result"])
        records.append(
            GameRecord(trajectories=trajectories, result=info["result"], moves_san=moves_san)
        )

    return records


@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    masks: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def build_ppo_batch(
    games: list[GameRecord], gamma: float, gae_lambda: float, device: torch.device
) -> PPOBatch:
    all_obs, all_actions, all_masks = [], [], []
    all_logprobs, all_advantages, all_returns = [], [], []

    for game in games:
        for traj in game.trajectories.values():
            if not traj.rewards:
                continue
            advantages, returns = compute_gae(traj.rewards, traj.values, gamma, gae_lambda)
            all_obs.extend(traj.obs)
            all_actions.extend(traj.actions)
            all_masks.extend(traj.masks)
            all_logprobs.extend(traj.logprobs)
            all_advantages.extend(advantages)
            all_returns.extend(returns)

    return PPOBatch(
        obs=torch.tensor(np.stack(all_obs), dtype=torch.float32, device=device),
        actions=torch.tensor(all_actions, dtype=torch.long, device=device),
        masks=torch.tensor(np.stack(all_masks), device=device),
        old_logprobs=torch.tensor(all_logprobs, dtype=torch.float32, device=device),
        advantages=torch.tensor(all_advantages, dtype=torch.float32, device=device),
        returns=torch.tensor(all_returns, dtype=torch.float32, device=device),
    )


def ppo_update(
    policy: ChessPolicyNet,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    config: dict,
) -> dict[str, float]:
    ppo_cfg = config["ppo"]
    clip = ppo_cfg["clip"]
    epochs = ppo_cfg["epochs"]
    minibatch_size = ppo_cfg["minibatch_size"]
    entropy_coef = ppo_cfg["entropy_coef"]
    value_coef = ppo_cfg["value_coef"]
    max_grad_norm = ppo_cfg["max_grad_norm"]

    advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

    n = batch.obs.shape[0]
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

    for _ in range(epochs):
        perm = torch.randperm(n, device=batch.obs.device)
        for start in range(0, n, minibatch_size):
            idx = perm[start : start + minibatch_size]

            # Mask reapplied here, identical to how it was applied at
            # collection time -- required for a correct importance ratio.
            logits, values = policy(batch.obs[idx], mask=batch.masks[idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_logprob = dist.log_prob(batch.actions[idx])
            entropy = dist.entropy()

            ratio = torch.exp(new_logprob - batch.old_logprobs[idx])
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = torch.nn.functional.mse_loss(values, batch.returns[idx])

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (batch.old_logprobs[idx] - new_logprob).mean()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.mean().item())
            stats["approx_kl"].append(approx_kl.item())

    return {k: float(np.mean(v)) for k, v in stats.items()}
