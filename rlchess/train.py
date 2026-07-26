"""Self-play + PPO training loop, config-driven end to end.

Logging is optional (dependency-injected via `logger`) so this loop can be
exercised on a tiny run before `rlchess/logging.py` exists, and wired to a
real `RunLogger` afterwards without changing this module.
"""

from __future__ import annotations

import torch

from rlchess.policy import ChessPolicyNet
from rlchess.ppo import build_ppo_batch, collect_self_play_games, ppo_update


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(config: dict, logger=None, verbose: bool = False) -> ChessPolicyNet:
    device = _resolve_device(config["train"]["device"])
    policy = ChessPolicyNet.from_config(config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["ppo"]["lr"])

    for step in range(config["train"]["num_updates"]):
        games = collect_self_play_games(
            config, policy, config["ppo"]["games_per_batch"], device
        )
        batch = build_ppo_batch(
            games, config["ppo"]["gamma"], config["ppo"]["gae_lambda"], device
        )
        stats = ppo_update(policy, optimizer, batch, config)

        if logger is not None:
            logger.log_batch(step, games, stats)
            if step % config["logging"]["checkpoint_interval"] == 0:
                logger.save_checkpoint(step, policy)
                logger.save_games(
                    step, games[: config["logging"]["sample_games_per_checkpoint"]]
                )

        if verbose:
            print(f"step {step}: {stats}")

    return policy


def _main() -> None:
    import argparse

    import yaml

    from rlchess.logging import RunLogger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger = RunLogger(config)
    print(f"Run dir: {logger.run_dir}")
    train(config, logger=logger, verbose=True)
    print(f"Done. See {logger.run_dir}")


if __name__ == "__main__":
    _main()
