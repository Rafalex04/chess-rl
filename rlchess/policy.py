"""Policy network: board tensor -> masked action logits + scalar value.

Small conv/ResNet trunk feeding two heads (policy, value), per
planning/SPEC.md sec 2. Value head is the PPO critic (used later, once
ppo.py exists). Architecture size is config-driven (configs/default.yaml
`policy:` section) and deliberately small so it runs comfortably on CPU for
local dev/tests as well as on a single GPU for real training.
"""

from __future__ import annotations

import torch
from torch import nn

from rlchess.env import ACTION_SPACE_SIZE, NUM_PLANES


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set illegal-move logits to -inf. Shared by act-time inference and, later,
    PPO's logprob recomputation in the update -- one implementation, two call
    sites, so the two can't silently drift apart (planning/CLAUDE.md's "mask
    in the PPO ratio" trap).
    """
    return logits.masked_fill(~mask, float("-inf"))


class _ResidualBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return torch.relu(out)


class ChessPolicyNet(nn.Module):
    def __init__(
        self,
        num_filters: int = 64,
        num_res_blocks: int = 6,
        value_hidden_dim: int = 64,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[_ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        self.policy_head_conv = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_head_fc = nn.Linear(2 * 8 * 8, ACTION_SPACE_SIZE)

        self.value_head_conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_head_fc1 = nn.Linear(1 * 8 * 8, value_hidden_dim)
        self.value_head_fc2 = nn.Linear(value_hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(obs)
        x = self.res_blocks(x)

        policy = self.policy_head_conv(x)
        policy = policy.flatten(start_dim=1)
        logits = self.policy_head_fc(policy)
        if mask is not None:
            logits = apply_action_mask(logits, mask)

        value = self.value_head_conv(x)
        value = value.flatten(start_dim=1)
        value = torch.relu(self.value_head_fc1(value))
        value = torch.tanh(self.value_head_fc2(value))
        value = value.squeeze(-1)

        return logits, value

    @classmethod
    def from_config(cls, config: dict) -> "ChessPolicyNet":
        policy_cfg = config["policy"]
        return cls(
            num_filters=policy_cfg["num_filters"],
            num_res_blocks=policy_cfg["num_res_blocks"],
            value_hidden_dim=policy_cfg["value_hidden_dim"],
        )
