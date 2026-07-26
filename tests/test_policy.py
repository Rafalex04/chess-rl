import random

import chess
import numpy as np
import pytest
import torch
import yaml

from rlchess.env import ACTION_SPACE_SIZE, ChessEnv, NUM_PLANES, board_to_tensor, legal_action_mask
from rlchess.policy import ChessPolicyNet, apply_action_mask


@pytest.fixture
def config():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


def _random_boards(n: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    for _ in range(n):
        board = chess.Board()
        for _ in range(rng.randint(0, 30)):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        boards.append(board)
    return boards


def test_forward_shapes(config):
    net = ChessPolicyNet.from_config(config)
    net.eval()

    boards = _random_boards(4)
    obs = torch.tensor(np.stack([board_to_tensor(b) for b in boards]))
    assert obs.shape == (4, NUM_PLANES, 8, 8)

    with torch.no_grad():
        logits, value = net(obs)

    assert logits.shape == (4, ACTION_SPACE_SIZE)
    assert value.shape == (4,)


def test_masked_illegal_logits_are_neg_inf(config):
    net = ChessPolicyNet.from_config(config)
    net.eval()

    boards = _random_boards(4, seed=1)
    obs = torch.tensor(np.stack([board_to_tensor(b) for b in boards]))
    masks = torch.tensor(np.stack([legal_action_mask(b) for b in boards]))

    with torch.no_grad():
        logits, _ = net(obs, mask=masks)

    assert torch.all(torch.isneginf(logits[~masks]))
    assert torch.all(torch.isfinite(logits[masks]))


def test_masking_prevents_illegal_sampling(config):
    net = ChessPolicyNet.from_config(config)
    net.eval()

    boards = _random_boards(6, seed=2)
    obs = torch.tensor(np.stack([board_to_tensor(b) for b in boards]))
    masks_np = np.stack([legal_action_mask(b) for b in boards])
    masks = torch.tensor(masks_np)

    with torch.no_grad():
        logits, _ = net(obs, mask=masks)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        samples = dist.sample((200,))  # (200, batch)

    for step in range(samples.shape[0]):
        for b in range(samples.shape[1]):
            action = samples[step, b].item()
            assert masks_np[b, action]


def test_runs_on_cpu(config):
    net = ChessPolicyNet.from_config(config)
    net.to("cpu")
    net.eval()

    board = chess.Board()
    obs = torch.tensor(board_to_tensor(board)).unsqueeze(0).to("cpu")
    mask = torch.tensor(legal_action_mask(board)).unsqueeze(0).to("cpu")

    with torch.no_grad():
        logits, value = net(obs, mask=mask)

    assert logits.device.type == "cpu"
    assert value.device.type == "cpu"


def test_config_driven_architecture(config):
    net = ChessPolicyNet.from_config(config)
    assert len(net.res_blocks) == config["policy"]["num_res_blocks"]

    small_net = ChessPolicyNet(num_filters=8, num_res_blocks=1, value_hidden_dim=4)
    small_net.eval()
    obs = torch.zeros(2, NUM_PLANES, 8, 8)
    with torch.no_grad():
        logits, value = small_net(obs)
    assert logits.shape == (2, ACTION_SPACE_SIZE)
    assert value.shape == (2,)


def test_apply_action_mask_standalone():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    masked = apply_action_mask(logits, mask)
    assert torch.isneginf(masked[0, 1])
    assert masked[0, 0] == 1.0
    assert masked[0, 2] == 3.0
