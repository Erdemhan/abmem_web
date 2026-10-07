"""Tests of the shared actor and critic networks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from abmem.services.algorithms.device import get_device
from abmem.services.algorithms.networks import (
    FINAL_LAYER_INIT,
    DeterministicActor,
    QCritic,
    RecurrentDeterministicActor,
    RecurrentQCritic,
    hard_update,
    mlp,
    soft_update,
)


def test_get_device_matches_cuda_availability() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device().type == expected


def test_mlp_structure() -> None:
    net = mlp(3, 2, (8, 4))
    linear = [m for m in net if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear] == [(3, 8), (8, 4), (4, 2)]
    assert isinstance(net[-1], nn.Linear)
    assert sum(isinstance(m, nn.ReLU) for m in net) == 2
    assert isinstance(mlp(3, 2, ()), nn.Sequential)
    with pytest.raises(ValueError):
        mlp(3, 2, (0,))


@pytest.mark.parametrize("action_dim", [1, 5])
def test_actor_tanh_range_and_shape(action_dim: int) -> None:
    actor = DeterministicActor(3, action_dim, (16, 16))
    # Push the pre-activation far out to exercise saturation.
    with torch.no_grad():
        actor.net[-1].weight.fill_(10.0)
    out = actor(torch.randn(64, 3) * 100)
    assert out.shape == (64, action_dim)
    assert torch.all(out <= 1.0) and torch.all(out >= -1.0)


def test_final_layer_init_range() -> None:
    nets = [
        DeterministicActor(3, 2, (32,)).net[-1],
        QCritic(3, 2, (32,)).net[-1],
        RecurrentDeterministicActor(3, 2, 8, (16,)).head[-1],
        RecurrentQCritic(3, 2, 8, (16,)).head[-1],
    ]
    for layer in nets:
        assert layer.weight.abs().max().item() <= FINAL_LAYER_INIT
        assert layer.bias.abs().max().item() <= FINAL_LAYER_INIT
        assert layer.weight.abs().max().item() > 0.0


def test_actor_gradient_non_zero() -> None:
    actor = DeterministicActor(3, 2, (16,))
    critic = QCritic(3, 2, (16,))
    obs = torch.randn(32, 3)
    loss = -critic(obs, actor(obs)).mean()
    loss.backward()
    grads = [p.grad for p in actor.parameters()]
    assert all(g is not None for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0.0


def test_critic_output_shape() -> None:
    critic = QCritic(3, 5, (16,))
    assert critic(torch.randn(7, 3), torch.rand(7, 5) * 2 - 1).shape == (7, 1)


def test_forward_deterministic_without_noise() -> None:
    actor = DeterministicActor(3, 2, (16,))
    obs = torch.randn(4, 3)
    torch.testing.assert_close(actor(obs), actor(obs))
    ractor = RecurrentDeterministicActor(3, 2, 8, (16,))
    seq = torch.randn(4, 5, 3)
    torch.testing.assert_close(ractor(seq)[0], ractor(seq)[0])


def test_recurrent_actor_shapes_and_zero_hidden() -> None:
    actor = RecurrentDeterministicActor(3, 2, 8, (16,))
    seq = torch.randn(4, 6, 3)
    actions, (h, c) = actor(seq)
    assert actions.shape == (4, 6, 2)
    assert h.shape == (1, 4, 8) and c.shape == (1, 4, 8)
    assert torch.all(actions.abs() <= 1.0)
    zeros = (torch.zeros(1, 4, 8), torch.zeros(1, 4, 8))
    torch.testing.assert_close(actor(seq, zeros)[0], actions)
    # Carrying the state over two halves equals one pass over the whole sequence.
    first, state = actor(seq[:, :3])
    second, _ = actor(seq[:, 3:], state)
    torch.testing.assert_close(torch.cat([first, second], dim=1), actions)


def test_recurrent_critic_uses_per_step_actions() -> None:
    critic = RecurrentQCritic(3, 2, 8, (16,))
    obs = torch.randn(2, 6, 3)
    actions = torch.rand(2, 6, 2) * 2 - 1
    q = critic(obs, actions)
    assert q.shape == (2, 6, 1)
    k = 3
    changed = actions.clone()
    changed[:, k] = -changed[:, k]
    q_changed = critic(obs, changed)
    torch.testing.assert_close(q_changed[:, :k], q[:, :k])
    assert not torch.allclose(q_changed[:, k:], q[:, k:])


def test_soft_update() -> None:
    target = QCritic(3, 2, (8,))
    source = QCritic(3, 2, (8,))
    t0 = [p.detach().clone() for p in target.parameters()]
    s0 = [p.detach().clone() for p in source.parameters()]
    soft_update(target, source, 0.1)
    for t, a, b in zip(target.parameters(), t0, s0, strict=True):
        torch.testing.assert_close(t, 0.9 * a + 0.1 * b)
    soft_update(target, source, 1.0)
    for t, b in zip(target.parameters(), s0, strict=True):
        torch.testing.assert_close(t, b)


@pytest.mark.parametrize("tau", [0.0, -0.1, 1.5])
def test_soft_update_invalid_tau(tau: float) -> None:
    with pytest.raises(ValueError):
        soft_update(QCritic(3, 2, (8,)), QCritic(3, 2, (8,)), tau)


def test_hard_update_and_structure_mismatch() -> None:
    target = DeterministicActor(3, 2, (8,))
    source = DeterministicActor(3, 2, (8,))
    hard_update(target, source)
    for t, s in zip(target.parameters(), source.parameters(), strict=True):
        assert torch.equal(t, s)
    with pytest.raises(ValueError):
        hard_update(target, DeterministicActor(3, 2, (16,)))
    with pytest.raises(ValueError):
        soft_update(target, DeterministicActor(3, 2, (16,)), 0.5)
