
import torch

from uresnet50_repro import AGUResNet50, UResNet50, AttentionGate, bce_dice_loss


def test_attention_gate_shape():
    gate = AttentionGate(gate_channels=1024, skip_channels=512)
    g = torch.randn(2, 1024, 15, 15)
    x = torch.randn(2, 512, 30, 30)
    y = gate(g, x)
    assert y.shape == x.shape


def test_uresnet50_output_shape():
    model = UResNet50()
    x = torch.randn(2, 1, 240, 240)
    y = model(x)
    assert y.shape == (2, 1, 240, 240)


def test_aguresnet50_output_shape():
    model = AGUResNet50()
    x = torch.randn(2, 1, 240, 240)
    y = model(x)
    assert y.shape == (2, 1, 240, 240)


def test_loss_finite():
    model = AGUResNet50()
    x = torch.randn(2, 1, 240, 240)
    y = model(x)
    target = torch.rand_like(y)
    loss = bce_dice_loss(y, target)
    assert torch.isfinite(loss)
