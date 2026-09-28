
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from sklearn.model_selection import KFold
except Exception:
    KFold = None

from uresnet50_repro import AGUResNet50, UResNet50, bce_dice_loss, dice_score


class SegmentationDataset(Dataset):
    """
    Minimal dataset wrapper.
    Expects tensors shaped [C,H,W] for images and [1,H,W] for masks,
    or callables returning those tensors.
    """
    def __init__(self, images: Sequence[torch.Tensor], masks: Sequence[torch.Tensor]):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx].float()
        y = self.masks[idx].float()
        return x, y


@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 2
    lr: float = 1e-4
    num_folds: int = 5
    model_name: str = 'UResNet50'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(name: str) -> nn.Module:
    if name == 'UResNet50':
        return UResNet50()
    if name == 'AGUResNet50':
        return AGUResNet50()
    raise ValueError(f'Unknown model: {name}')


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Adam, device: str) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = bce_dice_loss(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = bce_dice_loss(pred, y)
        dsc = dice_score(pred, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_dice += dsc.item() * bs
        n += bs
    return total_loss / n, total_dice / n


def run_kfold(dataset: Dataset, cfg: TrainConfig):
    if KFold is None:
        raise ImportError('scikit-learn is required for KFold cross-validation.')

    splitter = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=42)
    results = []
    indices = list(range(len(dataset)))

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(indices), start=1):
        model = build_model(cfg.model_name).to(cfg.device)
        optimizer = Adam(model.parameters(), lr=cfg.lr)
        train_loader = DataLoader(Subset(dataset, tr_idx), batch_size=cfg.batch_size, shuffle=True)
        valid_loader = DataLoader(Subset(dataset, va_idx), batch_size=cfg.batch_size, shuffle=False)

        history = []
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)
            val_loss, val_dice = validate(model, valid_loader, cfg.device)
            history.append({
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice,
            })
        results.extend(history)
    return results
