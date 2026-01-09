"""
model.py

A simple CNN classifier implemented as a PyTorch LightningModule.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ConvolutionalNetwork(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Input expected: (B, 3, 224, 224)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1)

        # After conv/pool:
        # 224 -> conv3 => 222 -> pool2 => 111
        # 111 -> conv3 => 109 -> pool2 => 54
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, num_classes)

        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = torch.flatten(x, 1)  # (B, 16*54*54)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
