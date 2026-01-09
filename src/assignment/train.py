"""
train.py

Train + evaluate the CNN on an ImageFolder dataset.

Example:
  python train.py --data_dir /path/to/PokemonData --max_epochs 20 --batch_size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.metrics import classification_report

from data import ImageFolderDataModule
from model import ConvolutionalNetwork


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Root folder for ImageFolder dataset.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--accelerator", type=str, default="auto", help="auto/cpu/gpu/mps")
    p.add_argument("--devices", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    dm = ImageFolderDataModule(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
    )
    dm.setup()
    model = ConvolutionalNetwork(num_classes=dm.num_classes, lr=args.lr)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

    # Classification report on the test set
    model.eval()
    device = model.device
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dm.test_dataloader():
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(yb.cpu().tolist())

    print("\nClassification report (test):")
    labels = np.arange(len(dm.class_names))  # all 150 classes
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(dm.class_names),
        digits=4,
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
