"""
train.py

Train + evaluate the CNN on an ImageFolder dataset.

Example:
  python train.py --data_dir /path/to/PokemonData --max_epochs 20 --batch_size 32
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
import load_from_env
from sklearn.metrics import classification_report
from pytorch_lightning.loggers import WandbLogger
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

    # Initialize the WandbLogger
    wandb_logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        config=vars(args) # This logs all your hyperparameters (lr, batch_size, etc.)
    )


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
        logger=wandb_logger, #tells Lightning to send metrics to W&B
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
