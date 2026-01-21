"""
train.py

Train + evaluate the CNN on an ImageFolder dataset.

Example:
  python train.py --data_dir /path/to/PokemonData --max_epochs 20 --batch_size 32

Example for quick test run from the root:
  uv run python -m src.assignment.train --data_dir ./PokemonData --max_epochs 1 --batch_size 2

Example for GCS bucket:
    uv run python3 -m src.assignment.train-kopi --data_dir gs://mlopsproject-data/PokemonData --max_epochs 1 --batch_size 2

"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
import load_from_env
from loguru import logger
import logging_setup
from sklearn.metrics import classification_report
from pytorch_lightning.loggers import WandbLogger
from src.assignment.data import ImageFolderDataModule
from src.assignment.model import ConvolutionalNetwork
import tempfile
import subprocess



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


def resolve_data_path(data_dir: str) -> Path:
    """
    If the path starts with 'gs://', download the contents locally.
    Otherwise, return as a Path object.
    """
    if data_dir.startswith("gs://"):
        temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Downloading bucket {data_dir} to temporary folder {temp_dir} ...")
        subprocess.run(["gsutil", "-m", "cp", "-r", f"{data_dir}/*", str(temp_dir)], check=True)
        return temp_dir
    else:
        return Path(data_dir)



def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # Logging the configuration
    logger.info(f"Training pipeline started with config: {vars(args)}")

    # Initialize the WandbLogger
    wandb_logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        config=vars(args)  # logs all hyperparameters (lr, batch_size, etc.) to W&B
    )

    try:
        # Resolve data path (local or GCS bucket)
        data_path = resolve_data_path(args.data_dir)

        # Data Setup
        dm = ImageFolderDataModule(
            data_dir=data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            seed=args.seed,
        )
        dm.setup()

        # Data status check
        logger.info(f"DataModule initialized. Found {dm.num_classes} classes.")

        model = ConvolutionalNetwork(num_classes=dm.num_classes, lr=args.lr)

        # Hardware check
        logger.info(f"Running on accelerator: {args.accelerator} with {args.devices} device(s)")

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            logger=wandb_logger,
            log_every_n_steps=10,
        )

        # Starting the actual training loop
        logger.warning(f"Starting model training for {args.max_epochs} epochs...")
        trainer.fit(model, dm)

        # Testing
        logger.info("Starting testing phase...")
        trainer.test(model, datamodule=dm)

        # Success notification
        logger.success("Training and testing completed successfully!")

    except Exception as e:
        # Error handling
        logger.exception(f"The program crashed due to an unexpected error: {e}")


if __name__ == "__main__":
    main()
