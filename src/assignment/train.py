"""
train.py

Train + evaluate the CNN on an ImageFolder dataset.

Example:
  python train.py --data_dir /path/to/PokemonData --max_epochs 20 --batch_size 32

Example for running from current best model from the root:
    uv run python -m src.assignment.train.py   --data_dir ./PokemonData   --max_epochs 20   --batch_size 32   --lr 1e-4   --ckpt_path models/model-epoch=17-val_acc=0.38.ckpt

Example for GCS bucket:
    uv run python3 -m src.assignment.train.py --data_dir gs://mlopsproject-data/PokemonData --max_epochs 1 --batch_size 2
    test

"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import pytorch_lightning as pl
from loguru import logger
#import logging_setup
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from assignment.data import ImageFolderDataModule
from assignment.model import ConvolutionalNetwork
import tempfile
import subprocess
import wandb
from dotenv import load_dotenv
import os

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
    p.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to resume training.")
    return p.parse_args()


def resolve_data_path(data_dir: str) -> Path:
    # 1. If it's a cloud URI (MacBook testing mode), download it
    if data_dir.startswith("gs://"):
        logger.info(f"Downloading data from {data_dir} to temporary directory...")
        temp_dir = Path(tempfile.mkdtemp())
        subprocess.run(["gsutil", "-m", "cp", "-r", f"{data_dir}/*", str(temp_dir)], check=True)
        return temp_dir

    # 2. If it's a local path or GCP mount (/gcs/...), just use it directly
    path = Path(data_dir)
    if not path.exists():
        # This safety check helps you debug if the mount fails
        logger.error(f"Data path not found: {data_dir}")
        raise FileNotFoundError(f"Could not find data at {data_dir}")

    return path



def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    #logging the configuration
    logger.info(f"Training pipeline started with config: {vars(args)}")

    # Initialize the WandbLogger #os.getenv("WANDB_PROJECT"), din't work
    wandb_logger = WandbLogger(
        project="mlops_assignment",
        entity=os.getenv("WANDB_ENTITY"),
        config=vars(args) # logs all your hyperparameters (lr, batch_size, etc.) to Wandb
    )

    try:
        data_path = resolve_data_path(args.data_dir)

        dm = ImageFolderDataModule(
            data_dir=data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            seed=args.seed,
        )
        dm.setup()

        # 2. UPDATED CHECKPOINT PATH
        # We check if we are on GCP (path starts with /gcs/) to save to the bucket
        # Otherwise, we save locally.
        save_path = "/gcs/mlopsproject-data/models/" if args.data_dir.startswith("/gcs/") else "models/"

        checkpoint_callback = ModelCheckpoint(
            dirpath=save_path,
            filename="model-{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            monitor="val_acc",
            mode="max",
        )

        #Data status check
        logger.info(f"DataModule initialized. Found {dm.num_classes} classes.")

        # #Define the checkpoint callback to save the best model (highest validation accuracy)
        # checkpoint_callback = ModelCheckpoint(
        #     dirpath="models/",
        #     filename="model-{epoch:02d}-{val_acc:.2f}",
        #     save_top_k=1,
        #     monitor="val_acc",
        #     mode="max",
        # )

        model = ConvolutionalNetwork(num_classes=dm.num_classes, lr=args.lr)

        #Hardware check
        logger.info(f"Running on accelerator: {args.accelerator} with {args.devices} device(s)")

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            logger=wandb_logger, #tells Lightning to send metrics to W&B
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
        )

        #Starting the actual training loop
        logger.warning(f"Starting model training for {args.max_epochs} epochs...")
        trainer.fit(model, dm, ckpt_path=args.ckpt_path)


        # Testing
        logger.info("Starting testing phase...")
        trainer.test(model, datamodule=dm)

        #Success notification
        logger.success("Training and testing completed successfully!")

    except Exception as e:
        # Error handling (Saves the error traceback to your log file)
        logger.exception(f"The program crashed due to an unexpected error: {e}")

if __name__ == "__main__":
    main()
