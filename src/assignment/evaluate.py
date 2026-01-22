"""
eval.py

Evaluate a trained CNN on an ImageFolder dataset.

Example:
  python eval.py --data_dir /path/to/PokemonData --checkpoint /path/to/checkpoint.ckpt --batch_size 32
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
    """Parse command line arguments for evaluation.

    Returns:
        Parsed command line arguments.
    """
    p = argparse.ArgumentParser(description="Evaluate a trained CNN model.")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder for ImageFolder dataset.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt).",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--accelerator", type=str, default="auto", help="auto/cpu/gpu/mps")
    p.add_argument("--devices", type=int, default=1)
    return p.parse_args()


def main() -> None:
    """Main evaluation pipeline entry point."""
    args = parse_args()
    pl.seed_everything(42, workers=True)

    dm = ImageFolderDataModule(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=42,
    )
    dm.setup()

    print(f"\n🔍 Loading model from checkpoint: {args.checkpoint}")
    model = ConvolutionalNetwork.load_from_checkpoint(
        args.checkpoint,
        num_classes=dm.num_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.accelerator != "cpu" else "cpu")
    model = model.to(device)
    model.eval()

    print("\nEvaluating on test set...")
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dm.test_dataloader():
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(yb.cpu().tolist())

    print(" Classification report (test):")
    labels = np.arange(len(dm.class_names))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(dm.class_names),
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
