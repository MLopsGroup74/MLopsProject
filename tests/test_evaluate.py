# tests/test_evaluate.py
import sys
from pathlib import Path

import pytest
from PIL import Image
import pytorch_lightning as pl

# IMPORTANT: adjust these imports to your actual module paths
from src.assignment.model import ConvolutionalNetwork
from src.assignment.data import ImageFolderDataModule
from src.assignment import evaluate  # if your file is src/assignment/evaluate.py


def _make_tiny_imagefolder(root: Path) -> None:
    """Create a tiny ImageFolder dataset for testing."""
    for cls in ["Abra", "Bulbasaur"]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            Image.new("RGB", (224, 224)).save(d / f"{i}.png")


@pytest.mark.fast
def test_evaluate_smoke(monkeypatch, tmp_path, capsys) -> None:
    # Keep any outputs inside tmp_path (checkpoints/logs)
    monkeypatch.chdir(tmp_path)

    # 1) Create tiny dataset
    data_dir = tmp_path / "fake_data"
    data_dir.mkdir()
    _make_tiny_imagefolder(data_dir)

    # 2) Build datamodule and infer num_classes
    dm = ImageFolderDataModule(
        data_dir=data_dir,
        batch_size=2,
        num_workers=0,
        img_size=224,
        seed=42,
    )
    dm.setup()

    # 3) Train 1 fast step and save a checkpoint (so evaluate can load it)
    pl.seed_everything(42, workers=True)
    model = ConvolutionalNetwork(num_classes=dm.num_classes, lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        default_root_dir=str(tmp_path),
        limit_train_batches=1,
        limit_val_batches=0,
        enable_model_summary=False,
    )
    trainer.fit(model, dm)

    ckpt_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(str(ckpt_path))

    # 4) Run evaluation script entrypoint with CLI args
    args = [
        "evaluate.py",
        "--data_dir",
        str(data_dir),
        "--checkpoint",
        str(ckpt_path),
        "--batch_size",
        "2",
        "--num_workers",
        "0",
        "--img_size",
        "224",
        "--accelerator",
        "cpu",
        "--devices",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate.main()

    # 5) Assert it printed the report header
    out = capsys.readouterr().out
    assert "Classification report" in out
