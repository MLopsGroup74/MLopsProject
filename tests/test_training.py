import sys
import pytest
from PIL import Image
from pathlib import Path

from src.assignment import train


def _make_tiny_imagefolder(root: Path) -> None:
    for cls in ["Abra", "Bulbasaur"]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            img = Image.new("RGB", (224, 224))
            img.save(d / f"{i}.png")


@pytest.mark.fast
def test_train_minimal(monkeypatch, tmp_path) -> None:
    # Put ALL relative output (checkpoints/logs) inside tmp_path
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "fake_data"
    data_dir.mkdir()
    _make_tiny_imagefolder(data_dir)

    # Use real WandB, but avoid network + keep files in tmp_path
    monkeypatch.setenv("WANDB_MODE", "offline")  # use "disabled" if you want *no* wandb run dir
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))

    args = [
        "train.py",
        "--data_dir", str(data_dir),
        "--batch_size", "2",
        "--num_workers", "0",
        "--img_size", "224",
        "--max_epochs", "1",
        "--lr", "1e-3",
        "--accelerator", "cpu",
        "--devices", "1",
    ]
    monkeypatch.setattr(sys, "argv", args)

    train.main()

