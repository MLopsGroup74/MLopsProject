import sys
from unittest.mock import patch

import pytest

from PIL import Image

from src.assignment import train
from pathlib import Path

from tests import _PATH_DATA 

def _make_tiny_imagefolder(root: Path):
    for cls in ["Abra", "Bulbasaur"]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        # 2 tiny images per class
        for i in range(2):
            img = Image.new("RGB", (224, 224))
            img.save(d / f"{i}.png")



@pytest.mark.fast
def test_train_minimal(monkeypatch, tmp_path):
    # --- create tiny fake dataset ---
    data_dir = tmp_path / "fake_data" 
    data_dir.mkdir()
    _make_tiny_imagefolder(data_dir)

    # --- disable wandb completely (prevents any network / init) ---
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


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

    # Patch parse_args to return parsed args without touching sys.argv
    monkeypatch.setattr(sys, "argv", args)

    parsed = train.parse_args()

    # Patch WandbLogger where it is USED (inside src.assignment.train)
    with patch.object(train, "WandbLogger", autospec=True) as _mock_wandb:
        with patch.object(train, "parse_args", return_value=parsed):
            train.main()


