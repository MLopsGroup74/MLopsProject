import shutil
from pathlib import Path

import torch
from PIL import Image

from src.assignment.data import ImageFolderDataModule
from tests import _PATH_DATA 


def _make_tiny_imagefolder(root: Path, n_per_class: int = 10):
    # 2 classes, n images each
    for cls in ["Abra", "Bulbasaur"]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = Image.new("RGB", (32, 32))
            img.save(d / f"{i}.png")


def test_imagefolder_datamodule():
    # Put test data in a dedicated subdir so we don't clash with real data
    data_dir = Path(_PATH_DATA)

    # Ensure clean slate
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    _make_tiny_imagefolder(data_dir, n_per_class=10)

    dm = ImageFolderDataModule(
        data_dir=str(data_dir),
        batch_size=2,
        num_workers=0,
        img_size=32,
        seed=42,
    )
    dm.setup()

    assert dm.num_classes == 2
    assert dm._train is not None and len(dm._train) > 0
    assert dm._val is not None and len(dm._val) > 0
    assert dm._test is not None and len(dm._test) > 0

    images, labels = next(iter(dm.train_dataloader()))
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.ndim == 4
    assert labels.ndim == 1
    assert images.shape[1] == 3
    assert images.shape[2] == 32 and images.shape[3] == 32
    assert labels.min().item() >= 0
    assert labels.max().item() < dm.num_classes
