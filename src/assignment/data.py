"""
data.py

Lightning DataModule for an ImageFolder-style dataset (e.g. PokemonData/{class_name}/*.jpg).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl


def default_transforms(img_size: int = 224) -> transforms.Compose:
    """Transforms matching common ImageNet-pretrained conventions."""
    return transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@dataclass
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0. Got {total}.")


class ImageFolderDataModule(pl.LightningDataModule):
    """
    Lightning DataModule wrapping torchvision.datasets.ImageFolder.

    Directory structure:
      data_dir/
        class_a/
          img1.png
          ...
        class_b/
          ...
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 0,
        img_size: int = 224,
        split: SplitConfig = SplitConfig(),
        seed: int = 42,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.split = split
        self.seed = seed
        self.transform = transform or default_transforms(img_size)

        self._dataset: Optional[datasets.ImageFolder] = None
        self._train = None
        self._val = None
        self._test = None

    @property
    def class_names(self) -> Tuple[str, ...]:
        if self._dataset is None:
            # Create a lightweight dataset to read classes
            tmp = datasets.ImageFolder(root=str(self.data_dir), transform=None)
            return tuple(tmp.classes)
        return tuple(self._dataset.classes)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def setup(self, stage: Optional[str] = None) -> None:
        self.split.validate()
        self._dataset = datasets.ImageFolder(root=str(self.data_dir), transform=self.transform)

        n = len(self._dataset)
        n_train = int(self.split.train * n)
        n_val = int(self.split.val * n)
        n_test = n - n_train - n_val

        gen = torch.Generator().manual_seed(self.seed)
        self._train, self._val, self._test = random_split(self._dataset, [n_train, n_val, n_test], generator=gen)

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None, "Call setup() before requesting dataloaders."
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None, "Call setup() before requesting dataloaders."
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test is not None, "Call setup() before requesting dataloaders."
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
