"""CSV-backed dataset and the transforms used for the submitted models."""

from __future__ import annotations

import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OPTIONAL_AUGMENTATIONS = frozenset(("affine", "elastic", "color", "blur"))
REQUIRED_COLUMNS = frozenset(("image_path", "sample_id", "target", "split"))
cv2.setNumThreads(1)


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")
    if set(frame["target"].unique()) != {0, 1}:
        raise ValueError("target must contain both binary classes")
    expected_folds = {f"fold_{fold}" for fold in range(5)}
    if set(frame["split"].unique()) != expected_folds:
        raise ValueError("split must contain exactly fold_0 through fold_4")
    return frame


def fold_frames(frame: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_name = f"fold_{fold}"
    validation = frame.loc[frame["split"] == fold_name].reset_index(drop=True)
    training = frame.loc[frame["split"] != fold_name].reset_index(drop=True)
    return training, validation


class RareDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.data_dir / row["image_path"]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
        tensor = self.transform(image=array)["image"]
        return tensor, int(row["target"]), str(row["image_path"])


def build_transform(training: bool, image_size: int, augmentations=()):
    augmentations = tuple(augmentations)
    unknown = set(augmentations) - OPTIONAL_AUGMENTATIONS
    if unknown:
        raise ValueError(f"unknown augmentations: {sorted(unknown)}")

    if not training:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    transforms = [
        A.RandomResizedCrop(
            size=(image_size, image_size), scale=(0.8, 1.0),
            ratio=(0.75, 1.33), p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.8),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8
        ),
    ]
    selected = set(augmentations)
    if "affine" in selected:
        transforms.append(A.Affine(
            scale=(0.9, 1.1), translate_percent=(-0.1, 0.1),
            rotate=(-15, 15), shear=(-5, 5),
            interpolation=cv2.INTER_LINEAR, p=0.5,
        ))
    if "elastic" in selected:
        transforms.append(A.ElasticTransform(
            alpha=120.0, sigma=6.0, interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5,
        ))
    if "color" in selected:
        transforms.extend([
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30,
                val_shift_limit=20, p=0.5,
            ),
            A.RGBShift(
                r_shift_limit=20, g_shift_limit=20,
                b_shift_limit=20, p=0.25,
            ),
        ])
    if "blur" in selected:
        transforms.extend([
            A.MotionBlur(blur_limit=7, p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.21),
        ])
    transforms.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return A.Compose(transforms)


def weighted_sampler(labels) -> WeightedRandomSampler:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    weights = (1.0 / np.clip(counts, 1, None))[labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights), replacement=True,
    )


def create_loaders(train_frame, val_frame, config, fold: int):
    train_dataset = RareDataset(
        train_frame, config.data_dir,
        build_transform(True, config.image_size, config.augmentations(fold)),
    )
    val_dataset = RareDataset(
        val_frame, config.data_dir,
        build_transform(False, config.image_size),
    )
    sampler = weighted_sampler(train_frame["target"].to_numpy())
    return (
        DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=sampler,
            num_workers=config.num_workers, pin_memory=True,
        ),
        DataLoader(
            val_dataset, batch_size=config.batch_size * 2, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
        ),
    )

