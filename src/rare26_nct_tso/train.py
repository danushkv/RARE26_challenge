"""Train the five linear or five C4-head ResNet50 models."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import TrainingConfig, parse_args
from .data import create_loaders, fold_frames, load_split
from .models import SubmissionResNet50
from .selection_metric import corrected_ppv_at_recall


LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inverse_frequency_weights(labels, device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    return torch.tensor(1.0 / np.clip(counts, 1, None), dtype=torch.float32, device=device)


@torch.inference_mode()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    losses, labels, logits, paths = [], [], [], []
    for images, batch_labels, batch_paths in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        batch_labels = batch_labels.to(device, non_blocking=device.type == "cuda")
        outputs = model(images)
        losses.append(float(criterion(outputs, batch_labels).item()))
        labels.append(batch_labels)
        logits.append(outputs)
        paths.extend(batch_paths)
    labels_tensor = torch.cat(labels)
    logits_tensor = torch.cat(logits)
    probabilities = functional.softmax(logits_tensor, dim=1)[:, 1]
    ppv, threshold = corrected_ppv_at_recall(labels_tensor, probabilities)
    predictions = torch.argmax(logits_tensor, dim=1)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float((predictions == labels_tensor).float().mean().item()),
        "selection_ppv": ppv,
        "selection_threshold": threshold,
        "labels": labels_tensor.cpu().numpy(),
        "logits": logits_tensor.cpu().numpy(),
        "paths": paths,
    }


def initialize_wandb(config: TrainingConfig, fold: int):
    if not config.use_wandb:
        return None
    try:
        import wandb
    except ImportError:
        LOGGER.warning("wandb is unavailable; continuing without remote logging")
        return None
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        group=f"resnet50_{config.family}",
        name=f"resnet50_{config.family}_fold{fold}",
        config={**config.serializable(), "fold": fold},
        reinit=True,
    )


def save_run_config(config: TrainingConfig, fold: int, output_dir: Path, device) -> None:
    metadata = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "config": {
            **config.serializable(),
            "fold": fold,
            "augmentations": list(config.augmentations(fold)),
            "use_class_weights": True,
            "use_weighted_sampling": True,
            "eq_group_order": 4 if config.equivariant_head else None,
            "eq_reflections": False if config.equivariant_head else None,
            "eq_hidden_fields": [64, 64] if config.equivariant_head else None,
        },
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": sys.version,
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))


def train_fold(config: TrainingConfig, frame: pd.DataFrame, fold: int, device) -> None:
    fold_seed = config.seed + fold
    seed_everything(fold_seed)
    train_frame, val_frame = fold_frames(frame, fold)
    train_loader, val_loader = create_loaders(train_frame, val_frame, config, fold)
    output_dir = config.output_root / config.family / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(config, fold, output_dir, device)

    model = SubmissionResNet50(config.family, config.pretrained_path).to(device)
    class_weights = inverse_frequency_weights(train_frame["target"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
    )
    wandb_run = initialize_wandb(config, fold)
    best_ppv = -1.0
    checkpoint_path = output_dir / "best.pth"

    LOGGER.info(
        "%s fold %d: %d train / %d validation; epochs=%d; augmentations=%s",
        config.family, fold, len(train_frame), len(val_frame), config.epochs,
        config.augmentations(fold) or "base only",
    )
    try:
        for epoch in range(config.epochs):
            model.train()
            losses, correct, samples = [], 0, 0
            for images, labels, _ in train_loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                labels = labels.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
                samples += labels.numel()
            scheduler.step()
            validation = validate(model, val_loader, criterion, device)
            metrics = {
                "epoch": epoch + 1,
                "train/loss": float(np.mean(losses)),
                "train/accuracy": correct / samples,
                "val/loss": validation["loss"],
                "val/accuracy": validation["accuracy"],
                "val/selection_ppv": validation["selection_ppv"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            LOGGER.info("fold=%d %s", fold, metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=epoch + 1)
            if validation["selection_ppv"] > best_ppv:
                best_ppv = validation["selection_ppv"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "fold": fold,
                    "best_ppv": best_ppv,
                    "config": config.serializable(),
                }, checkpoint_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    validation = validate(model, val_loader, criterion, device)
    records = []
    for path, label, logits in zip(
        validation["paths"], validation["labels"], validation["logits"]
    ):
        records.append({
            "image_path": path,
            "sample_id": Path(path).name,
            "target": int(label),
            "logits_0": float(logits[0]),
            "logits_1": float(logits[1]),
            "fold": fold,
            "split_type": "val",
        })
    pd.DataFrame(records).to_csv(output_dir / "oof_val_predictions.csv", index=False)
    LOGGER.info(
        "%s fold %d complete: selected epoch=%d, PPV=%.6f",
        config.family, fold, checkpoint["epoch"] + 1, validation["selection_ppv"],
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    config = parse_args(argv)
    if not config.data_dir.is_dir():
        raise FileNotFoundError(config.data_dir)
    if not config.pretrained_path.is_file():
        raise FileNotFoundError(config.pretrained_path)
    frame = load_split(config.split_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = [config.fold] if config.fold is not None else list(range(5))
    for fold in folds:
        train_fold(config, frame, fold, device)


if __name__ == "__main__":
    main()

