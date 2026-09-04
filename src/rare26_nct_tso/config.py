"""Configuration for the two ResNet50 families used in the submission."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path


FAMILIES = ("linear", "eqc4")
FOLD_AUGMENTATIONS = {
    0: (),
    1: ("affine", "color"),
    2: ("blur",),
    3: ("color", "blur"),
    4: ("affine",),
}


@dataclass(frozen=True)
class TrainingConfig:
    family: str
    data_dir: Path
    split_csv: Path
    pretrained_path: Path
    output_root: Path
    fold: int | None = None
    image_size: int = 518
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "rare26"
    wandb_entity: str | None = None
    epochs_override: int | None = None

    @property
    def epochs(self) -> int:
        if self.epochs_override is not None:
            return self.epochs_override
        return 50 if self.family == "linear" else 75

    @property
    def equivariant_head(self) -> bool:
        return self.family == "eqc4"

    def augmentations(self, fold: int) -> tuple[str, ...]:
        return FOLD_AUGMENTATIONS[fold]

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.fold is not None and self.fold not in FOLD_AUGMENTATIONS:
            raise ValueError("fold must be in [0, 4]")
        if self.image_size != 518:
            raise ValueError("The submitted models were trained at image_size=518")
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError("batch size/epochs must be positive and workers non-negative")

    def serializable(self) -> dict:
        values = asdict(self)
        values.update(
            epochs=self.epochs,
            equivariant_head=self.equivariant_head,
            data_dir=str(self.data_dir),
            split_csv=str(self.split_csv),
            pretrained_path=str(self.pretrained_path),
            output_root=str(self.output_root),
        )
        return values


def parse_args(argv: list[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train the RARE26 ResNet50 linear or C4-head family"
    )
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--pretrained-path", required=True, type=Path)
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/5fold_cv.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", dest="epochs_override", type=int)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="rare26")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args(argv)
    config = TrainingConfig(**vars(args))
    config.validate()
    return config

