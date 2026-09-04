#!/usr/bin/env python3
"""Recreate the image-level split procedure used before the submitted runs.

Use the committed CSV for exact reproduction. This script documents the
procedure and is useful when auditing a fresh copy of the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


CLASS_TARGETS = {"ndbe": 0, "neo": 1}


def collect(data_dir: Path) -> pd.DataFrame:
    rows = []
    for center in ("center_1", "center_2"):
        for class_name, target in CLASS_TARGETS.items():
            directory = data_dir / center / class_name
            for path in sorted(directory.glob("*.png")):
                rows.append({
                    "image_path": str(path.relative_to(data_dir)),
                    "sample_id": path.name,
                    "center": center,
                    "class_name": class_name,
                    "target": target,
                })
    if not rows:
        raise ValueError(f"no PNG images found under {data_dir}")
    return pd.DataFrame(rows)


def assign_folds(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    stratification = frame["center"] + "_" + frame["class_name"]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    frame["split"] = ""
    for fold, (_, validation_indices) in enumerate(splitter.split(frame, stratification)):
        frame.loc[validation_indices, "split"] = f"fold_{fold}"
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    output = assign_folds(collect(args.data_dir))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(output)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

