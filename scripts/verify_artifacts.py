#!/usr/bin/env python3
"""Verify the immutable split and GastroNet checkpoint used in training."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SPLIT = "dc07de48a0cb1c94a65069a75bff7903be262d04619b78f48c1eb03903e4e2c4"
EXPECTED_PRETRAINED = "5688929fea4437031604001495fb77fb18cdc2ff92ae24120f93aeceaf5aa16d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained_checkpoint", type=Path)
    parser.add_argument(
        "--split", type=Path, default=Path("data/splits/5fold_cv.csv")
    )
    args = parser.parse_args()
    observed = {
        args.split: (sha256(args.split), EXPECTED_SPLIT),
        args.pretrained_checkpoint: (
            sha256(args.pretrained_checkpoint), EXPECTED_PRETRAINED
        ),
    }
    failed = False
    for path, (actual, expected) in observed.items():
        state = "OK" if actual == expected else "MISMATCH"
        print(f"{state:8} {path}\n         {actual}")
        failed |= actual != expected
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

