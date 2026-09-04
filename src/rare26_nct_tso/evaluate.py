"""Validate OOF artifacts and compare the two submitted model families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .challenge_metric import challenge_style_ppv, percentile_ranks


REQUIRED_SPLIT_COLUMNS = frozenset(("image_path", "sample_id", "target", "split"))


def load_canonical_split(path: Path) -> pd.DataFrame:
    """Load the canonical split without importing the GPU training stack."""
    frame = pd.read_csv(path)
    missing = REQUIRED_SPLIT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")
    if set(frame["target"].unique()) != {0, 1}:
        raise ValueError("target must contain both binary classes")
    if set(frame["split"].unique()) != {f"fold_{fold}" for fold in range(5)}:
        raise ValueError("split must contain exactly fold_0 through fold_4")
    return frame


def prediction_path(root: Path, family: str, fold: int) -> Path:
    names = (family, "eqC4") if family == "eqc4" else (family,)
    candidates = []
    for name in names:
        fold_dir = root / name / f"fold_{fold}"
        candidates.extend((
            fold_dir / "oof_val_predictions.csv",
            fold_dir / "resnet50_5fold_cv" / "oof_val_predictions.csv",
        ))
    if family == "linear":
        historical = root / f"fold_{fold}" / "resnet50_5fold_cv"
        candidates.append(historical / "oof_val_predictions.csv")
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected one prediction file for {family}/fold_{fold}; found {existing}"
        )
    return existing[0]


def load_family(root: Path, family: str, canonical: pd.DataFrame) -> np.ndarray:
    score_by_id = {}
    for fold in range(5):
        path = prediction_path(root, family, fold)
        predictions = pd.read_csv(path)
        required = {"sample_id", "target", "logits_0", "logits_1", "fold"}
        if not required.issubset(predictions.columns):
            raise ValueError(f"{path} is missing {sorted(required - set(predictions.columns))}")
        expected = canonical.loc[canonical["split"] == f"fold_{fold}"]
        if set(predictions["sample_id"]) != set(expected["sample_id"]):
            raise ValueError(f"{path} does not match the canonical fold")
        target_by_id = expected.set_index("sample_id")["target"].to_dict()
        for row in predictions.itertuples(index=False):
            if int(row.target) != int(target_by_id[row.sample_id]):
                raise ValueError(f"target mismatch for {row.sample_id}")
            if row.sample_id in score_by_id:
                raise ValueError(f"duplicate OOF prediction for {row.sample_id}")
            score_by_id[row.sample_id] = float(row.logits_1) - float(row.logits_0)
    if set(score_by_id) != set(canonical["sample_id"]):
        raise ValueError(f"{family} does not cover every canonical sample")
    return canonical["sample_id"].map(score_by_id).to_numpy(dtype=np.float64)


def fold_normalize(scores: np.ndarray, folds: np.ndarray) -> np.ndarray:
    normalized = np.empty_like(scores)
    for fold in range(5):
        mask = folds == fold
        normalized[mask] = percentile_ranks(scores[mask])
    return normalized


def summarize(labels, scores, iterations, seed) -> dict:
    return {
        "challenge_style_ppv_at_90_recall": challenge_style_ppv(
            labels, scores, iterations=iterations, seed=seed
        ),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision_on_enriched_oof_set": float(
            average_precision_score(labels, scores)
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("outputs"))
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/5fold_cv.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    canonical = load_canonical_split(args.split_csv)
    labels = canonical["target"].to_numpy(dtype=np.int8)
    folds = canonical["split"].str.removeprefix("fold_").astype(int).to_numpy()
    linear_raw = load_family(args.results_root, "linear", canonical)
    eqc4_raw = load_family(args.results_root, "eqc4", canonical)
    linear_rank = fold_normalize(linear_raw, folds)
    eqc4_rank = fold_normalize(eqc4_raw, folds)
    ensemble = (linear_rank + eqc4_rank) / 2.0

    report = {
        "protocol": {
            "positive_class": 1,
            "score": "logits_1 - logits_0",
            "resampling_iterations": args.iterations,
            "negative_to_positive_ratio": "100:1",
            "warning": (
                "This follows the public challenge description. Internal OOF "
                "scores are not estimates of private multicenter performance."
            ),
        },
        "linear_fold_rank": summarize(labels, linear_rank, args.iterations, args.seed),
        "eqc4_fold_rank": summarize(labels, eqc4_rank, args.iterations, args.seed),
        "linear_plus_eqc4_equal_rank_ensemble": summarize(
            labels, ensemble, args.iterations, args.seed
        ),
        "diagnostic_raw_pooled": {
            "linear": summarize(labels, linear_raw, args.iterations, args.seed),
            "eqc4": summarize(labels, eqc4_raw, args.iterations, args.seed),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({
        "sample_id": canonical["sample_id"],
        "target": labels,
        "fold": folds,
        "linear_logit_difference": linear_raw,
        "eqc4_logit_difference": eqc4_raw,
        "linear_fold_percentile": linear_rank,
        "eqc4_fold_percentile": eqc4_rank,
        "ensemble_score": ensemble,
    }).to_csv(args.output_dir / "oof_scores.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
