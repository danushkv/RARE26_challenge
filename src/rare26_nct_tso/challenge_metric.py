"""NumPy implementation of the publicly described RARE26 metric protocol."""

from __future__ import annotations

import math

import numpy as np


def ppv_at_min_recall(labels, scores, min_recall: float = 0.90) -> float:
    """PPV at the highest score threshold attaining at least ``min_recall``.

    Scores are treated as positive when ``score >= threshold``. Tied scores at
    the selected threshold are all included.
    """
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("labels and scores must be non-empty and have equal shape")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("labels must be binary and scores must be finite")
    positive_scores = scores[labels == 1]
    if positive_scores.size == 0 or np.all(labels == 1):
        raise ValueError("both classes are required")
    required_tp = max(1, math.ceil(min_recall * positive_scores.size))
    threshold = np.sort(positive_scores)[::-1][required_tp - 1]
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & (labels == 1)))
    fp = int(np.count_nonzero(predicted & (labels == 0)))
    return tp / (tp + fp)


def challenge_style_ppv(
    labels,
    scores,
    iterations: int = 1000,
    negatives_per_positive: int = 100,
    min_recall: float = 0.90,
    seed: int = 42,
) -> dict:
    """Estimate RARE26 PPV@90Recall using its public resampling description.

    Every iteration retains all negatives and samples positives with
    replacement to approximate a 100:1 negative-to-positive ratio. The point
    estimate is the median; the interval is the 2.5/97.5 percentile range.
    """
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have equal shape")
    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)
    if negative_indices.size == 0 or positive_indices.size == 0:
        raise ValueError("both classes are required")
    if iterations < 1 or negatives_per_positive < 1:
        raise ValueError("iterations and negatives_per_positive must be positive")

    sampled_positive_count = max(
        1, int(round(negative_indices.size / negatives_per_positive))
    )
    negative_scores = scores[negative_indices]
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)

    for iteration in range(iterations):
        sampled = rng.choice(
            positive_indices, size=sampled_positive_count, replace=True
        )
        iteration_labels = np.concatenate(
            (np.zeros(negative_indices.size, dtype=np.int8),
             np.ones(sampled_positive_count, dtype=np.int8))
        )
        iteration_scores = np.concatenate((negative_scores, scores[sampled]))
        values[iteration] = ppv_at_min_recall(
            iteration_labels, iteration_scores, min_recall=min_recall
        )

    lower, upper = np.quantile(values, (0.025, 0.975))
    return {
        "median": float(np.median(values)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "mean": float(np.mean(values)),
        "iterations": iterations,
        "negatives": int(negative_indices.size),
        "available_positives": int(positive_indices.size),
        "sampled_positives_per_iteration": sampled_positive_count,
        "seed": seed,
    }


def percentile_ranks(values) -> np.ndarray:
    """Map scores to average empirical percentile ranks, preserving ties."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    left = np.searchsorted(sorted_values, values, side="left")
    right = np.searchsorted(sorted_values, values, side="right")
    return (left + right) / (2.0 * values.size)
