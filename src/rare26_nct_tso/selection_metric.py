"""Historical metric used for checkpoint selection during training."""

from __future__ import annotations

import torch


def corrected_ppv_at_recall(
    labels: torch.Tensor,
    scores: torch.Tensor,
    prevalence: float = 1 / 101,
    min_recall: float = 0.90,
) -> tuple[float, float]:
    """Analytic prevalence-corrected PPV used by the original run.

    This is retained to reproduce checkpoint selection. It is not the full
    organizer resampling protocol; use ``challenge_style_ppv`` for reporting.
    """
    labels = labels.flatten().float()
    scores = scores.flatten()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have equal shape")
    thresholds = torch.sort(torch.unique(scores), descending=True).values
    decisions = scores.unsqueeze(0) >= thresholds.unsqueeze(1)
    positives = labels == 1
    negatives = ~positives
    tp = (decisions & positives.unsqueeze(0)).sum(dim=1).float()
    fp = (decisions & negatives.unsqueeze(0)).sum(dim=1).float()
    recall = tp / positives.sum().clamp_min(1)
    specificity = 1.0 - fp / negatives.sum().clamp_min(1)
    numerator = recall * prevalence
    denominator = numerator + (1.0 - specificity) * (1.0 - prevalence)
    ppv = torch.where(denominator > 0, numerator / denominator, 0.0)
    valid = torch.where(recall >= min_recall)[0]
    if valid.numel() == 0:
        raise ValueError("no threshold reaches the requested recall")
    best = valid[torch.argmax(ppv[valid])]
    return float(ppv[best].item()), float(thresholds[best].item())

