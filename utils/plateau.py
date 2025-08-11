"""Plateau detection utilities."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def plateau_detected(
    val_history: Sequence[float],
    grad_var_history: Sequence[float],
    window: int = 5,
    slope_thresh: float = -1e-3,
) -> bool:
    """Return ``True`` if the training appears to have plateaued.

    A plateau is triggered when the median relative slope of the validation
    loss over ``window`` epochs is greater than ``slope_thresh`` and the
    gradient variance has increased over the same window.

    Args:
        val_history: Sequence of validation losses.
        grad_var_history: Sequence of gradient variance estimates.
        window: Number of recent points to consider.
        slope_thresh: Threshold on the relative slope of the validation loss.

    Returns:
        bool: ``True`` if a plateau is detected.
    """

    if len(val_history) < window + 1 or len(grad_var_history) < window + 1:
        return False

    vals = np.array(val_history[-window:])
    diffs = np.diff(vals)
    # Relative slope as median percentage change per epoch
    rel_slope = np.median(diffs / np.maximum(np.abs(vals[:-1]), 1e-8))

    grad_recent = grad_var_history[-window:]
    grad_increase = grad_recent[-1] > grad_recent[0]

    return rel_slope > slope_thresh and grad_increase
