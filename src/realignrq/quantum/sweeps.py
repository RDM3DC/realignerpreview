"""Sweep utilities for quantum-control experiments."""

import numpy as np

def ensemble_sweep(params_list, cost_fn):
    """Evaluate cost function over an ensemble of parameter sets."""
    results = []
    for params in params_list:
        results.append(cost_fn(*params))
    return np.array(results)
