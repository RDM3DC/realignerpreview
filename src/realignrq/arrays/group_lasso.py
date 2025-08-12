"""Group lasso prior utilities (placeholder)."""

import numpy as np

def group_lasso_penalty(M, lam):
    """Compute a simple group lasso penalty."""
    return lam * np.sum(np.sqrt(np.sum(M**2, axis=0)))
