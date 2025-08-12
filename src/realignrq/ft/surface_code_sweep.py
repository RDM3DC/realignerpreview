"""Surface code sweep utilities using Stim (placeholder)."""

import numpy as np

def surface_code_threshold(noise_level, distances):
    """Compute a fake logical error rate curve."""
    return {d: np.exp(-d) * noise_level for d in distances}
