import numpy as np

def phase_basis_shaping(A, phases):
    """Apply phase-basis shaping with simple constraints."""
    phases = np.asarray(phases)
    return A * np.exp(1j * phases[:, None])
