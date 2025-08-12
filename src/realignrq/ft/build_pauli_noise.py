"""Convert RB-style data into Pauli+leakage noise models (placeholder)."""

def build_pauli_noise(rb_data):
    """Return a dummy noise model from RB data."""
    return {"pauli_error": float(sum(rb_data))}
