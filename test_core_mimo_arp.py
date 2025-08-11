import numpy as np
from quantum.core_mimo_arp import mimo_arp_shaper, apply_drag

def test_shapes_and_stability():
    C, T, dt = 4, 1000, 0.01
    S = np.zeros((C, T)); S[:, 100:300] = 1.0
    M = np.eye(C) * 0.5
    alpha = np.ones(C) * 0.5
    A, dA = mimo_arp_shaper(S, M, alpha, dt)
    assert A.shape == (C, T) and dA.shape == (C, T)
    # bounded response to a step
    assert np.isfinite(A).all() and A.max() <= 1.0 + 1e-6
    # DRAG does not explode
    Delta = 2*np.pi*np.array([0.25, 0.26])
    A2 = apply_drag(A[:4,:], dA[:4,:], Delta, beta=1.0, iq_pairs=[(0,1),(2,3)])
    assert np.isfinite(A2).all()
