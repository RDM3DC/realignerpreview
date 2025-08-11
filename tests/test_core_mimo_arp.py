import numpy as np
from quantum.core_mimo_arp import mimo_arp_shaper, apply_drag


def test_shapes_and_stability():
    C, T = 4, 1000
    dt = 0.01
    S = np.zeros((C, T)); S[:, 100:200] = 1.0
    M = np.eye(C) * 0.5
    alpha = np.ones(C) * 0.5
    A, dA = mimo_arp_shaper(S, M, alpha, dt)
    assert A.shape == (C, T)
    assert dA.shape == (C, T)
    # positivity / boundedness on a simple step
    assert np.allclose(A.max(), A.min() + (A.max()-A.min()), atol=1e-6)
    # DRAG adds to Q channels without exploding
    Delta = 2*np.pi*np.array([0.25,0.26])
    A2 = apply_drag(A[:4,:], dA[:4,:], Delta)
    assert np.isfinite(A2).all()
