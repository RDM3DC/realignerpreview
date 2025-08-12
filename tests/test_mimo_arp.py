import numpy as np
from realignrq.quantum.mimo_arp import mimo_arp_shaper

def test_shapes():
    S = np.zeros((4, 256))
    M = np.zeros((4, 4))
    A, dA = mimo_arp_shaper(S, M, alpha=0.1, dt=1e-3)
    assert A.shape == S.shape and dA.shape == S.shape
