import numpy as np
from realignrq.quantum.drag2 import apply_drag2

def test_apply_drag2_shape():
    A = np.zeros((4, 16))
    dA = np.zeros_like(A)
    d2A = np.zeros_like(A)
    Delta = np.ones(2)
    iq_pairs = [(0, 1), (2, 3)]
    A2 = apply_drag2(A, dA, d2A, Delta, iq_pairs)
    assert A2.shape == A.shape
