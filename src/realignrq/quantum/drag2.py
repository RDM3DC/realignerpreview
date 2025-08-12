import numpy as np

def apply_drag2(A, dA_dt, d2A_dt2, Delta, iq_pairs, beta1=1.0, beta2=0.2):
    """Apply second-order DRAG corrections to I/Q channels."""
    A2 = A.copy()
    for qi, (i_ch, q_ch) in enumerate(iq_pairs):
        den = max(1e-9, float(Delta[qi]))
        A2[q_ch] += -beta1 * dA_dt[i_ch] / den
        A2[q_ch] += -beta2 * d2A_dt2[i_ch] / (den ** 2)
    return A2
