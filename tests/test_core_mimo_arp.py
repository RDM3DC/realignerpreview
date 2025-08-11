import numpy as np
from quantum.core_mimo_arp import (
    mimo_arp_shaper,
    apply_drag,
    rfft_all,
    band_mag,
)


def test_mimo_arp_shaper_matches_analytic():
    T = 100
    dt = 0.01
    S = np.ones((1, T))
    M = np.array([[0.5]])
    alpha = np.array([1.0])
    A, dA = mimo_arp_shaper(S, M, alpha, dt)
    t = np.arange(T) * dt
    # Analytic solution for dA/dt = alpha*S - m*A with S=1
    m = M[0, 0]
    A_exact = alpha[0] / m * (1 - np.exp(-m * t))
    assert np.allclose(A[0], A_exact, atol=1e-2)
    # derivative should match at final step
    assert np.allclose(dA[0, -1], alpha[0] * S[0, -1] - m * A[0, -1])


def test_apply_drag_pairs():
    C, T = 4, 5
    A = np.zeros((C, T))
    dA_dt = np.zeros((C, T))
    dA_dt[0] = np.arange(T)
    dA_dt[2] = 2 * np.arange(T)
    Delta = np.array([2.0, 4.0])
    A2 = apply_drag(A, dA_dt, Delta)
    expected_q0 = -dA_dt[0] / Delta[0]
    expected_q1 = -dA_dt[2] / Delta[1]
    assert np.allclose(A2[1], expected_q0)
    assert np.allclose(A2[3], expected_q1)
    # I channels remain unchanged
    assert np.allclose(A2[0], 0)
    assert np.allclose(A2[2], 0)


def test_fft_band_mag():
    dt = 0.01
    T = 100
    t = np.arange(T) * dt
    freq0 = 10.0
    x = np.sin(2 * np.pi * freq0 * t)
    spec = rfft_all(x[None, :])[0]
    freq = np.fft.rfftfreq(T, dt)
    mag_sig = band_mag(freq, spec, freq0, 1.0)
    mag_off = band_mag(freq, spec, 30.0, 1.0)
    assert mag_sig > 1.0
    assert mag_off < 1e-3
