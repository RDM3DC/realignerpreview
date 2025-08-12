import numpy as np

def mimo_arp_shaper(S, M, alpha, dt):
    """Multi-input multi-output ARP shaper with spectral smoothing."""
    A = (np.eye(M.shape[0]) + M) @ S
    F = np.fft.rfft(A, axis=1)
    w = np.fft.rfftfreq(A.shape[1], dt)
    H = 1.0 / (1.0 + alpha * (1j * 2 * np.pi * w))
    A = np.fft.irfft(F * H[None, :], n=A.shape[1], axis=1)
    dA = np.gradient(A, dt, axis=1)
    return A, dA
