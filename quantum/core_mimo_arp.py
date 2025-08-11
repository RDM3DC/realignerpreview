import numpy as np
from typing import Iterable


def mimo_arp_shaper(S: np.ndarray, M: np.ndarray, alpha: np.ndarray, dt: float, method: str = "euler"):
    """Simple MIMO ARP shaper using forward Euler integration.

    Args:
        S: Target envelopes with shape (C, T).
        M: Coupling matrix with shape (C, C).
        alpha: Per-channel drive strengths (C,).
        dt: Time step.
        method: Integration method (only "euler" supported).

    Returns:
        Tuple of (A, dA_dt) each of shape (C, T).
    """
    C, T = S.shape
    A = np.zeros_like(S)
    dA = np.zeros_like(S)
    for k in range(1, T):
        dA[:, k - 1] = alpha * S[:, k - 1] - M @ A[:, k - 1]
        if method == "euler":
            A[:, k] = A[:, k - 1] + dt * dA[:, k - 1]
        else:
            raise ValueError(f"Unsupported method: {method}")
    dA[:, -1] = alpha * S[:, - 1] - M @ A[:, -1]
    return A, dA


def apply_drag(A: np.ndarray, dA_dt: np.ndarray, Delta: np.ndarray, beta: float = 1.0, pairs: Iterable[tuple[int, int]] | None = None) -> np.ndarray:
    """Apply DRAG quadrature to paired I/Q channels.

    Args:
        A: Pulses array (C, T).
        dA_dt: Time derivatives of pulses (C, T).
        Delta: Detunings per pair (P,).
        beta: Scaling factor for DRAG term.
        pairs: Iterable of (I, Q) channel index pairs. Defaults to consecutive pairs.

    Returns:
        Modified pulses with DRAG applied.
    """
    A2 = A.copy()
    if pairs is None:
        pairs = [(i, i + 1) for i in range(0, A.shape[0], 2)]
    for i_ch, q_ch in pairs:
        pair_index = i_ch // 2
        A2[q_ch, :] += -beta * dA_dt[i_ch, :] / Delta[pair_index]
    return A2


def rfft_all(X: np.ndarray) -> np.ndarray:
    """Real FFT along the time axis for all channels."""
    return np.fft.rfft(X, axis=1)


def band_mag(freq: np.ndarray, spec: np.ndarray, f0: float, bw: float) -> float:
    """Average magnitude of spectrum within a frequency band."""
    sel = (freq >= f0 - bw) & (freq <= f0 + bw)
    if np.any(sel):
        return float(np.mean(np.abs(spec[sel])))
    return 0.0


def amp_noise_integral(freq: np.ndarray, specs: Iterable[np.ndarray], A1: float = 1.0, A0: float = 0.05) -> float:
    """Amplitude noise integral for a collection of spectra."""
    w = freq.copy()
    if w[0] == 0:
        w[0] = w[1]
    power = sum(np.abs(s) ** 2 for s in specs)
    d_w = freq[1] - freq[0]
    return float(np.sum((A1 / w + A0) * power) * d_w)
