# MIT
import numpy as np
from typing import Sequence, Tuple


def mimo_arp_shaper(S: np.ndarray, M: np.ndarray, alpha: np.ndarray, dt: float,
                    method: str = "euler") -> Tuple[np.ndarray, np.ndarray]:
    """
    dA/dt = alpha * S - M A
    S: (C,T), M: (C,C), alpha: (C,), dt in ns
    returns A, dA_dt both (C,T)
    """
    C, T = S.shape
    A = np.zeros_like(S, dtype=float)
    dA = np.zeros_like(S, dtype=float)
    if method != "euler":
        raise NotImplementedError("only forward Euler in this skeleton")
    for k in range(1, T):
        dA[:, k-1] = alpha * S[:, k-1] - M @ A[:, k-1]
        A[:, k] = A[:, k-1] + dt * dA[:, k-1]
    dA[:, -1] = alpha * S[:, -1] - M @ A[:, -1]
    return A, dA


def apply_drag(A: np.ndarray, dA_dt: np.ndarray, Delta: np.ndarray,
               beta: float = 1.0, iq_pairs: Sequence[tuple] | None = None) -> np.ndarray:
    """
    DRAG: for each (I,Q) pair, Q += -beta * dA_I/dt / Delta_q
    Delta: per-qubit [rad/ns]; len = #qubits
    iq_pairs: list of (I_idx, Q_idx); default (0,1),(2,3),...
    """
    if iq_pairs is None:
        iq_pairs = [(2*i, 2*i+1) for i in range(len(Delta))]
    A2 = A.copy()
    for qi, (i_ch, q_ch) in enumerate(iq_pairs):
        A2[q_ch, :] += -beta * dA_dt[i_ch, :] / float(Delta[qi])
    return A2


def rfft_all(X: np.ndarray) -> np.ndarray:
    return np.fft.rfft(X, axis=1)


def band_mag(freq: np.ndarray, spec_1d: np.ndarray, f0: float, bw: float) -> float:
    sel = (freq >= f0 - bw) & (freq <= f0 + bw)
    return float(np.mean(np.abs(spec_1d[sel]))) if np.any(sel) else 0.0


def amp_noise_integral(freq: np.ndarray, specs: Sequence[np.ndarray],
                       A1: float = 1.0, A0: float = 0.05) -> float:
    w = freq.copy()
    if len(w) > 1:
        w[0] = w[1]  # avoid div0
    power = np.zeros_like(freq, dtype=float)
    for s in specs:
        power += np.abs(s) ** 2
    dw = float(freq[1] - freq[0]) if len(freq) > 1 else 1.0
    return float(np.sum((A1 / w + A0) * power) * dw)
