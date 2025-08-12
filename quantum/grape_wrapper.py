# MIT License
import numpy as np
import pandas as pd
from typing import Callable
from .core_mimo_arp import rfft_all, amp_noise_integral


def grape_optimize(build_A: Callable[[dict], np.ndarray],
                   cost_J: Callable[[np.ndarray], tuple[float, dict]],
                   U0: dict,
                   steps: int = 300, lr: float = 0.2, samples: int = 64,
                   eps: float = 1e-3, momentum: float = 0.9,
                   smoother=None, time_cap_ns: float = 80.0,
                   seed: int = 0):
    """Finite-difference GRAPE with optional gradient smoothing.

    ``build_A(U)`` builds shaped pulses from controls ``U``.
    ``cost_J(A)`` returns (EPC_like, parts_dict).
    ``U0`` is a dict of initial controls, e.g. {"Ic": np.ndarray(T)}.

    Returns ``(U_opt, log_df)`` where ``log_df`` contains columns
    ``iter, EPC, gate_ns, amp_int`` plus any parts from ``cost_J``.
    """
    rng = np.random.default_rng(seed)
    U = {k: v.astype(float).copy() for k, v in U0.items()}
    M = {k: np.zeros_like(v) for k, v in U.items()}
    A0 = build_A(U)
    C, T = A0.shape
    freq = np.fft.rfftfreq(T, d=1.0)

    def full_cost(Udict):
        A = build_A(Udict)
        J, parts = cost_J(A)
        Aw = rfft_all(A)
        amp_int = amp_noise_integral(freq, [Aw[ch, :] for ch in range(C)])
        J += 2e-5 * amp_int
        def eff_len_samples(x, thr=1e-3):
            idx = np.where(np.abs(x) > thr)[0]
            return 0 if len(idx) == 0 else (idx[-1] - idx[0])
        gate_ns = 0.0
        for q in range(C // 2):
            gate_ns = max(gate_ns, eff_len_samples(A[2 * q, :]))
        if gate_ns > time_cap_ns:
            J += 1e-4 * (gate_ns - time_cap_ns) ** 2
        parts.update(dict(gate_ns=float(gate_ns), amp_int=float(amp_int)))
        return float(J), parts

    log = []
    for k in range(1, steps + 1):
        J, parts = full_cost(U)
        log.append({"iter": k, "EPC": J, **parts})
        total_len = sum(v.size for v in U.values())
        idx_flat = rng.choice(total_len, size=min(samples, total_len), replace=False)
        grads = {key: np.zeros_like(U[key]) for key in U}
        for flat in idx_flat:
            cum = 0
            for key in U:
                arr = U[key]
                n = arr.size
                if flat < cum + n:
                    i = flat - cum
                    old = arr.flat[i]
                    arr.flat[i] = old + eps
                    Jp, _ = full_cost(U)
                    arr.flat[i] = old - eps
                    Jm, _ = full_cost(U)
                    arr.flat[i] = old
                    grads[key].flat[i] = (Jp - Jm) / (2 * eps)
                    break
                cum += n
        if smoother:
            for key in grads:
                grads[key] = smoother.apply(grads[key])
        for key in U:
            M[key] = momentum * M[key] + (1 - momentum) * grads[key]
            U[key] -= lr * M[key]
    return U, pd.DataFrame(log)
