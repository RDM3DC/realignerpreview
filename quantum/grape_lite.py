# MIT License
import numpy as np
from typing import Callable


def grape_refine(Uc: np.ndarray, Ut: np.ndarray,
                 build_A: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 cost_J: Callable[[np.ndarray], tuple[float, dict]],
                 steps: int = 250, lr: float = 0.2, samples: int = 64,
                 eps: float = 1e-3, momentum: float = 0.9,
                 grad_smoother=None):
    """Stochastic finite-diff 'GRAPE-lite' with optional ARP gradient smoothing."""
    rng = np.random.default_rng(0)
    mUc = np.zeros_like(Uc); mUt = np.zeros_like(Ut)
    log = []
    for k in range(1, steps + 1):
        A = build_A(Uc, Ut)
        J, parts = cost_J(A)
        log.append({"iter": k, "EPC": float(J), **parts})
        idx = rng.choice(np.arange(len(Uc)), size=min(samples, len(Uc)), replace=False)
        dUc = np.zeros_like(Uc); dUt = np.zeros_like(Ut)
        for i in idx:
            for arr, dU in ((Uc, dUc), (Ut, dUt)):
                old = arr[i]; arr[i] = old + eps
                Jp, _ = cost_J(build_A(Uc, Ut))
                arr[i] = old - eps
                Jm, _ = cost_J(build_A(Uc, Ut))
                arr[i] = old
                dU[i] = (Jp - Jm) / (2 * eps)
        if grad_smoother:
            dUc = grad_smoother.apply(dUc); dUt = grad_smoother.apply(dUt)
        mUc = momentum * mUc + (1 - momentum) * dUc
        mUt = momentum * mUt + (1 - momentum) * dUt
        Uc -= lr * mUc; Ut -= lr * mUt
    return Uc, Ut, log
