# MIT License
import numpy as np
from typing import Callable


class ARPGrad:
    def __init__(self, mu: float = 0.1, alpha: float = 1.0):
        self.mu, self.alpha = mu, alpha
        self.v = None

    def apply(self, g: np.ndarray) -> np.ndarray:
        if self.v is None: self.v = np.zeros_like(g)
        self.v = (1.0 - self.mu) * self.v + self.alpha * g
        return self.v


def spsa(theta: np.ndarray, obj: Callable[[np.ndarray], float],
         steps: int = 100, a: float = 0.02, c: float = 0.02,
         alpha: float = 0.602, gamma: float = 0.101,
         proj: Callable[[np.ndarray], np.ndarray] | None = None,
         smoother: ARPGrad | None = None):
    th = theta.astype(float).copy()
    hist = []
    rng = np.random.default_rng(0)
    for k in range(1, steps + 1):
        ak = a / (k ** alpha); ck = c / (k ** gamma)
        Delta = rng.choice([-1.0, 1.0], size=th.shape)
        Jp = obj(th + ck * Delta); Jm = obj(th - ck * Delta)
        ghat = (Jp - Jm) / (2.0 * ck) * Delta
        if smoother: ghat = smoother.apply(ghat)
        th = th - ak * ghat
        if proj: th = proj(th)
        hist.append((k, float(Jp), float(Jm)))
    return th, hist
