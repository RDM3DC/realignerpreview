from collections import deque
import torch

class ActionTracker:
    def __init__(self, λ1=0.1, λ2=0.0, window=200):
        self.λ1 = λ1
        self.λ2 = λ2
        self.hist = deque(maxlen=window)
        self.prev_G = None

    def step(self, loss_val: float, G: torch.Tensor):
        if self.prev_G is None:
            self.prev_G = G.detach().clone()
            drift = 0.0
        else:
            drift = torch.norm(G - self.prev_G).item()
            self.prev_G.copy_(G.detach())
        action = loss_val + self.λ1 * drift**2  # curvature term λ2 TBD
        self.hist.append(action)
        return action, drift

    def plateau(self, tol=1e-4, min_len=100):
        if len(self.hist) < min_len:
            return False
        recent = list(self.hist)[-min_len:]
        slope = (recent[-1] - recent[0]) / min_len
        return abs(slope) < tol
