import torch
from torch.optim.optimizer import Optimizer


class RealignR(Optimizer):
    r"""Leaky-integrator optimizer: ``v = (1-μ) v + α g`` followed by
    ``θ ← θ - η v`` with decoupled weight decay.

    Args:
        params (iterable): parameters to optimize.
        lr (float, optional): learning rate. Default: ``1e-3``.
        mu (float, optional): leak coefficient. Default: ``0.1``.
        alpha (float, optional): gradient scaling factor. Default: ``1.0``.
        weight_decay (float, optional): decoupled weight decay. Default: ``0``.
        cma_xi (float, optional): curvature modulation strength. ``0`` disables
            curvature-modulated gain (CMA).
        cma_beta (float, optional): EMA coefficient for curvature proxy.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        mu: float = 0.1,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
        cma_xi: float = 0.0,
        cma_beta: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            mu=mu,
            alpha=alpha,
            weight_decay=weight_decay,
            cma_xi=cma_xi,
            cma_beta=cma_beta,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        loss = None
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            alpha = group["alpha"]
            wd = group["weight_decay"]
            xi = group["cma_xi"]
            beta = group["cma_beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                # Decoupled weight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)
                    state["g_ema2"] = torch.zeros((), device=p.device)

                v = state["v"]
                g2 = state["g_ema2"]

                # Simple curvature proxy: EMA of squared gradient norm
                g2.mul_(1 - beta).add_(beta * g.pow(2).mean())

                # CMA gain modulation
                alpha_eff = alpha / (1.0 + xi * g2.sqrt())

                # Leaky integrator update
                v.mul_(1 - mu).add_(alpha_eff * g)
                p.add_(v, alpha=-lr)
        return loss
