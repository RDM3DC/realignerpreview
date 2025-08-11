import torch
from torch.optim.optimizer import Optimizer


class RealignR(Optimizer):
    def __init__(self, params, lr=1e-3, mu=0.1, alpha=1.0,
                 weight_decay=0.0, cma_xi=0.0, cma_beta=0.0):
        super().__init__(params, dict(lr=lr, mu=mu, alpha=alpha,
                                      weight_decay=weight_decay,
                                      cma_xi=cma_xi, cma_beta=cma_beta))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, mu, alpha = g['lr'], g['mu'], g['alpha']
            wd, xi, beta = g['weight_decay'], g['cma_xi'], g['cma_beta']
            for p in g['params']:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1 - lr * wd)
                st = self.state.setdefault(p, {})
                v = st.get('v', torch.zeros_like(p))
                g2 = st.get('g2', torch.zeros((), device=p.device))
                g2.mul_(1 - beta).add_(beta * p.grad.pow(2).mean())
                alpha_eff = alpha / (1 + xi * g2.sqrt())
                v.mul_(1 - mu).add_(alpha_eff * p.grad)
                p.add_(v, alpha=-lr)
                st['v'], st['g2'] = v, g2
