import torch

class ARPAdamW(torch.optim.AdamW):
    def __init__(self, params, alpha=0.01, mu=0.00521, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.alpha = alpha
        self.mu = mu

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'G' not in state:
                    state['G'] = torch.ones_like(p)

                G = state['G']
                I = grad.abs()

                G += self.alpha * I - self.mu * G
                adaptive_scale = G / (G.mean() + 1e-8)

                p.mul_(adaptive_scale)

        return loss
