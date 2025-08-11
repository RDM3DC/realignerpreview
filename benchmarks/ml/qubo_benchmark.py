import argparse
import time
from pathlib import Path

import torch
import pandas as pd

from optim.realignr import RealignR


def make_Q(n, density, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n * n, generator=g)[: int(density * n * n)]
    Q = torch.zeros(n * n)
    Q[idx] = torch.randn(idx.size(0), generator=g)
    Q = Q.view(n, n)
    Q = (Q + Q.t()) / 2
    return Q


def loss(Q, z, tau=1.0, ent=1e-3):
    x = torch.sigmoid(z / tau)
    E = x @ Q @ x
    H = -(x * torch.log(x + 1e-8) + (1 - x) * torch.log(1 - x + 1e-8)).mean()
    return E + ent * H, dict(E=float(E.detach()), H=float(H.detach()))


def run(args):
    device = torch.device('cpu')
    Q = make_Q(args.n, args.density, args.seed).to(device)
    z = torch.zeros(args.n, device=device, requires_grad=True)
    steps = args.steps
    tau0, tau1 = 2.0, 0.3
    out_dir = Path('logs'); out_dir.mkdir(exist_ok=True)

    def cosine_tau(k):
        t = k / steps
        return tau1 + 0.5 * (tau0 - tau1) * (1 + torch.cos(torch.tensor(t * 3.1415926535)))

    opt_name = args.opt
    if opt_name == 'adamw':
        opt = torch.optim.AdamW([z], lr=args.lr)
        switch = None
    elif opt_name == 'adagrad':
        opt = torch.optim.Adagrad([z], lr=args.lr)
        switch = None
    elif opt_name in ('adamw-adagrad', 'adamw-realignr', 'adamw-realignr-cma'):
        opt = torch.optim.AdamW([z], lr=args.lr)
        def _make_second():
            if opt_name == 'adamw-adagrad':
                return torch.optim.Adagrad([z], lr=args.lr)
            else:
                return RealignR([z], lr=args.lr, mu=0.1, alpha=1.0,
                                cma_xi=0.1 if 'cma' in opt_name else 0.0,
                                cma_beta=0.05 if 'cma' in opt_name else 0.0)
        switch = _make_second
    else:
        raise ValueError('unknown opt')

    log = []
    E_best = float('inf')
    switched = False
    trigger_step = None
    trigger_gap = None
    escape_time = None
    start = time.time()
    for k in range(1, steps + 1):
        tau = float(cosine_tau(k))
        opt.zero_grad()
        L, parts = loss(Q, z, tau)
        L.backward()
        opt.step()
        E = parts['E']
        if E < E_best:
            E_best = E
        gap = (E - E_best) / abs(E_best)
        g = z.grad.detach()
        grad_var = float(g.var())
        snr = float(g.mean().abs() / (g.std() + 1e-8))
        log.append(dict(step=k, loss=float(L.detach()), E=E, tau=tau,
                        gap=float(gap), grad_var=grad_var, snr=snr))
        if switch and not switched:
            if k >= int(0.4 * steps):
                window = log[-5:]
                if len(window) == 5:
                    slope = (window[-1]['loss'] - window[0]['loss']) / max(abs(window[0]['loss']), 1e-9)
                    if slope > -0.001:
                        opt = switch()
                        switched = True
                        trigger_step = k
                        trigger_gap = gap
        if switched and escape_time is None and trigger_gap is not None:
            if gap <= trigger_gap * 0.9:
                escape_time = k - trigger_step
    elapsed = time.time() - start
    df = pd.DataFrame(log)
    df.to_csv(out_dir / f'qubo_{opt_name}.csv', index=False)
    pd.DataFrame([dict(final_gap=float(gap), escape_time=escape_time,
                       elapsed=elapsed)]).to_csv(out_dir / f'qubo_{opt_name}_summary.csv', index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--density', type=float, required=True)
    p.add_argument('--opt', required=True,
                   choices=['adamw', 'adagrad', 'adamw-adagrad', 'adamw-realignr', 'adamw-realignr-cma'])
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
