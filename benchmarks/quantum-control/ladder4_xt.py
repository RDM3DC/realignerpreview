import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from quantum.core_mimo_arp import mimo_arp_shaper, apply_drag, rfft_all
from quantum.cost import cost_J, PhysSettings, CostWeights
from quantum.grape_wrapper import grape_optimize
from quantum.optimizers import ARPGrad


def run(cfg, method):
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    dt, T_ns = cfg['dt_ns'], cfg['T_ns']
    t = np.arange(0, T_ns, dt)
    freq = np.fft.rfftfreq(t.size, d=dt)
    C = 8
    S = np.zeros((C, t.size))
    S[0, int(20/dt):int(40/dt)] = 1.0
    S[4, int(20/dt):int(40/dt)] = 1.0
    settings = PhysSettings(T1_us=cfg['T1_us'], Tphi_us=cfg['Tphi_us'],
                            f_anh_GHz=tuple(cfg['anh_GHz']),
                            f_xt_GHz=tuple(cfg['spectator_GHz']),
                            gate_cap_ns=cfg['T_ns'])
    wts = CostWeights()
    A = S.copy()
    log_df = pd.DataFrame()
    if method in ('diag-arp', 'mimo-arp', 'mimo-arp+grape'):
        tau = cfg['tau_ns']; mu = 1.0 / tau
        M = np.eye(C) * mu
        if method.startswith('mimo'):
            xt = cfg['xt_mean']
            M += xt * (np.ones((C, C)) - np.eye(C))
        alpha = np.array([mu] * C)
        A, dA = mimo_arp_shaper(S, M, alpha, dt)
        Delta = 2 * np.pi * np.array(cfg['anh_GHz'])
        A = apply_drag(A, dA, Delta, iq_pairs=[(2*i, 2*i+1) for i in range(4)])
        if method == 'mimo-arp+grape':
            U0 = {f'U{ch}': A[ch].copy() for ch in range(A.shape[0])}
            def build_A(U):
                return np.vstack([U[k] for k in sorted(U.keys())])
            def cost_wrap(Ap):
                return cost_J(Ap, freq, settings, wts)
            smoother = ARPGrad() if cfg.get('spsa_steps', 0) else None
            U_opt, log_df = grape_optimize(build_A, cost_wrap, U0,
                                           steps=cfg['grape_steps'],
                                           lr=cfg['grape_lr'],
                                           samples=cfg['grape_samples'],
                                           time_cap_ns=cfg['T_ns'],
                                           smoother=smoother,
                                           seed=cfg['seed'])
            A = build_A(U_opt)
    J, parts = cost_J(A, freq, settings, wts)
    if log_df.empty:
        log_df = pd.DataFrame([{'iter': 0, 'EPC': J, **parts}])
    summary = pd.DataFrame([{'method': method, 'EPC': J, **parts}])
    log_df.to_csv(out_dir / 'ladder4_xt_log.csv', index=False)
    summary.to_csv(out_dir / 'ladder4_xt_summary.csv', index=False)
    plt.figure(); plt.plot(log_df['iter'], log_df['EPC']); plt.xlabel('iter'); plt.ylabel('EPC');
    plt.savefig(out_dir / 'epc_vs_iter.png'); plt.close()
    plt.figure(); plt.plot([cfg['tau_ns']], [J], 'o'); plt.xlabel('tau_ns'); plt.ylabel('EPC');
    plt.savefig(out_dir / 'diag_arp_tau_curve.png'); plt.close()
    plt.figure(); plt.bar(range(4), [parts.get('leak', 0)]*4); plt.xlabel('qubit'); plt.ylabel('leak');
    plt.savefig(out_dir / 'leakage_per_qubit.png'); plt.close()
    spec = np.abs(rfft_all(A))
    plt.figure(); plt.plot(freq, spec[0]); plt.xlabel('freq'); plt.ylabel('mag');
    plt.savefig(out_dir / 'spectrum_pre_post.png'); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--method', required=True,
                   choices=['gaussian', 'diag-arp', 'mimo-arp', 'mimo-arp+grape'])
    args = p.parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    run(cfg, args.method)


if __name__ == '__main__':
    main()
