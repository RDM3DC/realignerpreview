# MIT
# sweep_amp_t1_tphi_xt_plus.py
# A1/A0 × Tphi × T1 × seed sweep with XT variability.
# Extras: risk-aware objective (mean/p90/Top-k CVaR), time/freq jitter,
# cosine-basis controls, predistortion, warm-start/save.
#
# Usage example (enable risk+p90, jitters, cosine basis, predistort):
# python sweep_amp_t1_tphi_xt_plus.py \
#   --A1 4.0 --A0 0.1 --Tphi 8.0 \
#   --T1 8.0 9.0 10.0 11.0 12.0 \
#   --seeds 7 17 27 37 47 \
#   --xt-mean 0.10 --xt-sigma 0.03 --xt-max 0.30 --xt-dist normal \
#   --risk p90 --risk-samples 6 --risk-topk 2 --time-jitter-pct 0.02 \
#   --freq-jitter-anh 0.008 --freq-jitter-xt 0.010 \
#   --basis cosine:64 --predistort tanh --predistort-gain 1.2 \
#   --grape-steps 400 --l1 1.4e-3 --tv 3.0e-3 --clip-amp 1.12 \
#   --out outputs/amp_t1_tphi_xt_plus.csv

import argparse, csv, os, math
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np

# ----------------------------
# Utilities
# ----------------------------
def gaussian_pulse(t, t0, sigma, A=1.0): return A*np.exp(-0.5*((t-t0)/sigma)**2)
def rfft_all(X): return np.fft.rfft(X, axis=1)
def band_mag(freq, spec, f0, bw):
    idx=(freq>=f0-bw)&(freq<=f0+bw)
    return float(np.mean(np.abs(spec[idx]))) if np.any(idx) else 0.0
def eff_len_ns(x, dt_ns, thr=1e-3):
    idx=np.where(np.abs(x)>thr)[0]
    return 0.0 if len(idx)==0 else (idx[-1]-idx[0])*dt_ns
def percentile(xs, p):
    xs=np.array(xs,float)
    return float(np.percentile(xs, p)) if xs.size else float("nan")
def mean_std(xs):
    xs=np.array(xs,float)
    return (float(xs.mean()), float(xs.std(ddof=0))) if xs.size else (float("nan"), float("nan"))

# ----------------------------
# ARP + DRAG pipeline
# ----------------------------
def mimo_arp_shaper(S, M, alpha, dt_ns):
    C,T=S.shape; A=np.zeros_like(S); dA=np.zeros_like(S)
    for k in range(1,T):
        dA[:,k-1]=alpha*S[:,k-1]-M@A[:,k-1]
        A[:,k]=A[:,k-1]+dt_ns*dA[:,k-1]
    dA[:,-1]=alpha*S[:,-1]-M@A[:,-1]
    return A,dA

def apply_drag(A, dA_dt, Delta_rad_per_ns, iq_pairs):
    A2=A.copy()
    for qi,(i_ch,q_ch) in enumerate(iq_pairs):
        A2[q_ch,:] += - dA_dt[i_ch,:] / float(Delta_rad_per_ns[qi])
    return A2

# ----------------------------
# Cost with optional amp PSD (A1/A0)
# ----------------------------
@dataclass
class PhysSettings:
    T1_us: float = 50.0
    Tphi_us: float = 20.0
    f_anh_GHz: Tuple[float, ...] = (0.24,0.25,0.26,0.27)
    f_xt_GHz: Tuple[float, ...] = (0.07,0.08,0.09)
    bw_GHz: float = 0.02
    gate_cap_ns: Optional[float] = 80.0

@dataclass
class CostWeights:
    lambda_amp: float = 2e-5
    lambda_leak: float = 6e-4
    lambda_xt:  float = 2e-3
    lambda_T:   float = 1e-4

@dataclass
class AmpPSD:
    A1: float = 1.0
    A0: float = 0.05

def cost_J(A, freq_GHz, settings: PhysSettings, wts: CostWeights, amp_psd: Optional[AmpPSD]=None):
    C,T=A.shape; Aw=rfft_all(A)
    leak=0.0
    for q,f0 in enumerate(settings.f_anh_GHz):
        i_ch, q_ch = 2*q, 2*q+1
        if i_ch>=C: break
        leak += band_mag(freq_GHz, Aw[i_ch,:], f0, settings.bw_GHz)
        if q_ch<C: leak += band_mag(freq_GHz, Aw[q_ch,:], f0, settings.bw_GHz)
    xt=0.0
    for f0 in settings.f_xt_GHz:
        for ch in range(C): xt += band_mag(freq_GHz, Aw[ch,:], f0, settings.bw_GHz)

    A1 = 1.0 if amp_psd is None else amp_psd.A1
    A0 = 0.05 if amp_psd is None else amp_psd.A0
    w = freq_GHz.copy()
    if len(w)>1: w[0]=w[1]
    dω = float(freq_GHz[1]-freq_GHz[0]) if len(freq_GHz)>1 else 1.0
    power = np.sum(np.abs(Aw)**2, axis=0)
    amp_int = float(np.sum((A1/w + A0)*power)*dω)

    dt_ns = 1.0/(2.0*(len(freq_GHz)-1)*dω) if len(freq_GHz)>1 else 1.0
    gate_ns = 0.0
    for q in range(C//2): gate_ns = max(gate_ns, eff_len_ns(A[2*q,:], dt_ns))

    t_s = gate_ns*1e-9
    deco = t_s/(settings.T1_us*1e-6)/2.0 + t_s/(settings.Tphi_us*1e-6)

    EPC = deco + wts.lambda_amp*amp_int + wts.lambda_leak*leak + wts.lambda_xt*xt
    if settings.gate_cap_ns and gate_ns>settings.gate_cap_ns:
        EPC += wts.lambda_T * (gate_ns - settings.gate_cap_ns)**2

    parts = dict(gate_ns=float(gate_ns), deco=float(deco), amp_int=float(amp_int),
                 leak=float(leak), xt=float(xt))
    return float(EPC), parts

# ----------------------------
# Optimizers (SPSA & GRAPE)
# ----------------------------
def spsa(theta, obj, steps=120, a=0.03, c=0.02, proj=None, seed=0):
    rng=np.random.default_rng(seed); th=theta.copy()
    for k in range(1,steps+1):
        ak=a/(k**0.6); ck=c/(k**0.1)
        Delta=rng.choice([-1.0,1.0], size=th.shape)
        Jp=obj(th+ck*Delta); Jm=obj(th-ck*Delta)
        ghat=(Jp-Jm)/(2*ck)*Delta
        th=th-ak*ghat
        if proj is not None: th=proj(th)
    return th

def _soft_thresh(x, lam): return np.sign(x)*np.maximum(np.abs(x)-lam,0.0)

# ----------------------------
# Basis adapter (cosine K modes) + predistortion + time jitter
# ----------------------------
class BasisAdapter:
    def __init__(self, T, mode: str):
        self.T=T; self.kind="none"; self.K=None; self.B=None
        if mode.startswith("cosine:"):
            self.kind="cosine"; self.K=int(mode.split(":")[1])
            n=np.arange(T)[:,None]; k=np.arange(self.K)[None,:]
            self.B = np.cos(np.pi*(n+0.5)*k/float(T))  # T×K
            # normalize columns
            self.B = self.B / np.sqrt(np.sum(self.B**2, axis=0, keepdims=True)+1e-12)
    def encode(self, u):
        if self.kind=="none": return u.copy()
        # least squares coefficients: c = (B^T B)^-1 B^T u  (B columns are normalized already)
        return self.B.T @ u
    def decode(self, c):
        if self.kind=="none": return c.copy()
        return self.B @ c
    def grad_pullback(self, g_u):
        if self.kind=="none": return g_u.copy()
        return self.B.T @ g_u

def predistort(u, kind="none", gain=1.0, cubic_gamma=0.0):
    if kind=="tanh":
        return np.tanh(gain*u) / max(np.tanh(gain), 1e-8)
    elif kind=="cubic":
        return u + cubic_gamma*(u**3)
    return u

def time_jitter_resample(u, scale):
    # resample u(t) with time scaling (nearest linear interp)
    T = u.shape[-1]
    x = np.arange(T)
    xp = np.clip(np.linspace(0, T-1, T)* (1.0/scale), 0, T-1)
    return np.interp(x, xp, u)

# ----------------------------
# Ladder + XT variability
# ----------------------------
def ladder_setup(dt_ns=0.05, T_ns=140.0):
    t=np.arange(0,T_ns,dt_ns); freq=np.fft.rfftfreq(t.size, d=dt_ns)
    Q=4; C=2*Q; Ic=[0,2,4,6]; pairs=[(0,1),(2,3),(4,5),(6,7)]
    S=np.zeros((C,t.size))
    S[Ic[0],:]=gaussian_pulse(t,60.0,7.0); S[Ic[2],:]=gaussian_pulse(t,60.0,7.0)
    S[Ic[1],:]=0.20*S[Ic[0],:];         S[Ic[3],:]=0.20*S[Ic[2],:]
    f_anh=np.array([0.24,0.25,0.26,0.27]); Delta=2*np.pi*f_anh
    return t,freq,S,Ic,pairs,f_anh,Delta

def build_X(C=8, xt_mean=0.10, xt_sigma=0.03, xt_max=0.30, xt_dist="normal", seed=42):
    rng=np.random.default_rng(seed); X=np.eye(C)
    I_pairs=[]; Q_pairs=[]
    for q in range(C//2 - 1):
        I_pairs += [(2*q,2*(q+1)), (2*(q+1),2*q)]
        Q_pairs += [(2*q+1,2*(q+1)+1), (2*(q+1)+1,2*q+1)]
    def sample_xt(n):
        if xt_dist=="lognormal" and xt_mean>0:
            s=np.sqrt(np.log(1 + (xt_sigma/max(xt_mean,1e-9))**2)); mu=np.log(max(xt_mean,1e-9))-0.5*s*s
            vals=rng.lognormal(mean=mu, sigma=s, size=n)
        else:
            vals=rng.normal(loc=xt_mean, scale=xt_sigma, size=n)
        return np.clip(vals, 0.0, xt_max)
    Ii=sample_xt(len(I_pairs)); Qi=sample_xt(len(Q_pairs))
    for v,(i,j) in zip(Ii,I_pairs): X[i,j]+=v
    for v,(i,j) in zip(Qi,Q_pairs): X[i,j]+=v
    return X, dict(xt_I_mean=float(Ii.mean()), xt_I_std=float(Ii.std()), xt_I_max=float(Ii.max()),
                   xt_Q_mean=float(Qi.mean()), xt_Q_std=float(Qi.std()), xt_Q_max=float(Qi.max()))

# ----------------------------
# Risk-aware objective wrapper
# ----------------------------
def make_risk_objective(
    build_u_to_A, # function: controls(dict)->A (C×T)
    base_settings: PhysSettings,
    wts: CostWeights,
    PSD: AmpPSD,
    freq0_anh, freq0_xt,
    risk_kind="mean", risk_samples=1, risk_topk=1,
    time_jitter_pct=0.0,
    freq_jitter_anh=0.0,
    freq_jitter_xt=0.0,
    seed=0
):
    rng = np.random.default_rng(seed)
    def sample_cost(ctrls):
        # draw jitters deterministically per call
        Js=[]; parts_list=[]
        for sidx in range(risk_samples):
            # jitter sampling
            tj_scale = 1.0 + rng.uniform(-time_jitter_pct, time_jitter_pct) if time_jitter_pct>0 else 1.0
            f_anh = np.array(freq0_anh) + (rng.uniform(-freq_jitter_anh, freq_jitter_anh, size=len(freq0_anh)) if freq_jitter_anh>0 else 0.0)
            f_xt  = tuple(float(f)+float(rng.uniform(-freq_jitter_xt, freq_jitter_xt)) for f in freq0_xt) if freq_jitter_xt>0 else freq0_xt
            A = build_u_to_A(ctrls, tj_scale)
            settings = PhysSettings(T1_us=base_settings.T1_us, Tphi_us=base_settings.Tphi_us,
                                    f_anh_GHz=tuple(f_anh), f_xt_GHz=f_xt, bw_GHz=base_settings.bw_GHz,
                                    gate_cap_ns=base_settings.gate_cap_ns)
            J, parts = cost_J(A, ctrl_stats["freq_GHz"], settings, wts, PSD)  # ctrl_stats is closed-over below
            Js.append(J); parts_list.append(parts)
        Js = np.array(Js, float)
        if risk_kind=="mean" or risk_samples<=1:
            j = float(Js.mean()); idx = int(np.argmax(Js))  # log worst
        else:
            if risk_kind=="p90":
                # top-k around 90th percentile → use provided risk_topk
                idxs = np.argsort(Js)[-risk_topk:]
                j = float(np.mean(Js[idxs])); idx = int(idxs[-1])
            else: # "cvar"
                # average of the worst α fraction (α implied by topk / samples)
                idxs = np.argsort(Js)[-risk_topk:]
                j = float(np.mean(Js[idxs])); idx = int(idxs[-1])
        return j, parts_list[idx]
    # attach mutable stats via closure
    ctrl_stats = {"freq_GHz": None}
    def set_freq(freq): ctrl_stats["freq_GHz"]=freq
    return sample_cost, set_freq

# ----------------------------
# GRAPE with basis + risk (finite-difference grad in coefficient space)
# ----------------------------
def grape_optimize_risk(
    build_u_to_A, cost_sampler, set_freq_fn, freq_GHz,
    controls0: Dict[str, np.ndarray], basis: Dict[str, BasisAdapter],
    steps=150, lr=0.22, samples=48, eps=1e-3, momentum=0.9,
    l1=5e-4, tv=1e-3, l2=1e-6,
    lr_min_frac=0.1, backtrack=True, bt_factor=0.5, bt_max=3,
    clip_amp=1.25, seed=0
):
    rng=np.random.default_rng(seed)
    set_freq_fn(freq_GHz)  # let sampler know the FFT grid
    # encode to coefficient space if basis enabled
    ctrls_c = {}
    for name, u in controls0.items():
        if basis[name].kind=="none": ctrls_c[name]=u.copy()
        else: ctrls_c[name]=basis[name].encode(u)
    vel   = {k: np.zeros_like(v) for k,v in ctrls_c.items()}
    J_prev=None

    def decode_all(ctrls_c_dict):
        # decode coefficients and apply clip before predistort (clip_amp is later enforced again)
        ctrls_u={}
        for name,c in ctrls_c_dict.items():
            if basis[name].kind=="none": u=c
            else: u=basis[name].decode(c)
            if clip_amp is not None: u=np.clip(u, -clip_amp, clip_amp)
            ctrls_u[name]=u
        return ctrls_u

    # objective from ctrls_c
    def obj(ctrls_c_dict):
        ctrls_u = decode_all(ctrls_c_dict)
        J, parts = cost_sampler(ctrls_u)
        if l2: J += l2 * sum((u*u).sum() for u in ctrls_u.values())
        return float(J), parts

    for it in range(1, steps+1):
        # current value
        J, parts = obj(ctrlls := ctrls_c)
        # coordinate sampling for finite-diff gradient (subset of indices)
        grads={k: np.zeros_like(v) for k,v in ctrls_c.items()}
        # choose time indices uniformly for each control in *sample domain* length
        # but we perturb in coefficient space → sample all coeffs sparsely:
        for name,c in ctrls_c.items():
            K = c.size
            idx = rng.choice(np.arange(K), size=min(samples, K), replace=False)
            for i in idx:
                old = c[i]
                c[i] = old + eps; Jp,_ = obj(ctrls_c)
                c[i] = old - eps; Jm,_ = obj(ctrls_c)
                c[i] = old
                grads[name][i] = (Jp - Jm)/(2*eps)

        lr_eff = lr_min_frac*lr + 0.5*(lr - lr_min_frac*lr)*(1 + math.cos(math.pi*it/steps))

        def propose(scale):
            tmp={k:v.copy() for k,v in ctrls_c.items()}
            vtmp={k:v.copy() for k,v in vel.items()}
            for name in tmp:
                vtmp[name]=momentum*vtmp[name]+(1-momentum)*grads[name]
                tmp[name]-=scale*vtmp[name]
                # prox L1/TV in *sample* domain (decode→prox→encode back)
                if (l1 or tv):
                    u = decode_all({name: tmp[name]})[name]
                    if clip_amp is not None: np.clip(u, -clip_amp, clip_amp, out=u)
                    # L1
                    if l1: u = np.sign(u)*np.maximum(np.abs(u)-scale*l1, 0.0)
                    # TV
                    if tv:
                        du = np.diff(u, prepend=u[:1])
                        du = np.sign(du)*np.maximum(np.abs(du)-scale*tv, 0.0)
                        u = np.cumsum(du)
                        u -= (u.mean() - decode_all({name:ctrlls := ctrls_c[name] if isinstance(ctrlls,np.ndarray) else ctrls_c[name]})[name].mean())
                    # re-encode to coefficients if basis
                    tmp[name] = tmp[name] if basis[name].kind=="none" else basis[name].encode(u)
            return tmp, vtmp

        # backtracking
        scale=lr_eff; tried=0
        while True:
            cand,vtmp=propose(scale)
            Jc,_=obj(cand)
            improve=(J_prev is None) or (Jc <= J + 1e-6)
            if (not backtrack) or improve or tried>=bt_max:
                ctrls_c=cand; vel=vtmp; J_prev=Jc; break
            scale*=bt_factor; tried+=1

    # final decode to sample domain
    ctrls_u = decode_all(ctrls_c)
    return ctrls_u

# ----------------------------
# Build A from controls with predistort + time jitter
# ----------------------------
def make_builder(Ic, S, M, alpha, dt, Delta, pairs,
                 predistort_kind="none", predistort_gain=1.0, cubic_gamma=0.0):
    # controls are two I-channels (Ic[0], Ic[2]); Q channels filled by DRAG
    def build(ctrls_u: Dict[str,np.ndarray], tj_scale: float):
        U=np.zeros_like(S)
        u0 = predistort(ctrls_u["Ic0"], predistort_kind, predistort_gain, cubic_gamma)
        u2 = predistort(ctrls_u["Ic2"], predistort_kind, predistort_gain, cubic_gamma)
        if abs(tj_scale-1.0)>1e-9:
            u0 = time_jitter_resample(u0, tj_scale)
            u2 = time_jitter_resample(u2, tj_scale)
        U[Ic[0],:], U[Ic[2],:] = u0, u2
        A,dA = mimo_arp_shaper(U, M, alpha, dt)
        return apply_drag(A,dA,Delta,pairs)
    return build

# ----------------------------
# One experiment (A1,A0,Tphi,T1,seed)
# ----------------------------
def run_case(args, A1,A0,Tphi_us,T1_us,seed):
    # time/freq grids and ladder
    dt=args.dt_ns
    t,freq,S,Ic,pairs,f_anh,Delta = ladder_setup(dt_ns=dt)
    X, xt_stats = build_X(C=S.shape[0], xt_mean=args.xt_mean, xt_sigma=args.xt_sigma,
                          xt_max=args.xt_max, xt_dist=args.xt_dist, seed=seed)
    # premix + baseline Gaussian+DRAG
    def Jwrap(A, settings): return cost_J(A, freq, settings, CostWeights(), AmpPSD(A1,A0))

    phys_base = PhysSettings(T1_us=T1_us, Tphi_us=Tphi_us,
                             f_anh_GHz=tuple(f_anh), f_xt_GHz=(0.07,0.08,0.09),
                             bw_GHz=0.02, gate_cap_ns=80.0)

    Ag = X@S
    Ag = apply_drag(Ag, np.gradient(Ag, dt, axis=1), Delta, pairs)
    Jg, Pg = Jwrap(Ag, phys_base)

    # SPSA over M off-diagonals
    tau=2.0; mu=1.0/tau; M0=np.diag([mu]*S.shape[0]); alpha=np.array([mu]*S.shape[0])
    neighbors=[]; Q=len(f_anh)
    for q in range(Q-1):
        neighbors += [(2*q,2*(q+1)), (2*(q+1),2*q), (2*q+1,2*(q+1)+1), (2*(q+1)+1,2*q+1)]
    theta0=np.zeros(len(neighbors))
    def M_from_theta(th):
        M=M0.copy()
        for v,(i,j) in zip(th, neighbors): M[i,j]=v
        return M

    # risk-ready sampler for SPSA (use mean; SPSA is just for M init)
    def spsa_obj(th):
        M = M_from_theta(th)
        A2,dA2 = mimo_arp_shaper(S, M, alpha, dt)
        A2 = apply_drag(A2, dA2, Delta, pairs)
        J,_ = cost_J(A2, freq, phys_base, CostWeights(), AmpPSD(A1,A0))
        return J + 5e-3*np.sum(th**2)
    th = spsa(theta0, spsa_obj, steps=args.spsa_steps, proj=lambda x: np.clip(x,0.0,0.3), seed=seed)
    M = M_from_theta(th)

    # GRAPE with basis, predistort, and risk-aware objective
    builder = make_builder(Ic, S, M, alpha, dt, Delta, pairs,
                           predistort_kind=args.predistort, predistort_gain=args.predistort_gain,
                           cubic_gamma=args.predistort_cubic_gamma)

    # basis per control
    T = S.shape[1]
    basis = {
        "Ic0": BasisAdapter(T, args.basis),
        "Ic2": BasisAdapter(T, args.basis),
    }

    # initial controls (optionally warm-start)
    if args.warm_start and os.path.exists(args.warm_start):
        data = np.load(args.warm_start)
        u0 = data["Ic0"]; u2 = data["Ic2"]
    else:
        u0 = S[Ic[0],:].copy(); u2 = S[Ic[2],:].copy()
    controls0 = {"Ic0": u0, "Ic2": u2}

    # risk sampler
    cost_sampler, set_freq = make_risk_objective(
        build_u_to_A=builder,
        base_settings=phys_base,
        wts=CostWeights(),
        PSD=AmpPSD(A1,A0),
        freq0_anh=np.array(phys_base.f_anh_GHz),
        freq0_xt=phys_base.f_xt_GHz,
        risk_kind=args.risk,
        risk_samples=args.risk_samples,
        risk_topk=max(1, args.risk_topk),
        time_jitter_pct=args.time_jitter_pct,
        freq_jitter_anh=args.freq_jitter_anh,
        freq_jitter_xt=args.freq_jitter_xt,
        seed=seed
    )
    set_freq(freq)

    ctrlsF = grape_optimize_risk(
        builder, cost_sampler, set_freq, freq,
        controls0, basis,
        steps=args.grape_steps, lr=args.grape_lr, samples=args.grape_samples,
        eps=args.eps, momentum=0.9, l1=args.l1, tv=args.tv, l2=args.l2,
        lr_min_frac=args.lr_min_frac, backtrack=True, bt_factor=0.5, bt_max=3,
        clip_amp=args.clip_amp, seed=seed
    )
    A_grape = builder(ctrlsF, tj_scale=1.0)
    Jx, Px = Jwrap(A_grape, phys_base)

    # sparsity metrics (TV & zero fraction on the two driven I channels)
    def sparsity(A):
        u0 = A[Ic[0],:]; u2 = A[Ic[2],:]
        tv = 0.5*(np.sum(np.abs(np.diff(u0))) + np.sum(np.abs(np.diff(u2))))
        zf = 0.5*(np.mean(np.abs(u0)<1e-3) + np.mean(np.abs(u2)<1e-3))
        return dict(tv=float(tv), zero_frac=float(zf))
    spars = sparsity(A_grape)
    spars_spsa = sparsity(apply_drag(*mimo_arp_shaper(np.vstack([S*0+controls0["Ic0"], S*0+controls0["Ic2"]])[:2], M, alpha, dt), Delta, pairs))  # quick baseline proxy

    rec = {
        "A1":A1, "A0":A0, "Tphi_us":Tphi_us, "T1_us":T1_us, "seed":seed,
        "EPC_gauss":Jg, "gate_gauss":Pg["gate_ns"], "amp_gauss":Pg["amp_int"],
        "EPC_grape":Jx, "gate_grape":Px["gate_ns"], "amp_grape":Px["amp_int"],
        "tv_grape":spars["tv"], "zero_grape":spars["zero_frac"],
        "tv_spsa":spars_spsa["tv"], "zero_spsa":spars_spsa["zero_frac"],
        **xt_stats,
    }
    # optional save warm finish
    if args.save_ctrls:
        os.makedirs(os.path.dirname(args.save_ctrls) or ".", exist_ok=True)
        np.savez(args.save_ctrls, Ic0=ctrlsF["Ic0"], Ic2=ctrlsF["Ic2"])
    return rec

# ----------------------------
# Summaries (mean/std, p50/p90, deltas, pass rates)
# ----------------------------
def summarize(rows: List[Dict], gate_thresh_ns=75.0):
    groups={}
    for r in rows:
        key=(float(r["A1"]), float(r["A0"]), float(r["Tphi_us"]), float(r["T1_us"]))
        groups.setdefault(key, []).append(r)
    out=[]
    for (A1,A0,Tphi,T1), g in groups.items():
        EPCg=[r["EPC_grape"] for r in g]
        gateg=[r["gate_grape"] for r in g]
        tvg =[r["tv_grape"] for r in g]; tvs=[r["tv_spsa"] for r in g]
        EPCg_m,EPCg_s=mean_std(EPCg)
        gate_m,gate_s=mean_std(gateg)
        p50=percentile(EPCg,50); p90=percentile(EPCg,90)
        tv_gain_each=[100.0*(1.0-(tg/ts if ts>0 else np.nan)) for tg,ts in zip(tvg,tvs)]
        tv_gain_m,tv_gain_s = mean_std(tv_gain_each)
        gate_ok_rate=100.0*np.mean(np.array(gateg)<=gate_thresh_ns)
        out.append({
            "A1":A1,"A0":A0,"Tphi_us":Tphi,"T1_us":T1,
            "EPC_grape_mean":EPCg_m,"EPC_grape_std":EPCg_s,
            "EPC_grape_p50":p50,"EPC_grape_p90":p90,
            "gate_grape_mean":gate_m,"gate_grape_std":gate_s,
            "TV_gain_pct_mean":tv_gain_m,"TV_gain_pct_std":tv_gain_s,
            f"gate_le_{int(gate_thresh_ns)}ns_%":gate_ok_rate,
        })
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/amp_t1_tphi_xt_plus.csv")
    ap.add_argument("--A1", nargs="+", type=float, default=[4.0])
    ap.add_argument("--A0", nargs="+", type=float, default=[0.1])
    ap.add_argument("--Tphi", nargs="+", type=float, default=[12.0])
    ap.add_argument("--T1", nargs="+", type=float, default=[10.0, 12.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7,17,27,37,47])
    # XT variability
    ap.add_argument("--xt-mean", type=float, default=0.10)
    ap.add_argument("--xt-sigma", type=float, default=0.03)
    ap.add_argument("--xt-max", type=float, default=0.30)
    ap.add_argument("--xt-dist", choices=["normal","lognormal"], default="normal")
    # time base & thresholds
    ap.add_argument("--dt-ns", type=float, default=0.05)
    ap.add_argument("--gate-threshold-ns", type=float, default=75.0)
    # risk options
    ap.add_argument("--risk", choices=["mean","p90","cvar"], default="mean")
    ap.add_argument("--risk-samples", type=int, default=1)
    ap.add_argument("--risk-topk", type=int, default=1)  # used for p90/cvar
    # jitters
    ap.add_argument("--time-jitter-pct", type=float, default=0.0)
    ap.add_argument("--freq-jitter-anh", type=float, default=0.0)
    ap.add_argument("--freq-jitter-xt", type=float, default=0.0)
    # basis & predistort
    ap.add_argument("--basis", type=str, default="none")  # e.g., "cosine:64"
    ap.add_argument("--predistort", choices=["none","tanh","cubic"], default="none")
    ap.add_argument("--predistort-gain", type=float, default=1.0)
    ap.add_argument("--predistort-cubic-gamma", type=float, default=0.0)
    # warm-start/save
    ap.add_argument("--warm-start", type=str, default="")
    ap.add_argument("--save-ctrls", type=str, default="")
    # optim knobs
    ap.add_argument("--spsa-steps", type=int, default=120)
    ap.add_argument("--grape-steps", type=int, default=150)
    ap.add_argument("--grape-lr", type=float, default=0.22)
    ap.add_argument("--grape-samples", type=int, default=48)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--l1", type=float, default=5e-4)
    ap.add_argument("--tv", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--lr-min-frac", type=float, default=0.1)
    ap.add_argument("--clip-amp", type=float, default=1.25)
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    raw=[]
    for A1 in args.A1:
        for A0 in args.A0:
            for Tphi in args.Tphi:
                for T1 in args.T1:
                    for sd in args.seeds:
                        rec=run_case(args, A1,A0,Tphi,T1, seed=sd)
                        raw.append(rec)

    raw_path=args.out.replace(".csv","_raw.csv")
    with open(raw_path,"w",newline="") as f:
        if raw:
            w=csv.DictWriter(f, fieldnames=list(raw[0].keys()))
            w.writeheader(); w.writerows(raw)
    print("wrote", raw_path)

    summ=summarize(raw, gate_thresh_ns=args.gate_threshold_ns)
    summ_path=args.out.replace(".csv","_summary.csv")
    with open(summ_path,"w",newline="") as f:
        if summ:
            w=csv.DictWriter(f, fieldnames=list(summ[0].keys()))
            w.writeheader(); w.writerows(summ)
    print("wrote", summ_path)

if __name__=="__main__":
    main()