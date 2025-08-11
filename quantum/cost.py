# MIT
from dataclasses import dataclass
import numpy as np
from .core_mimo_arp import rfft_all, band_mag, amp_noise_integral


@dataclass
class PhysSettings:
    T1_us: float = 50.0
    Tphi_us: float = 20.0
    f_anh_GHz: tuple = (0.25, 0.26)   # per-qubit anharm markers
    f_xt_GHz: tuple = (0.07, 0.08)    # spectator bands
    bw_GHz: float = 0.02
    gate_cap_ns: float | None = 80.0


@dataclass
class CostWeights:
    lambda_amp: float = 2e-5
    lambda_leak: float = 5e-4
    lambda_xt: float = 2e-3
    lambda_T: float = 1e-4


def _infer_dt_from_freq(freq: np.ndarray) -> float:
    # rfftfreq(N, d=dt): f[k] = k/(N*dt) ⇒ df = 1/(N*dt), N ≈ 2*(len(freq)-1)
    if len(freq) < 2:
        return 1.0
    df = float(freq[1] - freq[0])
    N_time = 2 * (len(freq) - 1)
    return 1.0 / (N_time * df)


def cost_J(A: np.ndarray, freq: np.ndarray, settings: PhysSettings,
           wts: CostWeights) -> tuple[float, dict]:
    """
    EPC-like proxy = decoherence + spectral penalties + XT + soft time penalty.
    Returns (EPC_proxy, parts_dict).
    """
    C, T = A.shape
    Aw = rfft_all(A)

    # leakage at anharmonics (sum I/Q per qubit)
    leak = 0.0
    for q, f0 in enumerate(settings.f_anh_GHz):
        i_ch, q_ch = 2*q, 2*q+1
        if i_ch >= C: break
        leak += band_mag(freq, Aw[i_ch, :], f0, settings.bw_GHz)
        if q_ch < C:
            leak += band_mag(freq, Aw[q_ch, :], f0, settings.bw_GHz)

    # crosstalk bands
    xt = 0.0
    for f0 in settings.f_xt_GHz:
        for ch in range(C):
            xt += band_mag(freq, Aw[ch, :], f0, settings.bw_GHz)

    # amplitude noise integral
    amp_int = amp_noise_integral(freq, [Aw[ch, :] for ch in range(C)])

    # gate length (samples where any I-channel active) → ns
    dt = _infer_dt_from_freq(freq)
    def eff_len_samples(x, thr=1e-3):
        idx = np.where(x > thr)[0]
        return 0 if len(idx) == 0 else (idx[-1] - idx[0])
    gate_ns = 0.0
    for q in range(C // 2):
        gate_ns = max(gate_ns, eff_len_samples(A[2*q, :]) * dt)

    # decoherence term (coarse)
    t_s = gate_ns * 1e-9
    deco = t_s/(settings.T1_us*1e-6)/2 + t_s/(settings.Tphi_us*1e-6)

    EPC = deco + wts.lambda_amp * amp_int + wts.lambda_leak * leak + wts.lambda_xt * xt
    if settings.gate_cap_ns and gate_ns > settings.gate_cap_ns:
        EPC += wts.lambda_T * (gate_ns - settings.gate_cap_ns) ** 2

    parts = dict(gate_ns=float(gate_ns), deco=float(deco),
                 amp_int=float(amp_int), leak=float(leak), xt=float(xt))
    return float(EPC), parts
