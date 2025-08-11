# MIT License
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


def cost_J(A: np.ndarray, freq: np.ndarray, settings: PhysSettings,
           wts: CostWeights) -> tuple[float, dict]:
    """
    EPC-like proxy = decoherence + spectral penalties + XT + soft time penalty.
    Returns (EPC_proxy, parts_dict).
    """
    C, T = A.shape
    Aw = rfft_all(A)
    # leakage at anharmonics (sum over I/Q of each qubit pair)
    leak = 0.0
    for q in range(len(settings.f_anh_GHz)):
        i_ch, q_ch = 2*q, 2*q+1
        leak += band_mag(freq, Aw[i_ch, :], settings.f_anh_GHz[q], settings.bw_GHz)
        leak += band_mag(freq, Aw[q_ch, :], settings.f_anh_GHz[q], settings.bw_GHz)
    # cross-talk bands
    xt = 0.0
    for f0 in settings.f_xt_GHz:
        for ch in range(C):
            xt += band_mag(freq, Aw[ch, :], f0, settings.bw_GHz)
    # amplitude noise integral (1/f + white)
    amp_int = amp_noise_integral(freq, [Aw[ch, :] for ch in range(C)])
    # gate length (effective span where any I-channel > thr)
    def eff_len(x, thr=1e-3):
        idx = np.where(x > thr)[0]
        return 0.0 if len(idx) == 0 else (idx[-1] - idx[0])
    gate_ns = 0.0
    for q in range(len(settings.f_anh_GHz)):
        gate_ns = max(gate_ns, eff_len(A[2*q, :]))  # I channels only
    # convert samples to ns if freq grid was built from dt
    # (caller can pass real ns directly; this skeleton treats it as ns already)
    t_s = gate_ns * 1e-9
    deco = t_s / (settings.T1_us * 1e-6) / 2 + t_s / (settings.Tphi_us * 1e-6)
    EPC = deco + wts.lambda_amp * amp_int + wts.lambda_leak * leak + wts.lambda_xt * xt
    if settings.gate_cap_ns and gate_ns > settings.gate_cap_ns:
        EPC += wts.lambda_T * (gate_ns - settings.gate_cap_ns) ** 2
    parts = dict(gate_ns=float(gate_ns), deco=float(deco),
                 amp_int=float(amp_int), leak=float(leak), xt=float(xt))
    return float(EPC), parts
