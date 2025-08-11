import numpy as np
from quantum.core_mimo_arp import mimo_arp_shaper
from quantum.cost import cost_J, PhysSettings, CostWeights

def test_cost_monotonic_wrt_gate_time():
    C, T, dt = 4, 2000, 0.02
    t = np.arange(T)*dt
    S_fast = np.zeros((C,T)); S_fast[:,100:200] = 1.0
    S_slow = np.zeros((C,T)); S_slow[:,100:400] = 1.0
    M = np.eye(C)*0.5; alpha = np.ones(C)*0.5
    Af,_ = mimo_arp_shaper(S_fast, M, alpha, dt)
    As,_ = mimo_arp_shaper(S_slow, M, alpha, dt)
    freq = np.fft.rfftfreq(T, d=dt)
    Jf,_ = cost_J(Af, freq, PhysSettings(), CostWeights())
    Js,_ = cost_J(As, freq, PhysSettings(), CostWeights())
    assert Js > Jf  # longer gate => larger decoherence term
