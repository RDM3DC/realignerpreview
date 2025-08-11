import numpy as np
from quantum.core_mimo_arp import mimo_arp_shaper, apply_drag
from quantum.cost import cost_J, PhysSettings, CostWeights

def test_4q_smoke():
    dt, T = 0.05, 100.0
    t = np.arange(0, T, dt); freq = np.fft.rfftfreq(t.size, d=dt)
    C = 8; S = np.zeros((C, t.size)); S[0,200:400] = 1.0; S[4,200:400] = 1.0
    tau = 2.0; mu = 1.0/tau
    M = np.diag([mu]*C); alpha = np.array([mu]*C)
    A, dA = mimo_arp_shaper(S, M, alpha, dt)
    Delta = 2*np.pi*np.array([0.24,0.25,0.26,0.27])
    A = apply_drag(A, dA, Delta, iq_pairs=[(0,1),(2,3),(4,5),(6,7)])
    J, parts = cost_J(A, freq, PhysSettings(), CostWeights())
    assert J > 0.0 and np.isfinite(J)
    assert parts["gate_ns"] > 0
