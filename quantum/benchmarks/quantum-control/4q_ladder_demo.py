import numpy as np
from quantum.core_mimo_arp import mimo_arp_shaper, apply_drag, rfft_all
from quantum.cost import cost_J, PhysSettings, CostWeights
from quantum.optimizers import ARPGrad, spsa


# Minimal 4-qubit ladder demo: build S, run ARP MIMO, compute cost.
def main():
    dt = 0.05; T = 140.0; t = np.arange(0, T, dt)
    freq = np.fft.rfftfreq(len(t), d=dt)
    Q = 4; C = 2 * Q
    # Gaussian targets on I channels 0 and 4
    def g(t, t0, s): return np.exp(-0.5*((t-t0)/s)**2)
    S = np.zeros((C, len(t)))
    S[0,:] = g(t, 60.0, 7.0); S[4,:] = g(t, 60.0, 7.0)
    # MIMO shaper
    tau = 2.0; mu = 1.0 / tau
    M = np.diag([mu]*C); alpha = np.array([mu]*C)
    A, dA = mimo_arp_shaper(S, M, alpha, dt)
    Delta = 2*np.pi*np.array([0.24,0.25,0.26,0.27])
    A = apply_drag(A, dA, Delta)
    settings = PhysSettings()
    wts = CostWeights()
    J, parts = cost_J(A, freq, settings, wts)
    print("EPC_proxy:", J, "parts:", parts)


if __name__ == "__main__":
    main()
