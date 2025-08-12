import numpy as np
from quantum.grape_wrapper import grape_optimize


def test_grape_smoke():
    T = 100
    Ic = np.zeros(T); Ic[:40] = 1.0
    It = np.zeros(T); It[:40] = 1.0
    U0 = {"Ic": Ic, "It": It}

    def build_A(U):
        A = np.zeros((2, T))
        A[0, :40] = U["Ic"][:40]
        A[1, :40] = U["It"][:40]
        return A

    def cost_J(A):
        return float(np.mean(A ** 2)), {}

    U_opt, log = grape_optimize(build_A, cost_J, U0,
                                steps=10, lr=0.1, samples=100,
                                time_cap_ns=80.0, seed=0, momentum=0.0)
    assert log["EPC"].iloc[-1] < log["EPC"].iloc[0]
    assert log["gate_ns"].max() <= 80.0
