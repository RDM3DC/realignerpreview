import numpy as np
from quantum.optimizers import ARPGrad, spsa

def test_arpgrad_filters_noise():
    g = np.random.randn(1000)*0.5
    ag = ARPGrad(mu=0.2, alpha=1.0)
    v = np.array([ag.apply(np.array([gi]))[0] for gi in g])
    assert v.var() < g.var()

def test_spsa_runs_and_updates():
    obj = lambda th: float(np.sum(th**2))
    theta0 = np.array([0.2, -0.3, 0.1])
    th, hist = spsa(theta0, obj, steps=20)
    assert np.linalg.norm(th) < np.linalg.norm(theta0)
    assert len(hist) == 20
