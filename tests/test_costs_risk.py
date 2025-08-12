import numpy as np
from realignrq.quantum.costs import risk_p90

def test_risk_p90():
    data = np.linspace(0, 1, 101)
    assert np.isclose(risk_p90(data), 0.9)
