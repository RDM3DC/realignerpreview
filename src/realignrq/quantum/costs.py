import numpy as np

def epc_proxy(errors):
    """Simple error-per-gate proxy: mean of squared errors."""
    errors = np.asarray(errors)
    return float(np.mean(errors ** 2))

def risk_p90(samples):
    """Return the 90th percentile of a set of samples."""
    samples = np.asarray(samples)
    return float(np.percentile(samples, 90))
