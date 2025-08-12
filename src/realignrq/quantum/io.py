"""Input/output helpers for writing CSV summaries."""

import pandas as pd

def write_summary(path, data):
    """Write a summary dictionary to CSV."""
    df = pd.DataFrame([data])
    df.to_csv(path, index=False)
    return path
