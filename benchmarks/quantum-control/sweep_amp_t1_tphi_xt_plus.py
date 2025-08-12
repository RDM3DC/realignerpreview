#!/usr/bin/env python
"""Simple sweep benchmark CLI for quantum control."""

import argparse
import numpy as np
import pandas as pd
from realignrq.quantum import costs, io


def run_sweep(out):
    rng = np.random.default_rng(0)
    samples = rng.normal(size=8)
    risk = costs.risk_p90(samples)
    summary = {"p90": risk}
    io.write_summary(out.replace(".csv", "_summary.csv"), summary)
    pd.DataFrame({"samples": samples}).to_csv(out.replace(".csv", "_raw.csv"), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="sweep.csv")
    args = p.parse_args()
    run_sweep(args.out)

if __name__ == "__main__":
    main()
