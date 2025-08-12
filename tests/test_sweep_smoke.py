import subprocess
from pathlib import Path

def test_sweep_cli(tmp_path):
    out = tmp_path / "out.csv"
    subprocess.check_call([
        "python", "benchmarks/quantum-control/sweep_amp_t1_tphi_xt_plus.py",
        "--out", str(out)
    ])
    assert Path(str(out).replace(".csv", "_summary.csv")).exists()
