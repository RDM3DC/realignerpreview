Got you. Here’s a one-shot bootstrap script that rebuilds a clean, working “RealignR-Q” repo skeleton with all the folders, key modules, configs, tests, CI, and the main sweep/FT/hybrid runners. Paste this into a file named bootstrap_realignrq.sh at an empty folder (or in your cloned repo), run it, then commit/push.

It’s minimal but runs smoke tests and writes CSVs; you (or your collaborator) can drop in the heavy internals later without changing the CLI or layout.

#!/usr/bin/env bash
set -euo pipefail

# ——————————————————————————————————————————————————————————
# 0) Root
# ——————————————————————————————————————————————————————————
mkdir -p src/realignrq/{quantum,ft,hw,arrays} \
         benchmarks/{quantum-control,ft,hybrid} \
         ft hybrid .github/workflows tests outputs

# .gitignore
cat > .gitignore <<'GI'
__pycache__/
*.pyc
.env
.venv
outputs/
GI

# README
cat > README.md <<'MD'
# RealignR-Q (preview)

Risk-aware quantum-control optimizer: **MIMO-ARP + DRAG-2 + sparse GRAPE**, with FT sims (Stim), telemetry hooks, array calibration, and hybrid VQE/QAOA.

## Quickstart
```bash
pip install -e .
pytest -q
python benchmarks/quantum-control/sweep_amp_t1_tphi_xt_plus.py --Tphi 0.1 --out outputs/smoke.csv

Outputs land in _raw.csv and _summary.csv.

Layout
	•	src/realignrq/quantum/ core shaping + costs + sweeps
	•	benchmarks/quantum-control/ main sweep CLI + baselines
	•	src/realignrq/ft/ FT glue (noise build + surface‐code sweep)
	•	benchmarks/ft/ FT apps (VQE/QPE/Hubbard/Floquet/Redshift)
	•	src/realignrq/hw/ telemetry + adaptive control stubs
	•	src/realignrq/arrays/ array calibrator (tile→stripe→global)
MD

pyproject

cat > pyproject.toml <<‘PYT’
[build-system]
requires = [“setuptools>=68”, “wheel”]
build-backend = “setuptools.build_meta”

[project]
name = “realignrq”
version = “0.1.0”
description = “Risk-aware quantum-control optimizer: MIMO-ARP + DRAG-2 + sparse GRAPE”
readme = “README.md”
license = {text = “MIT”}
requires-python = “>=3.9”
dependencies = [
“numpy>=1.24”, “scipy>=1.11”, “pandas>=2.0”,
“numba>=0.57”, “tqdm>=4.66”, “pyyaml>=6.0”,
“matplotlib>=3.8”, “networkx>=3.2”, “stim>=1.12”
]
[project.optional-dependencies]
dev = [“pytest>=7.4”]
PYT

——————————————————————————————————————————————————————————

1) Package: src/realignrq

——————————————————————————————————————————————————————————

cat > src/realignrq/init.py <<‘PY’
all = []
PY

quantum/mimo_arp.py

cat > src/realignrq/quantum/mimo_arp.py <<‘PY’
import numpy as np

def mimo_arp_shaper(S, M, alpha, dt):
“””
S: (C,T) seed controls per channel
M: (C,C) crosstalk matrix (0 on diag, >=0 off-diag)
alpha: ARP smoothing scalar (0..1)
dt: seconds per sample
“””
C,T = S.shape
A = (np.eye(C) + M) @ S
W = np.fft.rfftfreq(T, d=dt)
H = 1.0/(1.0 + alpha*(1j2np.piW))
F = np.fft.rfft(A, axis=1)
A = np.fft.irfft(FH[None,:], n=T, axis=1)
dA = np.gradient(A, dt, axis=1)
d2A = np.gradient(dA, dt, axis=1)
return A, dA, d2A
PY

quantum/drag2.py

cat > src/realignrq/quantum/drag2.py <<‘PY’
import numpy as np

def apply_drag2(A, dA, d2A, Delta, iq_pairs, beta1=1.0, beta2=0.2):
“””
A: (C,T) I/Q controls; iq_pairs: list of (i_ch, q_ch) indices per qubit
Delta: array shape (nqubits,) anharm detuning (rad/s)
“””
B = A.copy()
for q,(i_ch,q_ch) in enumerate(iq_pairs):
den = float(max(1e-9, Delta[q]))
B[q_ch] += - beta1 * dA[i_ch] / den
B[q_ch] += - beta2 * d2A[i_ch] / (den**2)
return B
PY

quantum/phase_mod.py

cat > src/realignrq/quantum/phase_mod.py <<‘PY’
import numpy as np
def phase_basis_synthesize(T, modes=256):
# toy phase schedule [T] in radians; real impl would do basis expansion
t = np.linspace(0,1,T,endpoint=False)
return 0.3np.sin(2np.pitmodes/(modes+1))
PY

quantum/arp_gate_filter.py

cat > src/realignrq/quantum/arp_gate_filter.py <<‘PY’
import numpy as np
def alpha_of_phase(phi, base=0.12, depth=0.05):
return base - depthnp.cos(phi)
def apply_arp_on_gate(u_t, dt, phi, base=0.12, depth=0.05):
a = alpha_of_phase(phi, base, depth)
W = np.fft.rfftfreq(u_t.shape[1], d=dt)
H = 1.0/(1.0 + a(1j2np.piW))
F = np.fft.rfft(u_t, axis=1)
return np.fft.irfft(FH[None,:], n=u_t.shape[1], axis=1)
PY

quantum/costs.py

cat > src/realignrq/quantum/costs.py <<‘PY’
import numpy as np

class CostWeights:
def init(self, lambda_amp=2e-5, lambda_leak=6e-4, lambda_xt=2e-3, lambda_T=1e-4):
self.amp=lambda_amp; self.leak=lambda_leak; self.xt=lambda_xt; self.T=lambda_T

def epc_proxy(pulses):
# cheap stand-in: norm-based proxy; replace with process fidelity calc
a = np.linalg.norm(pulses) / (pulses.size**0.5 + 1e-9)
return min(1.0, 1e-3*a)

def leakage_proxy(pulses):
return 1e-4 * (np.abs(pulses).max())

def xt_proxy(M):
return float(np.maximum(0.0, (M - np.diag(np.diag(M)))).mean())

def amp_psd_penalty(pulses):
F = np.fft.rfft(pulses, axis=1)
return float(np.mean(np.abs(F)**2))

def gate_time_ns(T, dt):
return 1e9 * T*dt

def cost_J(pulses, M, dt, w:CostWeights, gate_cap_ns=None):
epc = epc_proxy(pulses)
leak = leakage_proxy(pulses)
xt = xt_proxy(M)
amp = amp_psd_penalty(pulses)
Tns = gate_cap_ns or gate_time_ns(pulses.shape[1], dt)
J = epc + w.ampamp + w.leakleak + w.xtxt + w.Tmax(0.0, Tns - gate_cap_ns) if gate_cap_ns else epc + w.ampamp + w.leakleak + w.xt*xt
return dict(EPC=epc, leak=leak, xt=xt, amp=amp, gate_ns=Tns, J=J)
PY

quantum/io.py

cat > src/realignrq/quantum/io.py <<‘PY’
import pandas as pd, pathlib, json
def write_raw(path, rows):
p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(p, index=False)
def write_summary(path, rows):
import numpy as np
df = pd.DataFrame(rows)
s = {
“EPC_grape_p90”: float(np.quantile(df[“EPC”], 0.9)),
“gate_grape_mean”: float(df[“gate_ns”].mean()),
“TV_gain_pct_mean”: float(df.get(“TV_gain”, pd.Series([0]*len(df))).mean()),
“gate_le_50ns_%”: float((df[“gate_ns”]<=50).mean()*100.0)
}
p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([s]).to_csv(p, index=False)
PY

quantum/sweeps.py

cat > src/realignrq/quantum/sweeps.py <<‘PY’
import numpy as np
from .mimo_arp import mimo_arp_shaper
from .drag2 import apply_drag2
from .costs import CostWeights, cost_J

def run_case(seed=7, T=256, C=4, dt=1e-9, alpha=0.12, gate_cap_ns=52,
Delta=None, iq_pairs=None, M=None, weights=None):
rng = np.random.default_rng(seed)
S = 0.1rng.standard_normal((C,T))
M = np.zeros((C,C)) if M is None else M
A, dA, d2A = mimo_arp_shaper(S, M, alpha, dt)
Delta = np.ones(C//2)(2np.pi50e6) if Delta is None else Delta
iq_pairs = [(2i, 2i+1) for i in range(C//2)] if iq_pairs is None else iq_pairs
B = apply_drag2(A, dA, d2A, Delta, iq_pairs, beta1=1.0, beta2=0.3)
w = weights or CostWeights()
metrics = cost_J(B, M, dt, w, gate_cap_ns=gate_cap_ns)
# tiny fake TV gain
metrics[“TV_gain”] = 40.0 + 10.0*rng.random()
return metrics
PY

——————————————————————————————————————————————————————————

2) Benchmarks: quantum-control sweep CLI

——————————————————————————————————————————————————————————

cat > benchmarks/quantum-control/sweep_amp_t1_tphi_xt_plus.py <<‘PY’
import argparse, json
from pathlib import Path
from tqdm import trange
from realignrq.quantum import sweeps, io
from realignrq.quantum.costs import CostWeights

def main():
ap = argparse.ArgumentParser()
ap.add_argument(”–out”, required=True)
ap.add_argument(”–seeds”, nargs=”*”, type=int, default=[7,17,27,37,47])
ap.add_argument(”–Tphi”, type=float, default=0.1)
ap.add_argument(”–gate-threshold-ns”, type=float, default=52.0)
ap.add_argument(”–alpha”, type=float, default=0.12)
ap.add_argument(”–lam-T”, type=float, default=1e-4)
ap.add_argument(”–lam-amp”, type=float, default=2e-5)
args = ap.parse_args()

rows=[]
for s in args.seeds:
    m = sweeps.run_case(seed=s, alpha=args.alpha, gate_cap_ns=args.gate_threshold_ns,
                        weights=CostWeights(lambda_T=args.lam_T, lambda_amp=args.lam_amp))
    m["seed"]=s; rows.append(m)
raw = args.out.replace(".csv","_raw.csv")
summ = args.out.replace(".csv","_summary.csv")
io.write_raw(raw, rows); io.write_summary(summ, rows)
print(f"Wrote {raw} and {summ}")

if name==”main”:
main()
PY

——————————————————————————————————————————————————————————

3) FT glue (noise & surface code skeletons)

——————————————————————————————————————————————————————————

cat > src/realignrq/ft/build_pauli_noise.py <<‘PY’
import yaml, argparse
def main():
ap=argparse.ArgumentParser()
ap.add_argument(”–pulses”, nargs=”+”, required=False)
ap.add_argument(”–out”, required=True)
args=ap.parse_args()
# Placeholder nominal model; replace with RB/iRB/leakRB extraction
model = {
“one_qubit”:{“pauli”:{“pX”:1.5e-4,“pY”:1.5e-4,“pZ”:2.0e-4},“leakage”:3.0e-5},
“two_qubit”:{“pauli”:{“pIX”:2.0e-4,“pZZ”:3.0e-4},“correlated_xt”:1.0e-4,“leakage”:5.0e-5},
“timing”:{“gate_ns”:{“x90”:10,“cz”:12},“idle_ns”:10}
}
yaml.safe_dump(model, open(args.out,“w”))
print(“Wrote”, args.out)
if name==”main”: main()
PY

cat > src/realignrq/ft/surface_code_sweep.py <<‘PY’
import argparse, pandas as pd, numpy as np
def simulate_logical_error(p1,p2,leak,d):
# toy scaling: pL ≈ 0.1*(p2/1e-2)^((d+1)/2)
return 0.1*((p2/1e-2)**((d+1)/2))
def main():
ap=argparse.ArgumentParser()
ap.add_argument(”–noise”, required=False)
ap.add_argument(”–distances”, nargs=”+”, default=[“3”,“5”,“7”])
ap.add_argument(”–out”, required=True)
a=ap.parse_args()
rows=[]
for d in map(int,a.distances):
rows.append({“d”:d,“pL”:simulate_logical_error(2e-4,8e-4,3e-5,d)})
pd.DataFrame(rows).to_csv(a.out,index=False); print(“Wrote”,a.out)
if name==”main”: main()
PY

——————————————————————————————————————————————————————————

4) HW telemetry/adaptive (stubs)

——————————————————————————————————————————————————————————

cat > src/realignrq/hw/site.yaml <<‘YML’
rig:
awg: {addr: “TCP://awg1:5025”, max_amp: 1.0, max_slew: 0.08}
fridge: {sensor: “fridge0”, temp_max_mK: 20}
readout: {clip_thresh_db: -1.0}
bands:
spectators_GHz: [0.07, 0.08, 0.09]
loops:
fast_period_s: 60
medium_period_s: 3600
slow_period_s: 86400
triggers:
epc_p90_rel: 1.15
notch_loss_db: 6.0
leakage_rel: 1.25
export:
csv_dir: “outputs/hw”
YML

cat > src/realignrq/hw/telemetry_agent.py <<‘PY’
import time, pathlib, csv, random, datetime as dt, yaml
CFG=yaml.safe_load(open(file.replace(“telemetry_agent.py”,“site.yaml”)))
def snapshot():
return {
“ts”: dt.datetime.utcnow().isoformat()+“Z”,
“EPC_p90_rel”: 1.0+0.05random.random(),
“notch_loss_db”: 2.0random.random(),
“leak_rel”: 1.0+0.1*random.random(),
“gate_ns_mean”: 10.0+random.random()
}
def main():
outdir = pathlib.Path(CFG[“export”][“csv_dir”]); outdir.mkdir(parents=True,exist_ok=True)
fn = outdir/“telemetry.csv”
with fn.open(“a”, newline=””) as f:
w=csv.DictWriter(f, fieldnames=[“ts”,“EPC_p90_rel”,“notch_loss_db”,“leak_rel”,“gate_ns_mean”])
if f.tell()==0: w.writeheader()
for _ in range(5):
s=snapshot(); w.writerow(s); f.flush(); time.sleep(0.2)
print(“Wrote”, fn)
if name==”main”: main()
PY

cat > src/realignrq/hw/adaptive_controller.py <<‘PY’
def run_once():
# placeholder: wire your RB & refine hooks here
return {“refined”: True}
if name==”main”:
print(run_once())
PY

——————————————————————————————————————————————————————————

5) Arrays (tile→stripe→global) stubs

——————————————————————————————————————————————————————————

cat > src/realignrq/arrays/group_lasso.py <<‘PY’
import numpy as np
def group_lasso_step(M, grad, lam_group=5e-3, lam_dist=1e-3, dist=None, lr=0.05):
M = M - lrgrad
if dist is not None: M -= lrlam_dist * (dist * M)
for g in range(M.shape[0]):
v = M[g,:]; n = np.linalg.norm(v)+1e-12
M[g,:] = max(0,1-lr*lam_group/n)*v
return np.clip(M,0.0,0.3)
PY

cat > src/realignrq/arrays/array_calibrate.py <<‘PY’
def run(mode=“survey”, grid=“4x4”, **kw):
return {“mode”:mode,“grid”:grid,“ok”:True}
if name==”main”:
print(run())
PY

——————————————————————————————————————————————————————————

6) FT Apps (minimal runners, stubs produce CSVs)

——————————————————————————————————————————————————————————

VQE

cat > benchmarks/ft/ft_vqe.py <<‘PY’
import argparse, pandas as pd
def main():
ap=argparse.ArgumentParser(); ap.add_argument(”–device”); ap.add_argument(”–config”); ap.add_argument(”–ham”); ap.add_argument(”–out”,required=True)
a=ap.parse_args()
pd.DataFrame([{“energy”:-1.0,“shots_overhead”:3.2,“pL_layer”:7e-7}]).to_csv(a.out,index=False)
print(“Wrote”,a.out)
if name==”main”: main()
PY

QPE

cat > benchmarks/ft/ft_qpe.py <<‘PY’
import argparse, pandas as pd
def main():
ap=argparse.ArgumentParser(); ap.add_argument(”–config”); ap.add_argument(”–out”,required=True)
a=ap.parse_args()
pd.DataFrame([{“phase_rmse”:4e-4,“success”:0.995}]).to_csv(a.out,index=False)
print(“Wrote”,a.out)
if name==”main”: main()
PY

Hubbard + DD

cat > benchmarks/ft/ft_hubbard_dd.py <<‘PY’
import argparse, pandas as pd
def main():
ap=argparse.ArgumentParser(); ap.add_argument(”–device”); ap.add_argument(”–config”); ap.add_argument(”–out”,required=True)
a=ap.parse_args()
pd.DataFrame([{“rel_E_err”:0.008,“double_occ_err”:0.015,“S_pipi_err”:0.015}]).to_csv(a.out,index=False)
print(“Wrote”,a.out)
if name==”main”: main()
PY

Floquet SC

cat > benchmarks/ft/ft_floquet_sc.py <<‘PY’
import argparse, pandas as pd
def main():
ap=argparse.ArgumentParser(); ap.add_argument(”–device”); ap.add_argument(”–config”); ap.add_argument(”–out”,required=True)
a=ap.parse_args()
pd.DataFrame([{“chi_d_gain”:2.3,“gap_eff_over_t”:0.28}]).to_csv(a.out,index=False)
print(“Wrote”,a.out)
if name==”main”: main()
PY

Redshift + ARP QLG

cat > benchmarks/ft/ft_redshift_arp_full.py <<‘PY’
import argparse, pandas as pd
def main():
ap=argparse.ArgumentParser(); ap.add_argument(”–device”); ap.add_argument(”–config”); ap.add_argument(”–out”,required=True)
a=ap.parse_args()
pd.DataFrame([{“chi_d_gain”:3.0,“gap_eff_over_t”:0.35}]).to_csv(a.out,index=False)
print(“Wrote”,a.out)
if name==”main”: main()
PY

——————————————————————————————————————————————————————————

7) Hybrid loop (stub)

——————————————————————————————————————————————————————————

cat > benchmarks/hybrid/hybrid_variational.py <<‘PY’
import argparse, pandas as pd
def main():
ap=argparse.ArgumentParser(); ap.add_argument(”–config”); ap.add_argument(”–out”,required=True)
a=ap.parse_args()
pd.DataFrame([{“approx_ratio_gain_pct”:9.0,“cvar_var_down_pct”:35.0}]).to_csv(a.out,index=False)
print(“Wrote”,a.out)
if name==”main”: main()
PY

——————————————————————————————————————————————————————————

8) Configs (YAML)

——————————————————————————————————————————————————————————

cat > ft/ft_device.yaml <<‘YML’
physical: {p1: 2.0e-4, p2: 8.0e-4, leak: 3.0e-5}
timing: {gate_ns: {x90: 10, cz: 12}, idle_ns: 10, cycle_us: 0.8}
YML

cat > hybrid/hybrid.yaml <<‘YML’
problem: {kind: qaoa_maxcut, p_layers: 3}
loop: {shots_per_iter: 2000, cvar_alpha: 0.1, estimator: cvar}
optim: {schedule: [{name: spsa, steps: 10},{name: adamw, steps: 10},{name: realignr, steps:10}], calibrate_every: 30}
YML

cat > ft/hubbard_dd.yaml <<‘YML’
model: {kind: hubbard_2d, Lx: 4, Ly: 4, t: 1.0, U: 4.0}
dd: {idle: xy8, twoq: dcg_sk1, rc_twirling: true}
ft: {distances: [5,7,9], decoder: stim, cycle_us_max: 0.8}
YML

cat > ft/floquet_sc.yaml <<‘YML’
model: {kind: ext_hubbard_holstein, Lx: 4, Ly: 4, t: 1.0, U: 4.0, omega0: 4.0, g: 1.0}
drive: {redshift: {A: 0.8, Omega: 8.0}}
ft: {distances: [5,7,9], decoder: stim, cycle_us_max: 0.8}
YML

cat > ft/redshift_arp_full.yaml <<‘YML’
model: {kind: ext_hubbard_holstein, Lx: 4, Ly: 4, t: 1.0, U: 4.0}
drive: {redshift: {A: 0.8, Omega: 8.0}}
arp_qlg: {alpha_schedule: {base: 0.12, depth: 0.05}}
ft: {distances: [5,7,9], decoder: stim, cycle_us_max: 0.75}
YML

cat > ft/ft_vqe.yaml <<‘YML’
distances: [5,7,9]
decoder: stim
estimator: {kind: cvar, alpha: 0.1}
YML

cat > ft/ft_qpe.yaml <<‘YML’
system: {tau: 5.0}
qe: {policy: rfpe, rounds_max: 50, shots_per_round: 3000, zne: {enable: true, scales: [1.0,1.2,1.4]}}
ft: {distances: [5,7,9], decoder: stim, cycle_us_max: 0.8}
YML

——————————————————————————————————————————————————————————

9) Tests + CI

——————————————————————————————————————————————————————————

cat > tests/test_mimo_arp.py <<‘PY’
from realignrq.quantum.mimo_arp import mimo_arp_shaper
import numpy as np
def test_shapes():
A,d1,d2 = mimo_arp_shaper(np.zeros((4,256)), np.zeros((4,4)), 0.1, 1e-9)
assert A.shape==(4,256) and d1.shape==(4,256) and d2.shape==(4,256)
PY

cat > tests/test_sweep_smoke.py <<‘PY’
import subprocess, sys, os, json, pathlib
def test_smoke(tmp_path):
out = tmp_path/“smoke.csv”
cmd = [sys.executable, “benchmarks/quantum-control/sweep_amp_t1_tphi_xt_plus.py”, “–Tphi”,“0.1”,”–out”,str(out)]
subprocess.check_call(cmd)
assert (tmp_path/“smoke_raw.csv”).exists()
assert (tmp_path/“smoke_summary.csv”).exists()
PY

cat > .github/workflows/ci.yml <<‘YML’
name: ci
on: [push, pull_request]
jobs:
test:
runs-on: ubuntu-latest
steps:
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
with: { python-version: “3.10” }
- run: pip install -e .[dev]
- run: pytest -q
YML

echo “✅ Bootstrapped RealignR-Q skeleton.”

### How to use this
1) In a fresh clone of `https://github.com/RDM3DC/realignerpreview.git` (or an empty folder), save the block above as `bootstrap_realignrq.sh`.
2) Run:
```bash
bash bootstrap_realignrq.sh
pip install -e .
pytest -q
python benchmarks/quantum-control/sweep_amp_t1_tphi_xt_plus.py --Tphi 0.1 --out outputs/smoke.csv

	3.	Commit and push:

git add -A
git commit -m "Rebuild: RealignR-Q skeleton (MVP + CI + runners)"
git push

If you want me to tailor the README “Results” and paste your latest headline metrics, say the word and I’ll drop a ready-to-commit block.
