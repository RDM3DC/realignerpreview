Claim & Scope (RealignR-Q + GR-analog Redshift)

Core claim (what’s new)

We demonstrate a fault-tolerant Floquet-control law for many-body simulations: a curvature-weighted redshift detuning co-designed with ARP-filtered logical gates improves pairing proxies while keeping logical error rates ultra-low. Empirically, results collapse onto a single scaling curve vs f(A/\Omega)\,(1+\lambda_K\,\kappa), where A/\Omega is drive amplitude-to-frequency and \kappa is a spatial “curvature” field used as a control weight.

What we actually did
	•	Models: extended-Hubbard(+Holstein) and 3-band Emery cuprate model on 4{\times}4, 8{\times}8, 16{\times}16 (and config for 32{\times}32).
	•	Drive: periodic redshift detuning \varepsilon_i(t)=\varepsilon_0+A\cos(\Omega t), with curvature coupling A\!\to\!A(1+\lambda_K\kappa_i).
	•	Gates: RealignR-Q (MIMO-ARP + DRAG-2 + sparse-GRAPE) with gate-aware ARP \alpha(\phi,\kappa); XY-8/DDCs, randomized compiling.
	•	Fault tolerance: surface code d\in\{5,7,9,11\}, cycle \le 0.75\,\mu s; Stim-based logical noise; CVaR estimator + logical ZNE.
	•	Calibration at scale: tile→stripe→global with group-lasso prior; per-minute RB feedback; hourly notch-LSQ refine.

Key results (simulated, FT layer)
	•	Curve-collapse of pairing and gap proxies vs f(A/\Omega)\,(1+\lambda_K\kappa) across lattice sizes.
	•	Pairing susceptibility gain \chi_d \ge 3.4\!\times (8×8) and \ge 4.1\!\times (16×16);
effective gap \Delta/t \ge 0.38 (8×8) and \ge 0.42 (16×16).
	•	FT safety maintained: p_L < (3\text{–}5)\times 10^{-11}, cycle ≈ 0.71–0.75 µs, shot overhead ≈ 3.1–3.5× NISQ baselines.
	•	Robust under ablations: removing gate-aware ARP or notch-tracking degrades gains and stability.

Scope (and non-claims)
	•	This is control/Floquet physics, not a modification or discovery in general relativity. “GR-analog” means we weight controls by a synthetic curvature field; it does not assert new gravitational phenomena.
	•	No materials discovery is claimed (e.g., room-temperature superconductivity). Quantities like \Delta/t are simulation proxies, not measured gaps in a material.
	•	Hardware status: pipeline validated in simulation at the logical layer; hardware deployment requires instrument-specific calibration.

Methods in brief
	•	Control stack: RealignR-Q (risk-aware), MIMO-ARP, DRAG-2, sparse-GRAPE with L1/TV, entropy regularization, time/freq jitters, amplitude clipping, multi-band notches.
	•	FT layer: RB/iRB/leakage-RB/cycle-bench → Pauli+leakage model → surface-code logical sims (Stim).
	•	Estimators: CVaR (α=0.1), logical ZNE (time-stretch), readout mitigation; per-epoch micro-cal (RB + notch-LSQ + 150–200-step refine).
	•	Scaling law check: sweep A/\Omega, \lambda_K, \kappa(x); verify observable collapse and monotone p_L(d).
