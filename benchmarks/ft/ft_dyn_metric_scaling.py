import argparse, yaml, pathlib, pandas as pd
from copy import deepcopy
from realignrq.ft.surface_code_sweep import simulate_logical_error  # placeholder
# assume you already have ft_redshift_arp_full-style functions wired


def run_point(cfg, L, T):
    # build per-point config
    pcfg = deepcopy(cfg)
    pcfg["model"]["Lx"] = pcfg["model"]["Ly"] = L
    pcfg["baths"]["T_grid"] = [T]
    # >>> call your existing FT runner here (redshift_arp_full path) <<<
    # Return minimal metrics dict per (L,T)
    return {"L":L,"T":T,"chi_d":2.0,"rho_s":0.3,"gap_over_t":0.5,"binder_U4":0.61,
            "pL":2e-11,"cycle_us":0.74,"shots_x":3.2}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config))
    outdir=pathlib.Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for L in cfg["scaling"]["sizes"]:
        for T in cfg["scaling"]["temps"]:
            rows.append(run_point(cfg, L, T))
    pd.DataFrame(rows).to_csv(outdir/"scaling_raw.csv", index=False)
    print("Wrote", outdir/"scaling_raw.csv")


if __name__=="__main__":
    main()
