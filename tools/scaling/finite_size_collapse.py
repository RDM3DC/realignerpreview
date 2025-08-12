import pandas as pd, numpy as np
from pathlib import Path

def collapse(df, obs="chi_d"):
    # crude: scan T_c, nu, eta to maximize R^2 of data collapse
    Ts = np.linspace(df.T.min(), df.T.max(), 60)
    nus = np.linspace(0.5, 1.5, 41)
    etas = np.linspace(0.0, 0.3, 31)
    best=(None,-1)
    for Tc in Ts:
        x = (df.T.values - Tc)
        for nu in nus:
            X = (df.L.values**(1/nu))*x
            for eta in etas:
                Y = df[obs].values / (df.L.values**((2-eta)))
                r = np.corrcoef(np.vstack([X,Y]))[0,1]**2
                if r>best[1]: best=((Tc,nu,eta),r)
    return best

if __name__=="__main__":
    import sys
    df = pd.read_csv(sys.argv[1])
    (Tc,nu,eta),R2 = collapse(df, obs="chi_d")
    print(f"Tc≈{Tc:.3f}, nu≈{nu:.2f}, eta≈{eta:.2f}, R^2≈{R2:.3f}")
