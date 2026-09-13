import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

def chua_diode(x, m0, m1):
    return m1*x + 0.5*(m0 - m1)*(np.abs(x + 1) - np.abs(x - 1))

def chua_system(t, state, alpha, beta, m0, m1):
    x, y, z = state
    fx = chua_diode(x, m0, m1)
    dx = alpha * (y - x - fx)
    dy = x - y + z
    dz = -beta * y
    return [dx, dy, dz]

REGIMES = {
    "A_double_scroll": {"alpha": 15.6, "beta": 28.0, "m0": -1.143, "m1": -0.714},
    "B_periodic":      {"alpha": 15.6, "beta": 28.0, "m0": -0.8,   "m1": -0.5},
    "C_single_scroll": {"alpha": 15.6, "beta": 33.0, "m0": -1.2,   "m1": -0.75}
}

T_START, T_END, DT = 0, 500, 0.05
FINAL_START, FINAL_END, FINAL_DT = 200, 350, 0.5
N_TRAJ = 20

np.random.seed(42)
os.makedirs("chua_output", exist_ok=True)
all_data = []

print("=== Running Chua Simulator ===")

for regime_name, params in REGIMES.items():
    print(f"Regime: {regime_name}")
    
    for traj_id in range(N_TRAJ):
        x0, y0, z0 = np.random.normal(0, 0.5, 3)
        
        sol = solve_ivp(
            chua_system,
            [T_START, T_END],
            [x0, y0, z0],
            args=(params["alpha"], params["beta"], params["m0"], params["m1"]),
            max_step=DT,
            rtol=1e-6,
            atol=1e-9,
            dense_output=True
        )
        
        t = np.arange(FINAL_START, FINAL_END, FINAL_DT)
        x, y, z = sol.sol(t)
        
        df = pd.DataFrame({
            "regime": regime_name,
            "traj_id": traj_id,
            "t": t,
            "x": x,
            "y": y,
            "z": z,
            "alpha": params["alpha"],
            "beta": params["beta"],
            "m0": params["m0"],
            "m1": params["m1"]
        })
        
        filename = f"chua_output/{regime_name}_traj{traj_id}.csv"
        df.to_csv(filename, index=False)
        all_data.append(df)
        print(f"  Saved {filename}")

print("\n=== Building Master Dataset ===")
master_df = pd.concat(all_data, ignore_index=True)
master_df.to_csv("chua_output/chua_master_dataset.csv", index=False)
print("  Saved chua_master_dataset.csv")

print("\n=== Verification ===")
expected_rows = N_TRAJ * len(REGIMES) * len(np.arange(FINAL_START, FINAL_END, FINAL_DT))
actual_rows = len(master_df)
print(f"  Expected rows: {expected_rows}, Actual: {actual_rows} {'✓' if actual_rows == expected_rows else '✗'}")

regimes_present = set(master_df["regime"])
print(f"  Regimes: {'✓' if regimes_present == set(REGIMES.keys()) else '✗'} {regimes_present}")

traj_check = all(
    set(master_df[master_df["regime"] == r]["traj_id"]) == set(range(N_TRAJ))
    for r in REGIMES
)
print(f"  Trajectory IDs: {'✓' if traj_check else '✗'}")

print("\n=== COMPLETE ===")
