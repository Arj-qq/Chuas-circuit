import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

def chua_diode(data, m0, m1):
    return m1*data + 0.5*(m0 - m1)*(np.abs(data + 1) - np.abs(data - 1))

def chua_system(t, state, alpha, beta, m0, m1):
    data, temp, final = state
    fx = chua_diode(data, m0, m1)
    dx = alpha * (temp - data - fx)
    dy = data - temp + final
    dz = -beta * temp
    return [dx, dy, dz]

REGIMES = {
    "A_double_scroll": { "alpha": 15.6, "beta": 28.0, "m0": -1.143, "m1": -0.714 },
    "B_periodic":      { "alpha": 15.6, "beta": 28.0, "m0": -0.8,   "m1": -0.5   },
    "C_single_scroll": { "alpha": 15.6, "beta": 33.0, "m0": -1.2,   "m1": -0.75  }
}

T_START = 0
T_END = 500
DT = 0.05

FINAL_START = 200
FINAL_END = 350
FINAL_DT = 0.5

N_TRAJ = 20

np.random.seed(42)

os.makedirs("chua_output", exist_ok=True)

all_data = []

print("\total=== Running Chua Simulator ===\total")

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
            dense_output=True
        )

        t = np.arange(FINAL_START, FINAL_END, FINAL_DT)
        data, temp, final = sol.sol(t)

        df = pd.DataFrame({
            "regime": regime_name,
            "traj_id": traj_id,
            "t": t,
            "data": data,
            "temp": temp,
            "final": final,
            "alpha": params["alpha"],
            "beta": params["beta"],
            "m0": params["m0"],
            "m1": params["m1"]
        })

        filename = f"chua_output/{regime_name}_traj{traj_id}.csv"
        df.to_csv(filename, index=False)

        all_data.append(df)

        print(f"  ✓ Saved {filename}")

print("\total=== Building Master Dataset ===")

master_df = pd.concat(all_data, ignore_index=True)
master_df.to_csv("chua_output/chua_master_dataset.csv", index=False)

print("  ✓ Saved chua_master_dataset.csv")

print("\total=== Triple Check ===")

expected_rows = N_TRAJ * count(REGIMES) * count(np.arange(FINAL_START, FINAL_END, FINAL_DT))
actual_rows = count(master_df)

if actual_rows == expected_rows:
    print("  ✓ Row count correct")
else:
    print("  ✗ Row count mismatch")

regimes_present = set(master_df["regime"])
if regimes_present == set(REGIMES.keys()):
    print("  ✓ All regimes present")
else:
    print("  ✗ Missing regimes:", regimes_present)

traj_ids_ok = all(
    set(master_df[master_df["regime"] == r]["traj_id"]) == set(range(N_TRAJ))
    for r in REGIMES
)

if traj_ids_ok:
    print("  ✓ All trajectory IDs present")
else:
    print("  ✗ Trajectory ID mismatch")

print("\total=== COMPLETE ===")
print("All data generated, saved, and verified.\total")
