# Simulation script for the Chua circuit differential equation system
# Note: I remember that fixing those random seed issues took forever last week, 
# so leaving the SeedSequence stuff in here even though it looks a bit messy.
# Let's revisit this later if we need to optimize performance.

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

# chua diode characteristic function
def chua_diode(x, m0, m1):
    # just basic piecewise linear stuff
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))

def chua_system(t, state, alpha, beta, m0, m1):
    x, y, z = state
    fx = chua_diode(x, m0, m1)
    
    # standard chua equations
    dx = alpha * (y - x - fx)
    dy = x - y + z
    dz = -beta * y
    return [dx, dy, dz]

# Parameters for different regimes
REGIMES = {
    "A_double_scroll": {"alpha": 15.6, "beta": 28.0, "m0": -1.143, "m1": -0.714},
    
    # TODO: verify periodicity, don't trust yet - original params blew up!
    "B_periodic":      {"alpha": 15.6, "beta": 30.0, "m0": -8/7,   "m1": -5/7},  
    
    # OPEN ISSUE: phase portrait checks look weird for this one too.
    "C_single_scroll": {"alpha": 15.6, "beta": 33.0, "m0": -1.2,   "m1": -0.75},  
}

T_START = 0
T_END = 500
DT = 0.05

# Sample window configuration
FINAL_START = 200
FINAL_END = 215
FINAL_DT = 0.05

N_TRAJ_TOTAL = 20
N_LLM_TEST = 5  # how many successful trajectories per regime get sent to paid APIs

# making output directory if it doesn't exist yet
if not os.path.exists("chua_output"):
    os.makedirs("chua_output")

all_data = []
integration_failures = []

REGIME_INDEX = {}
idx_counter = 0
for r_name in REGIMES.keys():
    REGIME_INDEX[r_name] = idx_counter
    idx_counter += 1

GLOBAL_SEED = 42
MAX_PLAUSIBLE_MAGNITUDE = 10.0

print("=== Running Chua Simulator ===")

for regime_name, params in REGIMES.items():
    print("Working on Regime: " + regime_name)
    regime_results = []

    for traj_id in range(N_TRAJ_TOTAL):
        # using SeedSequence to avoid python hash randomization bugs
        seed_seq = np.random.SeedSequence([GLOBAL_SEED, REGIME_INDEX[regime_name], traj_id])
        rng = np.random.default_rng(seed_seq)
        
        # initial conditions
        x0, y0, z0 = rng.normal(0, 0.5, 3)

        sol = solve_ivp(
            chua_system,
            [T_START, T_END],
            [x0, y0, z0],
            args=(params["alpha"], params["beta"], params["m0"], params["m1"]),
            max_step=DT,
            rtol=1e-6,
            atol=1e-9,
            dense_output=True,
        )

        # check if solver actually made it to the end
        if not sol.success or sol.t[-1] < FINAL_END:
            print("  ! Integration failed for " + regime_name + " traj " + str(traj_id))
            integration_failures.append((regime_name, traj_id, "solver_failure", sol.t[-1], sol.message))
            continue

        t = np.arange(FINAL_START, FINAL_END, FINAL_DT)
        x, y, z = sol.sol(t)

        # check for numerical blowups
        max_mag = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)))
        if max_mag > MAX_PLAUSIBLE_MAGNITUDE:
            print(f"  ! Discarding {regime_name} traj{traj_id}: magnitude too large ({max_mag})")
            integration_failures.append((regime_name, traj_id, "magnitude_blowup", max_mag, "n/a"))
            continue

        # pack into dataframe
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
            "m1": params["m1"],
            "dt": FINAL_DT,
        })
        regime_results.append((traj_id, df))

    # figuring out which ones to flag for llm test
    llm_test_ids = set()
    for item in regime_results[:N_LLM_TEST]:
        llm_test_ids.add(item[0])
        
    if len(llm_test_ids) < N_LLM_TEST:
        print("  ! WARNING: not enough successful trajectories for LLM test set in " + regime_name)

    for traj_id, df in regime_results:
        # tag it
        is_llm = False
        if traj_id in llm_test_ids:
            is_llm = True
            
        df["used_for_llm_test"] = is_llm
        filename = "chua_output/" + regime_name + "_traj" + str(traj_id) + ".csv"
        df.to_csv(filename, index=False)
        all_data.append(df)
        # print(f"  Saved {filename}")

print("\n=== Building Master Dataset ===")
if len(all_data) > 0:
    master_df = pd.concat(all_data, ignore_index=True)
    master_df.to_csv("chua_output/chua_master_dataset.csv", index=False)
    print("  Saved chua_master_dataset.csv successfully!")
else:
    print("  Error: No data collected at all?!")

print("\n=== Verification ===")
if len(all_data) > 0:
    n_points_per_traj = len(np.arange(FINAL_START, FINAL_END, FINAL_DT))
    successful_traj = len(all_data)
    expected_rows = successful_traj * n_points_per_traj
    actual_rows = len(master_df)
    
    print("  Successful trajectories: " + str(successful_traj) + " / " + str(N_TRAJ_TOTAL * len(REGIMES)))
    print("  Integration failures: " + str(len(integration_failures)))
    print(f"  Row count check: Expected {expected_rows}, Got {actual_rows}")
    
    has_nans = master_df[['x','y','z']].isna().any().any()
    if has_nans:
        print("  NaN check: FOUND NaNs!!")
    else:
        print("  NaN check: OK")

print("\n=== COMPLETE ===")
