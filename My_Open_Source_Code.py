"""
Public Demonstration Code for Ultimate Pit Limit Workflow
---------------------------------------------------------

This file provides a clean, self-contained, public-friendly version
of the workflow used in the study. All data and algorithms are 
represented in a simplified form for demonstration purposes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1) Example Block Model Loading
# ============================================================

def load_block_model():
    n = 100
    df = pd.DataFrame({
        "XC": np.random.randint(0, 20, n),
        "YC": np.random.randint(0, 20, n),
        "ZC": np.random.randint(0, 20, n),
        "ROCKTYPE": np.random.randint(0, 2, n),
        "GRADE": np.random.rand(n)
    })
    return df

# ============================================================
# 2) Adding I/J/K Block Coordinates
# ============================================================

def add_ijk_coordinates(df):
    df = df.copy()
    df["I"] = df["XC"].rank(method="dense").astype(int) - 1
    df["J"] = df["YC"].rank(method="dense").astype(int) - 1
    df["K"] = df["ZC"].rank(method="dense").astype(int) - 1
    return df

# ============================================================
# 3) Slope Adjustment Procedure
# ============================================================

def apply_slope(blocks):
    blocks = blocks.copy()
    blocks["Z_adj"] = blocks["ZC"] + 0.01 * (blocks["I"] + blocks["J"])
    return blocks

# ============================================================
# 4) PSO Workflow Demonstration
# ============================================================

def pso_framework():
    max_iter = 20
    npop = 10
    particles = np.random.rand(npop, 3)
    personal_best = particles.copy()
    global_best = particles[0]

    for it in range(max_iter):
        velocity = 0.1 * (np.random.rand(*particles.shape) - 0.5)
        particles = particles + velocity
        fitness = np.random.rand(npop)

        for i in range(npop):
            if fitness[i] > fitness.mean():
                personal_best[i] = particles[i]

        best_index = np.argmax(fitness)
        global_best = particles[best_index]

        print(f"Iteration {it+1}: PSO step completed.")

    return global_best

# ============================================================
# 5) Slope Constraints Loading and Processing
# ============================================================

def load_slope_constraints(filename="Slope Constraints.xlsx"):
    df = pd.read_excel(filename)
    df_180 = df.copy()
    df_180["Azimuths"] = (df_180["Azimuths"] + 180) % 360
    df_180 = df_180.sort_values("Azimuths", ignore_index=True)
    return df, df_180

# ============================================================
# 6) Main Workflow
# ============================================================

def main():
    print("Loading example block model...")
    data = load_block_model()

    print("Adding IJK coordinates...")
    data = add_ijk_coordinates(data)

    print("Applying slope adjustment...")
    data = apply_slope(data)

    print("Loading slope constraints...")
    slope, slope_180 = load_slope_constraints()

    print("Running PSO framework demonstration...")
    best_solution = pso_framework()

    print("\n--- Workflow Completed Successfully ---")
    print("Sample of processed blocks:")
    print(data.head())
    print("Sample slope data:")
    print(slope.head())
    print("PSO Best Position:", best_solution)

if __name__ == "__main__":
    main()
