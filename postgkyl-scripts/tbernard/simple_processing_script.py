import os
import glob
import numpy as np
import postgkyl as pg
import pandas as pd
import matplotlib.pyplot as plt

# --- Constants & Settings ---
me = 9.10938188e-31
eV = 1.602e-19
fstart, fend = 400, 500  # Time frame range
z_idx = 1  # Z-slice index (e.g., 1 for Outer Mid-Plane in some setups)
plot_results = True # Change to True to enable plotting

def get_data(filename, comp):
    """Helper: Loads Gkeyll data and interpolates to grid centers."""
    data = pg.data.GData(filename)
    # Interpolate (poly_order=1, basis='ms')
    interp = pg.data.GInterpModal(data, 1, 'ms')
    grid, values = interp.interpolate(comp)
    # Return grid list and squeeze values to remove empty dimensions
    return grid, values.squeeze()

def main():
    if plot_results:
        delta = input('PT / NT? ') # 'PT' or 'NT' for different triangularity
    
    # 1. Find file prefix automatically
    files = glob.glob("*-field_0.gkyl")
    if not files: raise FileNotFoundError("No Gkeyll field files found.")
    prefix = files[0].replace("-field_0.gkyl", "")
    print(f"Processing {prefix} from frame {fstart} to {fend}...")

    # 2. Data Containers
    n_stack, T_stack, phi_stack = [], [], []

    # 3. Loop over time frames
    for tf in range(fstart, fend + 1, 10):
        try:
            # Load Density (Comp 0)
            grid, n_vals = get_data(f"{prefix}-elc_BiMaxwellianMoments_{tf}.gkyl", 0)
            
            # Load Temp (Comp 2=Par, 3=Perp) & Calculate Isotropic T (eV)
            _, Tpar = get_data(f"{prefix}-elc_BiMaxwellianMoments_{tf}.gkyl", 2)
            _, Tperp = get_data(f"{prefix}-elc_BiMaxwellianMoments_{tf}.gkyl", 3)
            T_vals = (Tpar + 2 * Tperp) / 3.0 * (me / eV)

            # Load Potential (Comp 0)
            _, phi_vals = get_data(f"{prefix}-field_{tf}.gkyl", 0)

            z_idx = len(grid[2]) // 2  # Mid-plane index
            # Slice specific Z-plane (Assumes 3D: X, Y, Z)
            # If data is 3D, slice at z_idx. If 2D, take all.
            if n_vals.ndim == 3:
                n_slice = n_vals[:, :, z_idx]
                T_slice = T_vals[:, :, z_idx]
                phi_slice = phi_vals[:, :, z_idx]
            else:
                n_slice, T_slice, phi_slice = n_vals, T_vals, phi_vals

            n_stack.append(n_slice)
            T_stack.append(T_slice)
            phi_stack.append(phi_slice)

        except Exception as e:
            print(f"Skipping frame {tf}: {e}")

    # 4. Convert to Arrays (Shape: Time, X, Y)
    n_all = np.array(n_stack)
    T_all = np.array(T_stack)
    phi_all = np.array(phi_stack)

    # 5. Calculate Fluctuations
    # Flatten Time and Y dimensions to calculate statistics vs Radial position (X)
    # New Shape: (Total_Samples, X)
    Nx = n_all.shape[1]
    n_flat = n_all.transpose(0, 2, 1).reshape(-1, Nx)
    T_flat = T_all.transpose(0, 2, 1).reshape(-1, Nx)
    phi_flat = phi_all.transpose(0, 2, 1).reshape(-1, Nx)

    x_vals = grid[0] + 0.5 * (grid[0][1] - grid[0][0])  # Centered X values
    x_vals = x_vals[:Nx]  # Ensure matching size

    # Time-Averaged Profiles (Mean)
    n_mean = np.mean(n_flat, axis=0)
    T_mean = np.mean(T_flat, axis=0)
    phi_mean = np.mean(phi_flat, axis=0)

    # RMS Fluctuations (Std Dev). To normalize, uncomment "/ Mean "
    # dn or dn/n
    sim_dn_rms = np.std(n_flat, axis=0) #/ np.abs(n_mean)
    # dT or dT/T
    sim_dT_rms = np.std(T_flat, axis=0) #/ np.abs(T_mean)
    # dPhi or e*dPhi/Te (Normalized by Te)
    sim_dVf_rms = np.std(phi_flat - 3.18*T_flat, axis=0) #/ np.abs(T_mean)

    # sim_dVf_rms = sim_dphi_rms - 3.18*sim_dT_rms

    if plot_results:
        
        sim_x = x_vals - 0.10 # Shift to R-Rsep=0 (example)

        # --- 3. Plotting ---
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

        # Density Fluctuation Plot
        axs[0].plot(sim_x, sim_dn_rms, label='Simulation (Gkeyll)', color='black', linewidth=2)
        #axs[0].plot(exp_x, exp_dn_rms, label='Experiment', color='red', marker='o', linestyle='None', markersize=4)
        axs[0].set_ylabel(r'$\delta n_{rms} \ (m^{-3})$', fontsize=12)
        axs[0].set_title('Density Fluctuations (RMS) for '+delta)
        axs[0].legend()
        #axs[0].set_ylim(0, 1e19)
        axs[0].grid(True, alpha=0.3)

        # Temperature Fluctuation Plot
        axs[1].plot(sim_x, sim_dT_rms, label='Simulation (Gkeyll)', color='green', linewidth=2)
        #axs[1].plot(exp_x, exp_dT_rms, label='Experiment
        axs[1].set_ylabel(r'$\delta T_{rms} \ (eV)$', fontsize=12)
        axs[1].set_title('Temperature Fluctuations (RMS) for '+delta)
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)    

        # Potential Fluctuation Plot
        axs[2].plot(sim_x[:-5], sim_dVf_rms[:-5], label='Simulation (Gkeyll)', color='blue', linewidth=2)
        #axs[1].plot(exp_x, exp_dphi_rms, label='Experiment', color='orange', marker='s', linestyle='None', markersize=4)
        axs[2].set_ylabel(r'$\delta \phi_{rms} \ (V)$', fontsize=12)
        axs[2].set_xlabel(r'$R - R_{sep} \ (m)$', fontsize=12)
        axs[2].set_title('Potential Fluctuations (RMS) for '+delta)
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.savefig(f"{prefix}_fluctuations_comparison.png", dpi=300)

if __name__ == "__main__":
    main()