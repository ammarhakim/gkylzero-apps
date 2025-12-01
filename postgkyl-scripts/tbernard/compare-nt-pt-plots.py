import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
import os

mpl.rcParams.update({
    #"font.family": "serif",
    "font.size": 18,              # Default text size
    "axes.titlesize": 20,         # Subplot title size
    "axes.labelsize": 20,         # X and Y axis label size
    "xtick.labelsize": 16,        # X tick label size
    "ytick.labelsize": 16,        # Y tick label size
    "legend.fontsize": 16,        # Legend text size
    "figure.titlesize": 22,       # Main figure suptitle size
    "image.cmap": 'viridis',
    #"text.usetex": False,
})

# Fields to plot and their labels
fields = [
    'elcDensAve', 'elcTempAve', 'ionTempAve', 
    'phiAve', 'ErAve', 'VEshearAve', 
    'dn_norm', 'dT_norm', 'dphi_norm',
    'reynolds_stress', 'reynolds_force'
]
units = [
    'm$^{-3}$', 'eV', 'eV', 'V', 'kV/m', '1/s', '', '', '',
    r'm$^2$/s$^2$', r'm/s$^2$'  
]
titles = [
    r'a) $n_e$', r'b) $T_e$', r'c) $T_i$', 
    r'a) $\phi$', r'b) $E_r$', r'c) $|\gamma_E|$',
    r'a) $\tilde{n}_{rms}/\langle{n}\rangle$', r'b) $\tilde{T}_{e,rms}/\langle{T_e}\rangle$',
    r'c) $e\tilde{\phi}_{rms}/\langle{T_e}\rangle$',
    r'a) Reynolds Stress', r'b) Reynolds Force'
]

def load_hdf5_field(filepath, field, xstart_idx=0, xend_idx=None):
    with h5py.File(filepath, 'r') as f:
        return f['x_vals'][xstart_idx:xend_idx], f[field][xstart_idx:xend_idx]

def plot_profiles(file_posD, file_negD, lcfs_shift, show):
    x_idx = 2
    x_pos, _ = load_hdf5_field(file_posD, 'x_vals', x_idx)
    x_neg, _ = load_hdf5_field(file_negD, 'x_vals', x_idx)
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axs):
        field = fields[i]  # n_e, T_e, T_i
        _, pos_data = load_hdf5_field(file_posD, field, x_idx)
        _, neg_data = load_hdf5_field(file_negD, field, x_idx)
        ax.plot(x_pos, pos_data, label='PT', color='red', linewidth=2)
        ax.plot(x_neg, neg_data, label='NT', color='blue', linewidth=2)
        ax.axvline(x=0, color='gray')
        ax.set_title(titles[i])
        ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
        ax.set_ylabel(units[i])
        ax.set_xlim(-0.1, 0.05)
        ax.legend()
    plt.tight_layout()
    plt.savefig('ne-Te-Ti_from_hdf5.pdf')
    if show:
        plt.show()
    else:
        plt.close()

def plot_fluctuations(file_posD, file_negD, lcfs_shift, show):
    x_idx = 2
    x_pos, _ = load_hdf5_field(file_posD, 'x_vals', x_idx)
    x_neg, _ = load_hdf5_field(file_negD, 'x_vals', x_idx)
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axs):
        field = fields[i + 6]
        _, pos_data = load_hdf5_field(file_posD, field, x_idx)
        _, neg_data = load_hdf5_field(file_negD, field, x_idx)
        ax.plot(x_pos, pos_data, label='PT', color='red', linewidth=2)
        ax.plot(x_neg, neg_data, label='NT', color='blue', linewidth=2)
        ax.axvline(x=0, color='gray')
        ax.set_title(titles[i + 6])
        ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
        ax.set_ylabel(units[i + 6])
        ax.set_xlim(-0.1, 0.05)
        ax.legend()
    plt.tight_layout()
    plt.savefig('dn-dT-dphi_from_hdf5.pdf')
    if show:
        plt.show()
    else:
        plt.close()

def plot_potential_fields(file_posD, file_negD, lcfs_shift, show):
    x_idx = 10
    x_pos, _ = load_hdf5_field(file_posD, 'x_vals', x_idx)
    x_neg, _ = load_hdf5_field(file_negD, 'x_vals', x_idx)
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axs):
        field = fields[i + 3]
        _, pos_data = load_hdf5_field(file_posD, field, x_idx)
        _, neg_data = load_hdf5_field(file_negD, field, x_idx)
        if field == 'ErAve':
            pos_data /= 1e3
            neg_data /= 1e3
            ax.axhline(y=0, color='black', linestyle='--')
        elif field == 'VEshearAve':
            pos_data = np.abs(pos_data)
            neg_data = np.abs(neg_data)
        ax.plot(x_pos, pos_data, label='PT', color='red', linewidth=2)
        ax.plot(x_neg, neg_data, label='NT', color='blue', linewidth=2)
        ax.axvline(x=0, color='gray')
        ax.set_title(titles[i + 3])
        ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
        ax.set_ylabel(units[i + 3])
        ax.legend()
    plt.tight_layout()
    plt.savefig('phi-Ve-gamE_from_hdf5.pdf')
    if show:
        plt.show()
    else:
        plt.close()

# [unchanged imports and setup above]

def plot_fluxes(file_posD, file_negD, lcfs_shift, show):
    x_pos, _ = load_hdf5_field(file_posD, 'x_vals')
    x_neg, _ = load_hdf5_field(file_negD, 'x_vals')
    Nx = len(x_pos)
    lcfs_idx = int(Nx * 2 / 3)
    x_idx = 10

    Gx_field, Qxe_field, Qxi_field, Qpara_field, ne_field = 'Gx', 'Qxe', 'Qxi', 'QparaAve', 'elcDensAve'

    x_pos, Gx_pos = load_hdf5_field(file_posD, Gx_field, x_idx)
    x_neg, Gx_neg = load_hdf5_field(file_negD, Gx_field, x_idx)
    _, Qxe_pos = load_hdf5_field(file_posD, Qxe_field, x_idx)
    _, Qxe_neg = load_hdf5_field(file_negD, Qxe_field, x_idx)
    _, Qxi_pos = load_hdf5_field(file_posD, Qxi_field, x_idx)
    _, Qxi_neg = load_hdf5_field(file_negD, Qxi_field, x_idx)
    _, ne_pos = load_hdf5_field(file_posD, ne_field, x_idx)
    _, ne_neg = load_hdf5_field(file_negD, ne_field, x_idx)
    _, ne_pos_sol = load_hdf5_field(file_posD, ne_field, lcfs_idx, -5)
    _, ne_neg_sol = load_hdf5_field(file_negD, ne_field, lcfs_idx, -5)
    x_sol, Qpara_pos = load_hdf5_field(file_posD, Qpara_field, lcfs_idx, -5)
    x_sol, Qpara_neg = load_hdf5_field(file_negD, Qpara_field, lcfs_idx, -5)
    
    # Shift x-axis to LCFS
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift
    x_sol -= lcfs_shift     

    # Normalize by electron density
    Gx_pos /= ne_pos
    Gx_neg /= ne_neg
    Qxe_pos /= ne_pos
    Qxe_neg /= ne_neg
    Qxi_pos /= ne_pos
    Qxi_neg /= ne_neg
    Qpara_pos /= -ne_pos_sol
    Qpara_neg /= -ne_neg_sol

    fig = plt.figure(figsize=(14, 4))

    ax = fig.add_subplot(131)
    ax.set_title('a) Normalized particle flux')
    ax.axvline(x=0, color='gray')
    ax.plot(x_pos, Gx_pos, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, Gx_neg, label='NT', color='blue', linewidth=2)
    ax.set_xlabel(r'$R-R_{LCFS}$ (m)')
    ax.set_ylabel(r'$\Gamma_r / n_e$ (m/s)')
    ax.legend()

    ax = fig.add_subplot(132)
    ax.set_title('b) Normalized perpendicular heat flux')
    ax.axvline(x=0, color='gray')
    ax.plot(x_pos, Qxe_pos + Qxi_pos, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, Qxe_neg + Qxi_neg, label='NT', color='blue', linewidth=2)
    ax.set_xlabel(r'$R-R_{LCFS}$ (m)')
    ax.set_ylabel(r'$(Q_{\perp,e} + Q_{\perp,i}) / n_e$ (W s)')
    ax.legend()


    ax = fig.add_subplot(133)
    ax.set_title('c) Normalized parallel heat flux (at limiter)')
    ax.axvline(x=0, color='gray')
    ax.plot(x_sol, Qpara_pos, label='PT', color='red', linewidth=2)
    ax.plot(x_sol, Qpara_neg, label='NT', color='blue', linewidth=2)
    ax.set_xlabel(r'$R-R_{LCFS}$ (m)')
    ax.set_ylabel(r'$(Q_{\parallel,e} + Q_{\parallel,i}) / n_e$ (W s)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('Gr-Qr-Qpar_normed_by_ne.pdf')
    if show:
        plt.show()
    else:
        plt.close()

def plot_reynolds_stress(file_posD, file_negD, lcfs_shift, show):
    """
    Loads and plots the Reynolds stress and Reynolds force from HDF5 files.
    """
    x_idx = 10  # Define the starting index for plotting

    # --- Load Data ---
    x_pos, stress_pos = load_hdf5_field(file_posD, 'reynolds_stress', x_idx)
    x_neg, stress_neg = load_hdf5_field(file_negD, 'reynolds_stress', x_idx)
    
    _, force_pos = load_hdf5_field(file_posD, 'reynolds_force', x_idx)
    _, force_neg = load_hdf5_field(file_negD, 'reynolds_force', x_idx)

    # Shift x-axis to be relative to the LCFS
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift
    
    # --- Create Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    
    # Panel a) Reynolds Stress
    ax = axs[0]
    ax.plot(x_pos, stress_pos, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, stress_neg, label='NT', color='blue', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(titles[9]) # 'a) Reynolds Stress'
    ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
    ax.set_ylabel(r'$\langle \delta v_r \delta v_y \rangle$ (' + units[9] + ')')
    ax.legend()
    
    # Panel b) Reynolds Force
    ax = axs[1]
    ax.plot(x_pos, force_pos, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, force_neg, label='NT', color='blue', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(titles[10]) # 'b) Reynolds Force'
    ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
    ax.set_ylabel(r'$-\nabla_r \langle \delta v_r \delta v_y \rangle$ (' + units[10] + ')')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('reynolds_stress_from_hdf5.pdf')
    if show:
        plt.show()
    else:
        plt.close()

def plot_reynolds_and_shear(file_posD, file_negD, lcfs_shift, show):
    """
    Loads and plots the Reynolds stress and Reynolds force from HDF5 files.
    """
    x_idx = 10  # Define the starting index for plotting

    # --- Load Data ---
    x_pos, stress_pos = load_hdf5_field(file_posD, 'reynolds_stress', x_idx)
    x_neg, stress_neg = load_hdf5_field(file_negD, 'reynolds_stress', x_idx)
    
    _, Er = load_hdf5_field(file_posD, 'ErAve', x_idx)
    _, Er_neg = load_hdf5_field(file_negD, 'ErAve', x_idx)
    _, VEshear_pos = load_hdf5_field(file_posD, 'VEshearAve', x_idx)
    _, VEshear_neg = load_hdf5_field(file_negD, 'VEshearAve', x_idx)

    # Shift x-axis to be relative to the LCFS
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift
    
    # --- Create Plot ---
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    
    # Panel a) Reynolds Stress
    ax = axs[0]
    ax.plot(x_pos, stress_pos, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, stress_neg, label='NT', color='blue', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(titles[9]) # 'a) Reynolds Stress'
    ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
    ax.set_ylabel(r'$\langle \delta v_r \delta v_y \rangle$ (' + units[9] + ')')
    ax.legend()

    # Panel b) Er
    ax = axs[1]
    ax.plot(x_pos, Er/1e3, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, Er_neg/1e3, label='NT', color='blue', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(titles[4]) # 'b) $E_r$'
    ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
    ax.set_ylabel(units[4])
    ax.legend() 

    # Panel c) VEshear
    ax = axs[2]
    ax.plot(x_pos, np.abs(VEshear_pos), label='PT', color='red', linewidth=2)
    ax.plot(x_neg, np.abs(VEshear_neg), label='NT', color='blue', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(titles[5]) # 'c) $|\gamma_E|$'
    ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
    ax.set_ylabel(units[5])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('reynolds_and_shear_from_hdf5.pdf')
    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare diagnostics from two HDF5 simulations.")
    parser.add_argument("file_posD", help="HDF5 file path for positive D simulation")
    parser.add_argument("file_negD", help="HDF5 file path for negative D simulation")
    parser.add_argument("--lcfs", type=float, default=0.10, help="LCFS radial location (in meters) to shift x-axis")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    plot_profiles(args.file_posD, args.file_negD, args.lcfs, args.show)
    plot_fluctuations(args.file_posD, args.file_negD, args.lcfs, args.show)
    plot_potential_fields(args.file_posD, args.file_negD, args.lcfs, args.show)
    plot_fluxes(args.file_posD, args.file_negD, args.lcfs, args.show)
    plot_reynolds_stress(args.file_posD, args.file_negD, args.lcfs, args.show)
    plot_reynolds_and_shear(args.file_posD, args.file_negD, args.lcfs, args.show)

if __name__ == "__main__":
    main()
