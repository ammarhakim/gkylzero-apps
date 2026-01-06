import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
import os

mpl.rcParams.update({
    #"font.family": "serif",
    "font.size": 18,              # Default text size
    "axes.titlesize": 16,         # Subplot title size
    "axes.labelsize": 16,         # X and Y axis label size
    "xtick.labelsize": 16,        # X tick label size
    "ytick.labelsize": 16,        # Y tick label size
    "legend.fontsize": 16,        # Legend text size
    "figure.titlesize": 16,       # Main figure suptitle size
    "image.cmap": 'viridis',
    #"text.usetex": False,
})

eV = 1.602e-19  # Joules

# Fields to plot and their labels (now as dictionaries)
fields = {
    'profiles': {
        'elcDensAve': {'unit': 'm$^{-3}$', 'title': r'a) $n_e$'},
        'elcTempAve': {'unit': 'eV', 'title': r'b) $T_e$'},
        'ionTempAve': {'unit': 'eV', 'title': r'c) $T_i$'},
    },
    'potential': {
        'phiAve': {'unit': 'V', 'title': r'a) $\phi$'},
        'ErAve': {'unit': 'kV/m', 'title': r'b) $E_r$'},
        'VEshearAve': {'unit': '1/s', 'title': r'c) $|\gamma_E|$'},
    },
    'fluctuations': {
        'dn': {'unit': '', 'title': r'a) $\tilde{n}_{rms}/\langle{n}\rangle$'},
        'dT': {'unit': '', 'title': r'b) $\tilde{T}_{e,rms}/\langle{T_e}\rangle$'},
        'dphi': {'unit': '', 'title': r'c) $e\tilde{\phi}_{rms}/\langle{T_e}\rangle$'},
    },
    'reynolds': {
        'reynolds_stress': {'unit': r'm$^2$/s$^2$', 'title': r'a) Reynolds Stress'},
        'reynolds_force': {'unit': r'm/s$^2$', 'title': r'b) Reynolds Force'},
    }
}

def load_hdf5_field(filepath, field, xstart_idx=0, xend_idx=None):
    with h5py.File(filepath, 'r') as f:
        return f['x_vals'][xstart_idx:xend_idx], f[field][xstart_idx:xend_idx]

def plot_profiles(file_posD, file_negD, lcfs_shift, show):
    x_idx = 2
    x_pos, _ = load_hdf5_field(file_posD, 'x_vals', x_idx)
    x_neg, _ = load_hdf5_field(file_negD, 'x_vals', x_idx)
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift

    profile_fields = list(fields['profiles'].keys())
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axs):
        field = profile_fields[i]
        _, pos_data = load_hdf5_field(file_posD, field, x_idx)
        _, neg_data = load_hdf5_field(file_negD, field, x_idx)
        ax.plot(x_pos, pos_data, label='PT', color='red', linewidth=2)
        ax.plot(x_neg, neg_data, label='NT', color='blue', linewidth=2)
        ax.axvline(x=0, color='gray')
        ax.set_title(fields['profiles'][field]['title'])
        ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
        ax.set_ylabel(fields['profiles'][field]['unit'])
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

    fluct_fields = list(fields['fluctuations'].keys())
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axs):
        field = fluct_fields[i]
        _, pos_data = load_hdf5_field(file_posD, field, x_idx)
        _, neg_data = load_hdf5_field(file_negD, field, x_idx)
        ax.plot(x_pos, pos_data, label='PT', color='red', linewidth=2)
        ax.plot(x_neg, neg_data, label='NT', color='blue', linewidth=2)
        ax.axvline(x=0, color='gray')
        ax.set_title(fields['fluctuations'][field]['title'])
        ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
        ax.set_ylabel(fields['fluctuations'][field]['unit'])
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

    pot_fields = list(fields['potential'].keys())
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axs):
        field = pot_fields[i]
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
        ax.set_title(fields['potential'][field]['title'])
        ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
        ax.set_ylabel(fields['potential'][field]['unit'])
        ax.legend()
    plt.tight_layout()
    plt.savefig('phi-Ve-gamE_from_hdf5.pdf')
    if show:
        plt.show()
    else:
        plt.close()

def plot_reynolds_stress(file_posD, file_negD, lcfs_shift, show):
    x_idx = 10
    reyn_fields = list(fields['reynolds'].keys())

    x_pos, stress_pos = load_hdf5_field(file_posD, reyn_fields[0], x_idx)
    x_neg, stress_neg = load_hdf5_field(file_negD, reyn_fields[0], x_idx)
    _, force_pos = load_hdf5_field(file_posD, reyn_fields[1], x_idx)
    _, force_neg = load_hdf5_field(file_negD, reyn_fields[1], x_idx)

    x_pos -= lcfs_shift
    x_neg -= lcfs_shift

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    ax = axs[0]
    ax.plot(x_pos, stress_pos, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, stress_neg, label='NT', color='blue', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(fields['reynolds'][reyn_fields[0]]['title'])
    ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
    ax.set_ylabel(r'$\langle \delta v_r \delta v_y \rangle$ (' + fields['reynolds'][reyn_fields[0]]['unit'] + ')')
    ax.legend()

    ax = axs[1]
    ax.plot(x_pos, force_pos, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, force_neg, label='NT', color='blue', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(fields['reynolds'][reyn_fields[1]]['title'])
    ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
    ax.set_ylabel(r'$-\nabla_r \langle \delta v_r \delta v_y \rangle$ (' + fields['reynolds'][reyn_fields[1]]['unit'] + ')')
    ax.legend()

    plt.tight_layout()
    plt.savefig('reynolds_stress_from_hdf5.pdf')
    if show:
        plt.show()
    else:
        plt.close()

def plot_reynolds_and_shear(file_posD, file_negD, lcfs_shift, show):
    x_idx = 10
    reyn_fields = list(fields['reynolds'].keys())
    pot_fields = list(fields['potential'].keys())

    x_pos, stress_pos = load_hdf5_field(file_posD, reyn_fields[0], x_idx)
    x_neg, stress_neg = load_hdf5_field(file_negD, reyn_fields[0], x_idx)
    _, Er = load_hdf5_field(file_posD, pot_fields[1], x_idx)
    _, Er_neg = load_hdf5_field(file_negD, pot_fields[1], x_idx)
    _, VEshear_pos = load_hdf5_field(file_posD, pot_fields[2], x_idx)
    _, VEshear_neg = load_hdf5_field(file_negD, pot_fields[2], x_idx)

    x_pos -= lcfs_shift
    x_neg -= lcfs_shift

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    ax = axs[0]
    ax.plot(x_pos, stress_pos, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, stress_neg, label='NT', color='blue', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(fields['reynolds'][reyn_fields[0]]['title'])
    ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
    ax.set_ylabel(r'$\langle \delta v_r \delta v_y \rangle$ (' + fields['reynolds'][reyn_fields[0]]['unit'] + ')')
    ax.legend()

    ax = axs[1]
    ax.plot(x_pos, Er/1e3, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, Er_neg/1e3, label='NT', color='blue', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(fields['potential'][pot_fields[1]]['title'])
    ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
    ax.set_ylabel(fields['potential'][pot_fields[1]]['unit'])
    ax.legend()

    ax = axs[2]
    ax.plot(x_pos, np.abs(VEshear_pos), label='PT', color='red', linewidth=2)
    ax.plot(x_neg, np.abs(VEshear_neg), label='NT', color='blue', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(fields['potential'][pot_fields[2]]['title'])
    ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
    ax.set_ylabel(fields['potential'][pot_fields[2]]['unit'])
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
