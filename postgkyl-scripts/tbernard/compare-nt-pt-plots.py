import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Fields to plot and their labels
fields = ['elcDensAve', 'elcTempAve', 'ionTempAve', 'phiAve', 'ErAve', 'VEshearAve', 'dn_norm', 'dT_norm', 'dphi_norm']
units = ['m$^{-3}$', 'eV', 'eV', 'V', 'kV/m', '1/s', '', '', '']
titles = [
    r'a) $n_e$', r'b) $T_e$', r'c) $T_i$', r'a) $\phi$', r'b) $E_r$', r'c) $|\gamma_E|$',
    r'a) $\tilde{n}_{rms}/\langle{n}\rangle$', r'b) $\tilde{T}_{e,rms}/\langle{T_e}\rangle$',
    r'c) $e\tilde{\phi}_{rms}/\langle{T_e}\rangle$'
]

def load_hdf5_field(filepath, field):
    with h5py.File(filepath, 'r') as f:
        return f['x_vals'][:], f[field][:]

def plot_profiles(file_posD, file_negD, lcfs_shift, show):
    x_pos, _ = load_hdf5_field(file_posD, 'x_vals')
    x_neg, _ = load_hdf5_field(file_negD, 'x_vals')
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axs):
        field = fields[i]  # n_e, T_e, T_i
        _, pos_data = load_hdf5_field(file_posD, field)
        _, neg_data = load_hdf5_field(file_negD, field)
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
    x_pos, _ = load_hdf5_field(file_posD, 'x_vals')
    x_neg, _ = load_hdf5_field(file_negD, 'x_vals')
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axs):
        field = fields[i + 6]
        _, pos_data = load_hdf5_field(file_posD, field)
        _, neg_data = load_hdf5_field(file_negD, field)
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
    x_pos, _ = load_hdf5_field(file_posD, 'x_vals')
    x_neg, _ = load_hdf5_field(file_negD, 'x_vals')
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift

    x_pos = x_pos[10:]
    x_neg = x_neg[10:]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axs):
        field = fields[i + 3]
        _, pos_data = load_hdf5_field(file_posD, field)
        _, neg_data = load_hdf5_field(file_negD, field)
        pos_data = pos_data[10:]
        neg_data = neg_data[10:]
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
    x_pos -= lcfs_shift
    x_neg -= lcfs_shift

    Gx_field, Qxe_field, Qxi_field, ne_field = 'Gx', 'Qxe', 'Qxi', 'elcDensAve'

    _, Gx_pos = load_hdf5_field(file_posD, Gx_field)
    _, Gx_neg = load_hdf5_field(file_negD, Gx_field)
    _, Qxe_pos = load_hdf5_field(file_posD, Qxe_field)
    _, Qxe_neg = load_hdf5_field(file_negD, Qxe_field)
    _, Qxi_pos = load_hdf5_field(file_posD, Qxi_field)
    _, Qxi_neg = load_hdf5_field(file_negD, Qxi_field)
    _, ne_pos = load_hdf5_field(file_posD, ne_field)
    _, ne_neg = load_hdf5_field(file_negD, ne_field)

    # Normalize by electron density
    Gx_pos /= ne_pos
    Gx_neg /= ne_neg
    Qxe_pos /= ne_pos
    Qxe_neg /= ne_neg
    Qxi_pos /= ne_pos
    Qxi_neg /= ne_neg

    # Crop to x_pos[10:]
    x_pos = x_pos[10:]
    x_neg = x_neg[10:]
    Gx_pos = Gx_pos[10:]
    Gx_neg = Gx_neg[10:]
    Qxe_pos = Qxe_pos[10:]
    Qxe_neg = Qxe_neg[10:]
    Qxi_pos = Qxi_pos[10:]
    Qxi_neg = Qxi_neg[10:]

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
    ax.set_title('b) Normalized heat flux')
    ax.axvline(x=0, color='gray')
    ax.plot(x_pos, Qxe_pos, label='elc PT', color='red', linewidth=2)
    ax.plot(x_pos, Qxi_pos, label='ion PT', color='red', linestyle='--', linewidth=2)
    ax.plot(x_neg, Qxe_neg, label='elc NT', color='blue', linewidth=2)
    ax.plot(x_neg, Qxi_neg, label='ion NT', color='blue', linestyle='--', linewidth=2)
    ax.set_xlabel(r'$R-R_{LCFS}$ (m)')
    ax.set_ylabel(r'$Q_\perp / n_e$ (W s)')
    ax.legend()

    ax = fig.add_subplot(133)
    ax.set_title('c) Total normalized heat flux')
    ax.axvline(x=0, color='gray')
    ax.plot(x_pos, Qxe_pos + Qxi_pos, label='PT', color='red', linewidth=2)
    ax.plot(x_neg, Qxe_neg + Qxi_neg, label='NT', color='blue', linewidth=2)
    ax.set_xlabel(r'$R-R_{LCFS}$ (m)')
    ax.set_ylabel(r'$(Q_{\parallel,e} + Q_{\parallel,i}) / n_e$ (W s)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('Gr-Qr-Qpar_normed_by_ne.pdf')
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

if __name__ == "__main__":
    main()
