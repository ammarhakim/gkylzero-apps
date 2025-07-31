import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

# Define all available fields, units, and titles (can be overridden by input)
all_fields = {
    'elcDensAve':  ('m$^{-3}$', r'$n_e$'),
    'elcTempAve':  ('eV',       r'$T_e$'),
    'ionTempAve':  ('eV',       r'$T_i$'),
    'phiAve':      ('V',        r'$\phi$'),
    'ErAve':       ('kV/m',     r'$E_r$'),
    'VEshearAve':  ('1/s',      r'$|\gamma_E|$'),
    'dn_norm':     ('',         r'$\tilde{n}_{rms}/\langle{n}\rangle$'),
    'dT_norm':     ('',         r'$\tilde{T}_{e,rms}/\langle{T_e}\rangle$'),
    'dphi_norm':   ('',         r'$e\tilde{\phi}_{rms}/\langle{T_e}\rangle$')
}

def load_hdf5_field(filepath, field):
    with h5py.File(filepath, 'r') as f:
        return f['x_vals'][:], f[field][:]

def plot_fields(files, labels, fields, lcfs_shift, output_name, slice_from=0, end_idx=None, show=False):
    x_vals = []
    data = []

    for f in files:
        x, _ = load_hdf5_field(f, 'x_vals')
        x_vals.append(x - lcfs_shift)

    n_fields = len(fields)
    n_cols = 4
    n_rows = math.ceil(n_fields / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten() if n_fields > 1 else [axs]

    for i, field in enumerate(fields):
        unit, default_title = all_fields.get(field, ('', field))
        ax = axs[i]
        for f, label, x in zip(files, labels, x_vals):
            _, y = load_hdf5_field(f, field)
            y = y[slice_from:end_idx]
            ax.plot(x[slice_from:end_idx], y, label=label, linewidth=2)
        ax.axvline(x=0, color='gray', linestyle='--')
        ax.set_title(default_title)
        ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
        ax.set_ylabel(unit)
        ax.legend()

    # Hide unused subplots
    for j in range(n_fields, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(output_name)
    if show:
        plt.show()
    else:
        plt.close()

def list_available_fields():
    print("Available fields to plot:")
    for field, (unit, label) in all_fields.items():
        print(f"  {field:15} | {label:40} | Units: {unit}")

def main():
    parser = argparse.ArgumentParser(description="Plot arbitrary fields from HDF5 simulations.")
    parser.add_argument("--files", nargs='+', help="List of HDF5 files to compare")
    parser.add_argument("--labels", nargs='+', help="Labels for each HDF5 file")
    parser.add_argument("--fields", nargs='+', help="Fields to plot")
    parser.add_argument("--all_fields", action="store_true", help="Plot all predefined fields")
    parser.add_argument("--output", default="comparison.pdf", help="Output filename for the plot")
    parser.add_argument("--lcfs", type=float, default=0.10, help="LCFS radial shift")
    parser.add_argument("--slice_from", type=int, default=0, help="Index to slice x and y arrays from")
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    parser.add_argument("--list_fields", action="store_true", help="List available fields and exit")
    args = parser.parse_args()

    if args.list_fields:
        list_available_fields()
        return

    if not args.files or not args.labels:
        raise ValueError("--files and --labels must be specified unless --list_fields is used.")

    if len(args.files) != len(args.labels):
        raise ValueError("The number of files must match the number of labels.")

    if args.all_fields:
        selected_fields = list(all_fields.keys())
    elif args.fields:
        selected_fields = args.fields
    else:
        raise ValueError("You must specify either --fields or --all_fields.")

    plot_fields(args.files, args.labels, selected_fields, args.lcfs, args.output, args.slice_from, args.end_idx, args.show)

if __name__ == "__main__":
    main()
