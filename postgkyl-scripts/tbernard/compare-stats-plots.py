import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

# Optional: customize LaTeX-style labels
label_map = {
    'skew': r'skewness($\tilde{n}$)',
    'kurt': r'kurtosis($\tilde{n}$)',
    'skew_temp': r'skew($\tilde{T}_e$)',
    'kurt_temp': r'kurt($\tilde{T}_e$)',
    'skew_phi':  r'skew($\tilde{\phi}$)',
    'kurt_phi':  r'kurt($\tilde{\phi}$)',
    'l_rad' : r'radial correlation length'
}

def load_hdf5_field(filepath, field):
    with h5py.File(filepath, 'r') as f:
        if field == 'l_rad' and 'xVals_new' in f:
            x_vals = f['xVals_new'][:]
        else:
            x_vals = f['x_vals'][:]
        data = f[field][:]
        return x_vals[:], data[:]

def plot_moments(files, labels, fields, lcfs_shift, output, slice_from=0, show=False):
    n_fields = len(fields)
    n_cols = 3
    n_rows = math.ceil(n_fields / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten() if n_fields > 1 else [axs]

    for i, field in enumerate(fields):
        ax = axs[i]
        for f, label in zip(files, labels):
            x, y = load_hdf5_field(f, field)
            x = x - lcfs_shift
            x = x[slice_from:]
            y = y[slice_from:]
            ax.plot(x, y, label=label, linewidth=2)
        ax.axvline(x=0, color='gray', linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='-')
        ax.set_title(label_map.get(field, field))
        ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
        ax.legend()

    for j in range(n_fields, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(output)
    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot skewness and kurtosis profiles from HDF5 files.")
    parser.add_argument("--files", nargs='+', help="List of HDF5 files")
    parser.add_argument("--labels", nargs='+', help="Labels for files")
    parser.add_argument("--fields", nargs='+', help="Statistical moment fields to plot")
    parser.add_argument("--output", default="moments_comparison.pdf", help="Output PDF filename")
    parser.add_argument("--lcfs", type=float, default=0.10, help="LCFS radial shift")
    parser.add_argument("--slice_from", type=int, default=0, help="Index to slice x and y arrays from")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    args = parser.parse_args()

    if not args.files or not args.labels or not args.fields:
        raise ValueError("--files, --labels, and --fields must be specified")

    if len(args.files) != len(args.labels):
        raise ValueError("Each file must have a corresponding label")

    plot_moments(args.files, args.labels, args.fields, args.lcfs, args.output, args.slice_from, args.show)

if __name__ == "__main__":
    main()
