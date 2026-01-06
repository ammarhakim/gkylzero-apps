import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

# --- Plotting Aesthetics ---
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "lines.linewidth": 2
})

# --- Field Definitions ---
all_fields = {
    # Profiles
    'neAve':       ('m$^{-3}$',      r'$\langle n_e \rangle$'),
    'TeAve':       ('eV',            r'$\langle T_e \rangle$'),
    'TiAve':       ('eV',            r'$\langle T_i \rangle$'),
    'phiAve':      ('V',             r'$\langle \phi \rangle$'),
    'QparaAve':    ('W m$^{-2}$',    r'$\langle Q_\parallel \rangle$'),
    
    # Normalized Fluctuations
    'dn_norm':     ('',              r'$\tilde{n}_{rms}/\langle n \rangle$'),
    'dT_norm':     ('',              r'$\tilde{T}_{rms}/\langle T \rangle$'),
    'dphi_norm':   ('',              r'$e\tilde{\phi}_{rms}/\langle T_e \rangle$'),
    
    # Absolute Fluctuations
    'dn_rms':      ('m$^{-3}$',      r'$\tilde{n}_{rms}$'),
    'dT_rms':      ('eV',            r'$\tilde{T}_{e,rms}$'),
    'dphi_rms':    ('V',             r'$\tilde{\phi}_{rms}$'),

    # Transport Fluxes
    'Gamma_x':     ('m$^{-2}$ s$^{-1}$', r'$\Gamma_x$'),
    'Qxe':         ('W m$^{-2}$',    r'$Q_{e,x}$'),
    'Qxi':         ('W m$^{-2}$',    r'$Q_{i,x}$'),
    'Rey_stress':  ('m$^2$ s$^{-2}$', r'$\langle \tilde{v}_x \tilde{v}_y \rangle$'),
    'Rey_force':   ('m s$^{-2}$',    r'Reynolds Force'),

    # Statistics
    'skew':        ('',              r'Skewness ($n_e$)'),
    'kurt':        ('',              r'Kurtosis ($n_e$)'),
    'l_rad':       ('m',             r'$L_{rad, corr}$'),
}

def load_hdf5_field(filepath, field_name):
    """
    Loads x_vals and the requested data field from the HDF5 file.
    Handles scalar datasets (like global stats) by broadcasting them to x_vals shape.
    """
    x = np.array([])
    y = np.array([])
    
    try:
        with h5py.File(filepath, 'r') as f:
            # 1. Load X Axis
            if 'x_vals' in f:
                x = f['x_vals'][:]
            elif 'x' in f:
                x = f['x'][:]
            else:
                print(f"Error: Could not find 'x_vals' or 'x' in {filepath}")
                return np.array([]), np.array([])
            
            # 2. Check if field exists
            if field_name not in f:
                print(f"Warning: Field '{field_name}' not found in {filepath}. Returning zeros.")
                return x, np.zeros_like(x)
            
            # 3. Load Dataset (Handle Scalars vs Arrays)
            dset = f[field_name]
            
            if dset.ndim == 0:
                # SCALAR DATASET (e.g., l_rad might be a single number)
                val = dset[()] 
                # Broadcast scalar to array shape so it plots as a flat line
                y = np.full_like(x, val)
            else:
                # ARRAY DATASET
                y = dset[:]

            # 4. Check length consistency
            if len(y) != len(x):
                # If sizes mismatch (rare), try to resize or warn
                print(f"Warning: Shape mismatch in {filepath} for {field_name}. x:{x.shape}, y:{y.shape}")
                # Fallback: slice to match or pad
                if len(y) > len(x): y = y[:len(x)]
                else: return x, np.zeros_like(x)

            return x, y
            
    except Exception as e:
        print(f"Error reading {field_name} from {filepath}: {e}")
        # Return zeros if X was loaded successfully, so plotting doesn't crash
        if len(x) > 0:
            return x, np.zeros_like(x)
        return np.array([]), np.array([])

def plot_fields(files, labels, fields, lcfs_shift, output_name, slice_from=0, end_idx=None, show=False, colors=None, linestyles=None):
    
    # Pre-load X data just to establish the grid
    x_vals_list = []
    for f in files:
        x, _ = load_hdf5_field(f, 'neAve') # Dummy load
        if len(x) == 0: 
            print(f"Skipping file {f} due to load error.")
            return
        x_vals_list.append(x - lcfs_shift)

    n_fields = len(fields)
    n_cols = min(3, n_fields)
    n_rows = math.ceil(n_fields / n_cols)

    fig_width = 5 * n_cols
    fig_height = 4 * n_rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=True)
    if n_fields > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, field in enumerate(fields):
        unit, default_title = all_fields.get(field, ('', field))
        ax = axs[i]
        
        for idx, (f_path, label, x) in enumerate(zip(files, labels, x_vals_list)):
            _, y = load_hdf5_field(f_path, field)
            
            # Safety check: if load failed completely
            if len(y) == 0: continue

            # Slice data
            x_plot = x[slice_from:end_idx]
            y_plot = y[slice_from:end_idx]

            # Style logic
            c = colors[idx] if colors and idx < len(colors) else None
            ls = linestyles[idx] if linestyles and idx < len(linestyles) else '-'
            
            ax.plot(x_plot, y_plot, label=label, color=c, linestyle=ls)

        # Plot decorations
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(default_title)
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        
        # Only set xlabel on bottom rows
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel(r'$R - R_{LCFS}$ (m)')
        
        # Legend only on the first plot to avoid clutter
        if i == 0:
            ax.legend(loc='best', frameon=False, fontsize=12)

    # Remove empty subplots
    for j in range(n_fields, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    print(f"Saving plot to {output_name}...")
    plt.savefig(output_name, dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close()

def list_available_fields():
    print(f"{'Field Key':<15} | {'Description':<40} | {'Units'}")
    print("-" * 70)
    for field, (unit, label) in all_fields.items():
        clean_label = label.replace('$', '')
        print(f"{field:<15} | {clean_label:<40} | {unit}")

def main():
    parser = argparse.ArgumentParser(description="Plot fields from HDF5 diagnostics.")
    parser.add_argument("--files", nargs='+', help="List of HDF5 files to compare")
    parser.add_argument("--labels", nargs='+', help="Labels for each HDF5 file")
    parser.add_argument("--fields", nargs='+', help="Specific fields to plot")
    parser.add_argument("--all_fields", action="store_true", help="Plot ALL defined fields")
    parser.add_argument("--output", default="comparison.png", help="Output filename")
    parser.add_argument("--lcfs", type=float, default=0.0, help="Shift X axis by this amount (R_LCFS)")
    parser.add_argument("--slice_from", type=int, default=0, help="Start index for slicing")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for slicing")
    parser.add_argument("--show", action="store_true", help="Show plot window after saving")
    parser.add_argument("--list_fields", action="store_true", help="List all available field keys and exit")
    parser.add_argument("--colors", nargs='+', help="Custom colors")
    parser.add_argument("--linestyles", nargs='+', help="Custom linestyles")
    
    args = parser.parse_args()

    if args.list_fields:
        list_available_fields()
        return

    # Input Validation
    if not args.files or not args.labels:
        parser.error("You must provide --files and --labels.")
    if len(args.files) != len(args.labels):
        parser.error("Number of files must match number of labels.")
    
    # Select Fields
    if args.all_fields:
        selected_fields = list(all_fields.keys())
    elif args.fields:
        selected_fields = args.fields
        for f in selected_fields:
            if f not in all_fields:
                print(f"Warning: '{f}' is not in the standard dictionary.")
    else:
        selected_fields = ['neAve', 'TeAve', 'dn_norm', 'Gamma_x']

    plot_fields(
        files=args.files, 
        labels=args.labels, 
        fields=selected_fields, 
        lcfs_shift=args.lcfs, 
        output_name=args.output, 
        slice_from=args.slice_from, 
        end_idx=args.end_idx, 
        show=args.show, 
        colors=args.colors, 
        linestyles=args.linestyles
    )

if __name__ == "__main__":
    main()