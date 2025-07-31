# k_spectra_analysis.py
#
# A script to perform binormal wavenumber (k_y) spectra analysis on Gkeyll
# simulation data using raw, cell-averaged values.
#
#
import os
import argparse
import numpy as np
import postgkyl as pg
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import the user's utility functions
try:
    import utils
except ImportError:
    print("Error: 'utils.py' not found. Please ensure it is in the same directory.")
    exit()

# --- Matplotlib and Font Settings ---
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 14,
    "image.cmap": 'inferno',
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
})

# === Core Analysis and Utility Functions ===

def get_cell_avg_2d(gdata, component_idx, z_slice_idx):
    """Extracts the 2D cell-averaged data from a GData object."""
    dg = pg.GInterpModal(gdata, poly_order=1, basis_type='ms')
    raw_coeffs_3d = dg._getRawModal(component_idx)
    cell_avg_3d = raw_coeffs_3d[..., 0] / (2**1.5)
    return cell_avg_3d[:, :, z_slice_idx]

def calculate_ky_spectrum(data_2d):
    """Calculates the binormal wavenumber (ky) spectrum for a 2D data slice."""
    fluctuations = data_2d - np.mean(data_2d, axis=1, keepdims=True)
    ky_fft = np.fft.fft(fluctuations, axis=1)
    power_spectrum_2d = np.abs(ky_fft)**2
    radially_averaged_spectrum = np.mean(power_spectrum_2d, axis=0)
    return radially_averaged_spectrum

def save_spectra_to_hdf5(filename, ky_axis, spectra_data, metadata):
    """Saves the calculated spectra and metadata to an HDF5 file."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('ky_binormal', data=ky_axis)
        for name, spectrum in spectra_data.items():
            f.create_dataset(name, data=spectrum)
        for key, value in metadata.items(): f.attrs[key] = value
        print(f"Successfully saved spectra to {filename}")

def plot_spectra(ky_axis, spectra_data, fstart, fend, z_str, is_cell_avg=True):
    """Plots the calculated spectra on a log-log scale."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ky_shifted = np.fft.fftshift(ky_axis)
    
    for name, spectrum in spectra_data.items():
        spectrum_shifted = np.fft.fftshift(spectrum)
        # Plot only the positive wavenumbers
        positive_k_mask = ky_shifted > 0
        max_val = np.max(spectrum_shifted[positive_k_mask])
        ax.plot(ky_shifted[positive_k_mask], spectrum_shifted[positive_k_mask]/max_val, label=f'${name}$')
        
    ax.set_xlabel(r'$k_y$ (Binormal Wavenumber)')
    ax.set_ylabel('Power Spectrum (arb. units)')
    
    title_method = "Cell-Avg" if is_cell_avg else "Interpolated"
    ax.set_title(f'Time-Averaged $k_y$ Spectra ({title_method}, {z_str}, frames {fstart}-{fend})')
    
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.6), ax.legend()
    plt.tight_layout()
    
    output_filename = f"k_spectra_binormal_{fstart}to{fend}_{z_str}.png"
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")
    plt.show()

# === Main Execution Block ===
def main():
    parser = argparse.ArgumentParser(
        description="Perform or visualize binormal k-spectra analysis on Gkeyll data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Mode Selection ---
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help="Path to an HDF5 file to plot. If specified, script enters plot-only mode."
    )
    # --- Analysis Mode Arguments ---
    parser.add_argument("--fstart", type=int, help="Starting frame number (for analysis mode).")
    parser.add_argument("--fend", type=int, help="Ending frame number (for analysis mode).")
    parser.add_argument("--zstr", type=str, default="zmid", help="String identifier for the z-slice ('zmid' or 'zmin').")
    parser.add_argument("--xidx", type=int, default=None, help="Radial index to analyze. Default: middle of the domain.")

    args = parser.parse_args()

    # Otherwise, enter analysis mode
    # Check for required arguments in this mode
    if args.fstart is None or args.fend is None:
        parser.error("--fstart and --fend are required for analysis mode.")
    
    print("--- Entering Analysis Mode ---")
    run_analysis(args)

def run_analysis(args):
    """The main analysis workflow, called when not in plot-only mode."""
    try:
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        print(f"Using file prefix: {file_prefix}")
    except FileNotFoundError as e:
        print(e), exit()

    quantities_to_analyze = ['phi', 'elcDens', 'elcTemp', 'ionDens', 'ionTemp']
    spectra_time_series = {name: [] for name in quantities_to_analyze}

    try:
        phi_data_initial = pg.GData(f"{file_prefix}-field_{args.fstart}.gkyl")
        x_vals, y_vals, z_vals = phi_data_initial.get_grid()
        # Take cell-centered coordinates
        x_vals = (x_vals[1:]+x_vals[:-1])/2
        y_vals = (y_vals[1:]+y_vals[:-1])/2
        z_vals = (z_vals[1:]+z_vals[:-1])/2
        z_idx = 0 if args.zstr == 'zmin' else len(z_vals) // 2
    except Exception as e:
        print(f"Error: Could not load initial file for grid info. Exiting. {e}"), exit()

    for tf in range(args.fstart, args.fend + 1):
        print(f"Processing frame {tf}...")
        try:
            elc_data = pg.GData(f"{file_prefix}-elc_BiMaxwellianMoments_{tf}.gkyl")
            ion_data = pg.GData(f"{file_prefix}-ion_BiMaxwellianMoments_{tf}.gkyl")
            phi_data = pg.GData(f"{file_prefix}-field_{tf}.gkyl")
        except Exception as e:
            print(f"Warning: Could not load data for frame {tf}. Skipping. Error: {e}")
            continue

        phi_2d_cellave = get_cell_avg_2d(phi_data, 0, z_idx)
        elc_dens_2d_cellave = get_cell_avg_2d(elc_data, 0, z_idx)
        ion_dens_2d_cellave = get_cell_avg_2d(ion_data, 0, z_idx)
        
        elc_Tpar_2d_cellave = get_cell_avg_2d(elc_data, 2, z_idx)
        elc_Tperp_2d_cellave = get_cell_avg_2d(elc_data, 3, z_idx)
        elc_temp_2d_cellave = (elc_Tpar_2d_cellave + 2 * elc_Tperp_2d_cellave) / 3.0

        ion_Tpar_2d_cellave = get_cell_avg_2d(ion_data, 2, z_idx)
        ion_Tperp_2d_cellave = get_cell_avg_2d(ion_data, 3, z_idx)
        ion_temp_2d_cellave = (ion_Tpar_2d_cellave + 2 * ion_Tperp_2d_cellave) / 3.0

        data_map = {'phi': phi_2d_cellave, 'elcDens': elc_dens_2d_cellave, 'elcTemp': elc_temp_2d_cellave,
                    'ionDens': ion_dens_2d_cellave, 'ionTemp': ion_temp_2d_cellave}
        
        for name in quantities_to_analyze:
            spectrum = calculate_ky_spectrum(data_map[name])
            spectra_time_series[name].append(spectrum)

    if not any(spectra_time_series.values()):
        print("Error: No frames were successfully processed. Exiting."), exit()

    averaged_spectra = {name: np.mean(np.array(series), axis=0) for name, series in spectra_time_series.items() if series}

    ky_axis = 2 * np.pi * np.fft.fftfreq(len(y_vals), d=(abs(y_vals[1] - y_vals[0])))

    metadata = {"fstart": args.fstart, "fend": args.fend, "z_str": args.zstr, "z_idx": z_idx}
    output_h5_file = f"k_spectra_binormal_{args.fstart}to{args.fend}_{args.zstr}.h5"
    
    save_spectra_to_hdf5(output_h5_file, ky_axis, averaged_spectra, metadata)
    plot_spectra(ky_axis, averaged_spectra, args.fstart, args.fend, args.zstr)

if __name__ == "__main__":
    main()
