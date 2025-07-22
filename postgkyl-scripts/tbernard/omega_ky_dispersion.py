# dispersion_analysis.py
#
# A script to perform a 2D FFT in time and the binormal direction (y) to
# calculate the omega-ky dispersion relation for specified fields.
#
# This analysis is performed at a single radial (x_idx) and parallel (z_idx)
# location and is used to identify dominant linear instabilities.
#
import os
import argparse
import numpy as np
import postgkyl as pg
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import utils
except ImportError:
    print("Error: 'utils.py' not found. Please ensure it is in the same directory.")
    exit()

# Physical constants
# rho_s = omega_ci / c_s = e * B / (m_i * c_s) = e * B / (m_i * sqrt(T_e/m_i))
rho_s = 0.00163386 # [m] 

# --- Matplotlib and Font Settings ---
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif", #"font.serif": ["Palatino"], 
    "font.size": 14,
    "image.cmap": 'viridis',
    "axes.labelsize": 16, "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# === Core Analysis and Utility Functions ===

def get_cell_avg_1d(gdata, component_idx, x_slice_idx, z_slice_idx):
    """Extracts a 1D (binormal) array of cell-averaged data."""
    dg = pg.GInterpModal(gdata, poly_order=1, basis_type='ms')
    raw_coeffs_3d = dg._getRawModal(component_idx)
    cell_avg_3d = raw_coeffs_3d[..., 0] / (2**1.5)
    return cell_avg_3d[x_slice_idx, :, z_slice_idx]

def assemble_time_series(file_prefix, file_suffix, component, fstart, fend, x_idx, z_idx):
    """
    Loads data over a time range and assembles it into a 2D (time, y) array.
    """
    time_series = []
    time_vals = []
    
    # Load first frame to get time step
    try:
        gdata_f = pg.GData(f"{file_prefix}-{file_suffix}_{fstart}.gkyl")
        gdata_f_minus_1 = pg.GData(f"{file_prefix}-{file_suffix}_{fstart-1}.gkyl")
        time_1 = gdata_f.ctx["time"]
        time_0 = gdata_f_minus_1.ctx["time"]
        dt = time_1 - time_0
        print(f"Detected time step dt = {dt:.3e} s")
    except Exception as e:
        print(f"Warning: Could not determine time step. Will assume dt=1.0. Error: {e}")
        dt = 1.0

    for tf in range(fstart, fend + 1):
        print(f"Loading frame {tf}...", end='\r')
        try:
            gdata = pg.GData(f"{file_prefix}-{file_suffix}_{tf}.gkyl")
            data_1d = get_cell_avg_1d(gdata, component, x_idx, z_idx)
            time_series.append(data_1d)
            time_vals.append(gdata.ctx["time"])
        except Exception as e:
            print(f"\nWarning: Could not load data for frame {tf}. Skipping. Error: {e}")
            continue
            
    if not time_series:
        return None, None, None, None

    # Get the y-axis grid from the last successfully loaded file
    y_vals = gdata.get_grid()[1]
    y_vals = (y_vals[1:] + y_vals[:-1]) / 2  # Center coordinates
    
    # Convert list of 1D arrays into a single 2D array of shape (Nt, Ny)
    data_ty = np.array(time_series)
    
    return data_ty, dt, y_vals

def calculate_dispersion_spectrum(data_ty):
    """
    Calculates the omega-ky spectrum from a 2D (time, y) data series.
    """
    Nt, Ny = data_ty.shape
    
    # 1. Apply a Hann window (mask) along the time axis to reduce spectral leakage
    hann_window = np.hanning(Nt)
    data_windowed = data_ty * hann_window[:, np.newaxis] # Reshape window for broadcasting
    
    # 2. Perform a 2D FFT to transform from (t, y) to (omega, ky) space
    dispersion_fft = np.fft.fft2(data_windowed)
    
    # 3. Calculate the power spectrum
    power_spectrum = np.abs(dispersion_fft)**2
    
    return power_spectrum

def plot_dispersion(power_spectrum, omega_axis, ky_axis, metadata):
    """Plots the omega-ky dispersion diagram."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Use fftshift to move the zero-frequency/wavenumber to the center for plotting
    power_shifted = np.fft.fftshift(power_spectrum)
    omega_shifted = np.fft.fftshift(omega_axis)
    ky_shifted = np.fft.fftshift(ky_axis)
    
    # Use a logarithmic color scale to see both strong and weak modes
    # Add a small epsilon to avoid log(0)
    log_power = np.log10(power_shifted + 1e-20)
    
    # Determine plot limits to focus on the interesting part of the spectrum
    omega_max_plot = omega_shifted.max() / 4 # Often the highest frequencies are not interesting
    
    im = ax.imshow(
        log_power,
        extent=[ky_shifted.min()*rho_s, ky_shifted.max()*rho_s, omega_shifted.min(), omega_shifted.max()],
        origin='lower',
        aspect='auto',
        vmin=-1,
        vmax=np.max(log_power),
        cmap=mpl.colormaps.get_cmap('viridis'),
    )
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'log$_{10}$ |$F(\omega, k_y)$|$^2$')
    
    ax.set_xlabel(r'$k_y \rho_s$')
    ax.set_ylabel(r'$\omega$ (Frequency [rad/s])')
    
    qty_name = metadata['quantity']
    ax.set_title(f'Dispersion Relation for ${qty_name}$ at x_idx={metadata["x_idx"]}')
    
    ax.set_xlim(left=0) # Often only positive ky is of interest
    #ax.set_ylim(-omega_max_plot, omega_max_plot)
    
    plt.tight_layout()
    output_filename = f"dispersion_{qty_name}_x{metadata['x_idx']}_{metadata['fstart']}to{metadata['fend']}.png"
    plt.savefig(output_filename)
    print(f"\nSaved plot to {output_filename}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Calculate and plot the omega-ky dispersion relation from Gkeyll data.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--fstart", type=int, required=True, help="Starting frame number.")
    parser.add_argument("--fend", type=int, required=True, help="Ending frame number.")
    parser.add_argument("--quantity", type=str, default='phi', choices=['phi', 'elcDens'], help="Quantity to analyze.")
    parser.add_argument("--xidx", type=int, default=None, help="Radial index to analyze. Default: middle of the domain.")
    parser.add_argument("--zstr", type=str, default="zmid", help="String identifier for the z-slice ('zmid' or 'zmin').")

    args = parser.parse_args()

    # Map quantity names to file suffixes and component indices
    quantity_map = {
        'phi': ('field', 0),
        'elcDens': ('elc_BiMaxwellianMoments', 0),
    }
    file_suffix, component = quantity_map[args.quantity]

    # Ensure there are enough time points for a meaningful FFT
    if (args.fend - args.fstart) < 16:
        print("Warning: A short time series (fend-fstart < 16) may result in a poor frequency spectrum.")

    # --- Analysis Workflow ---
    try:
        file_prefix = utils.find_prefix(f"-{file_suffix}_0.gkyl", '.')
        print(f"Using file prefix: {file_prefix}")
    except FileNotFoundError as e:
        print(e), exit()

    # Load one file to get grid information for default indices
    try:
        gdata_initial = pg.GData(f"{file_prefix}-{file_suffix}_{args.fstart}.gkyl")
        x_vals, _, z_vals = gdata_initial.get_grid()
        x_idx = args.xidx if args.xidx is not None else len(x_vals) // 2
        z_idx = 0 if args.zstr == 'zmin' else len(z_vals) // 2
        if x_idx >= len(x_vals):
            raise ValueError(f"--xidx {x_idx} is out of bounds for radial grid size {len(x_vals)}")
    except Exception as e:
        print(f"Error: Could not load initial file for grid info. Exiting. {e}"), exit()
    
    print(f"Analyzing '{args.quantity}' at radial index x_idx = {x_idx} (R ≈ {x_vals[x_idx]:.3f})")

    # 1. Assemble the 2D (time, y) data array
    data_ty, dt, y_vals = assemble_time_series(file_prefix, file_suffix, component, args.fstart, args.fend, x_idx, z_idx)
    
    if data_ty is None:
        print("\nError: Could not assemble time series. No data was loaded. Exiting.")
        exit()

    # 2. Calculate the dispersion power spectrum
    power_spectrum = calculate_dispersion_spectrum(data_ty)

    # 3. Create the corresponding omega and ky axes
    Nt, Ny = data_ty.shape
    omega_axis = 2 * np.pi * np.fft.fftfreq(Nt, d=dt)
    ky_axis = 2 * np.pi * np.fft.fftfreq(Ny, d=(y_vals[1] - y_vals[0]))

    # 4. Plot the result
    metadata = {
        "fstart": args.fstart, "fend": args.fend, "x_idx": x_idx, 
        "z_str": args.zstr, "quantity": args.quantity
    }
    plot_dispersion(power_spectrum, omega_axis, ky_axis, metadata)

if __name__ == "__main__":
    main()