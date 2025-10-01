# dispersion_analysis.py
#
# A script to perform a 2D FFT in time and the binormal direction (y) to
# calculate the omega-ky dispersion relation for specified fields.
#
# It includes a correction for the E x B Doppler shift to transform the
# frequency from the lab frame to the local plasma frame.
#
import os
import argparse
import numpy as np
import postgkyl as pg
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import utils
except ImportError:
    print("Error: 'utils.py' not found. Please ensure it is in the same directory.")
    exit()

# Physical constants
rho_s = 0.00163386 # [m] 
ELEM_CHARGE = 1.60217662e-19
me = 9.1093837015e-31  # Electron mass [kg]
mi = 1.6726219e-27 * 2.014 # Deuterium ion mass [kg]

# --- Matplotlib and Font Settings ---
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif", "font.size": 14, "image.cmap": 'viridis',
    "axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
})

# === Core Analysis and Utility Functions ===

def get_cell_avg_1d_y(gdata, component_idx, x_slice_idx, z_slice_idx):
    """Extracts a 1D (binormal) array of cell-averaged data."""
    dg = pg.GInterpModal(gdata, poly_order=1, basis_type='ms')
    raw_coeffs_3d = dg._getRawModal(component_idx)
    cell_avg_3d = raw_coeffs_3d[..., 0] / (2**1.5)
    return cell_avg_3d[x_slice_idx, :, z_slice_idx]

def get_cell_avg_2d_xy(gdata, component_idx, z_slice_idx):
    dg = pg.GInterpModal(gdata, poly_order=1, basis_type='ms')
    raw_coeffs_3d = dg._getRawModal(component_idx)
    cell_avg_3d = raw_coeffs_3d[..., 0] / (2**1.5)
    return np.mean(cell_avg_3d[:, :, z_slice_idx], axis=1)

def get_cell_avg_3d(gdata, component_idx):
    dg = pg.GInterpModal(gdata, poly_order=1, basis_type='ms')
    raw_coeffs_3d = dg._getRawModal(component_idx)
    cell_avg_3d = raw_coeffs_3d[..., 0] / (2**1.5)
    return cell_avg_3d

def gradient_1d(gdata, component_idx, x_slice_idx, z_slice_idx):
    """Calculates the centered gradient of a 1D array."""
    dg = pg.GInterpModal(gdata, poly_order=1, basis_type='ms')
    raw_coeffs_3d = dg._getRawModal(component_idx)
    cell_avg_3d = raw_coeffs_3d[..., 0] / (2**1.5)
    
    # Get the grid and calculate the centered difference
    x_vals = gdata.get_grid()[0]
    dx = x_vals[1] - x_vals[0]
    
    gradx_3d = np.gradient(cell_avg_3d, dx, axis=0)

    return gradx_3d[x_slice_idx, :, z_slice_idx]

def calculate_normalization(elc_prefix, fstart, fend, x_idx, z_idx):
    """Calculates time-averaged background Te and ne for normalization."""
    print("\nCalculating background profiles for normalization...")
    te_series, ne_series = [], []
    for tf in range(fstart, fend + 1):
        try:
            elc_data = pg.GData(f"{elc_prefix}-elc_BiMaxwellianMoments_{tf}.gkyl")
            # Get 1D slices for ne and Te
            ne_1d = get_cell_avg_1d_y(elc_data, 0, x_idx, z_idx)
            Tpar_1d = get_cell_avg_1d_y(elc_data, 2, x_idx, z_idx)
            Tperp_1d = get_cell_avg_1d_y(elc_data, 3, x_idx, z_idx)
            Te_1d = (Tpar_1d + 2 * Tperp_1d) / 3.0 * me
            # Binormal average to get the background value at this time step
            ne_series.append(np.mean(ne_1d))
            te_series.append(np.mean(Te_1d))
        except Exception:
            continue
    
    if not te_series or not ne_series:
        raise ValueError("Could not load electron data to calculate background profiles.")
        
    # Time-average the background values
    n0 = np.mean(ne_series)
    Te0_joules = np.mean(te_series) # Gkeyll Te is already in Joules
    print(f"  <n_e> = {n0:.3e} m^-3")
    print(f"  <T_e> = {Te0_joules / ELEM_CHARGE:.2f} eV")
    return n0, Te0_joules

def calculate_background_parameters(file_prefix, fstart, fend, x_idx, z_idx, x_vals):
    print("\nCalculating background profiles and gradients...")
    profiles = {'ne': [], 'Te': [], 'Ti': []}
    for tf in range(fstart, fend + 1):
        try:
            elc_data = pg.GData(f"{file_prefix}-elc_BiMaxwellianMoments_{tf}.gkyl")
            ion_data = pg.GData(f"{file_prefix}-ion_BiMaxwellianMoments_{tf}.gkyl")
            profiles['ne'].append(get_cell_avg_2d_xy(elc_data, 0, z_idx))
            Tepar, Teperp = get_cell_avg_2d_xy(elc_data, 2, z_idx), get_cell_avg_2d_xy(elc_data, 3, z_idx)
            profiles['Te'].append((Tepar + 2*Teperp) / 3.0)
            Tipar, Tiperp = get_cell_avg_2d_xy(ion_data, 2, z_idx), get_cell_avg_2d_xy(ion_data, 3, z_idx)
            profiles['Ti'].append((Tipar + 2*Tiperp) / 3.0)
        except Exception:
            continue
    if not profiles['ne']: raise ValueError("Could not load data for background profiles.")
    mean_profiles = {key: np.mean(np.array(val), axis=0) for key, val in profiles.items()}
    grads = {key: np.gradient(val, x_vals[1] - x_vals[0]) for key, val in mean_profiles.items()}
    local_vals = {key: val[x_idx] for key, val in mean_profiles.items()}
    local_grads = {key: val[x_idx] for key, val in grads.items()}
    params = {}
    params['Ln'] = -local_vals['ne'] / local_grads['ne']
    params['LTe'] = -local_vals['Te'] / local_grads['Te']
    params['LTi'] = -local_vals['Ti'] / local_grads['Ti']
    params['eta_e'], params['eta_i'] = params['Ln'] / params['LTe'], params['Ln'] / params['LTi']
    params['Te0_eV'], params['Ti0_eV'] = local_vals['Te'] / ELEM_CHARGE * me, local_vals['Ti'] / ELEM_CHARGE * mi
    for key, val in params.items(): print(f"  {key:<8} = {val:.3f}")
    return params

def assemble_time_series(file_prefix, file_suffix, component, fstart, fend, x_idx, z_idx):
    """Loads data over a time range and assembles it into a 2D (time, y) array."""
    time_series = []
    # Load first frame to get time step
    try:
        gdata_f = pg.GData(f"{file_prefix}-{file_suffix}_{fstart}.gkyl")
        gdata_f_minus_1 = pg.GData(f"{file_prefix}-{file_suffix}_{fstart-1}.gkyl")
        dt = gdata_f.ctx["time"] - gdata_f_minus_1.ctx["time"]
        print(f"Detected time step dt = {dt:.3e} s")
    except Exception:
        dt = 1.0 # Fallback
    
    for tf in range(fstart, fend + 1):
        print(f"Loading frame {tf} for time series...", end='\r')
        try:
            gdata = pg.GData(f"{file_prefix}-{file_suffix}_{tf}.gkyl")
            time_series.append(get_cell_avg_1d_y(gdata, component, x_idx, z_idx))
        except Exception as e:
            print(f"\nWarning: Could not load data for frame {tf}. Skipping. Error: {e}")
            continue
    if not time_series:
        return None, None, None
    y_vals = gdata.get_grid()[1]
    y_vals = (y_vals[1:] + y_vals[:-1]) / 2
    return np.array(time_series), dt, y_vals

def calculate_doppler_velocity(prefix, fstart, fend, x_idx, z_idx):
    """Calculates the time-averaged E_r / B velocity."""
    print("\nCalculating Doppler shift velocity...")

    bmag_data = pg.GData(prefix + "-bmag.gkyl")
    b_i_data = pg.GData(prefix + "-b_i.gkyl")
    jacgeo_data = pg.GData(prefix + "-jacobgeo.gkyl")
    bmag = get_cell_avg_3d(bmag_data, 0)
    b_x = get_cell_avg_3d(b_i_data, 0)
    b_z = get_cell_avg_3d(b_i_data, 2)
    jacgeo = get_cell_avg_3d(jacgeo_data, 0)
    grid = bmag_data.get_grid()
    x_vals, y_vals, z_vals = grid[0], grid[1], grid[2]
    x_vals = (x_vals[1:] + x_vals[:-1]) / 2
    y_vals = (y_vals[1:] + y_vals[:-1]) / 2
    z_vals = (z_vals[1:] + z_vals[:-1]) / 2
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    dz = z_vals[1] - z_vals[2]

    # Calculate time-averaged radial electric field <E_r>
    Vy_series = []
    Vy_series_simple = []

    for tf in range(fstart, fend + 1):
        phi_data = pg.GData(f"{prefix}-field_{tf}.gkyl")
        phi = get_cell_avg_3d(phi_data, 0)
        dphi_dx = np.gradient(phi, dx, axis=0)
        dphi_dz = np.gradient(phi, dz, axis=2)
        VE_y = (b_z*dphi_dx - b_x*dphi_dz)/bmag/jacgeo
        VE_y_cart = dphi_dx/bmag
        if tf == fstart:
            plt.plot(x_vals, np.mean(VE_y[:,:,z_idx],axis=1),label='curv')
            plt.plot(x_vals, np.mean(VE_y_cart[:,:,z_idx],axis=1),label='cart')
            plt.legend()
            plt.savefig("VE_plot.png")
            print("saved VE plots to VE_plot.png")
        Vy_series.append(VE_y[x_idx,:,z_idx])
        
    V_doppler = np.mean(Vy_series)
    print("Shape of V_doppler series:", np.shape(np.array(Vy_series)))
    print(f"Resulting Doppler Velocity V_ExB = {V_doppler:.3e} m/s")
    return V_doppler

def calculate_dispersion_spectrum(data_ty, dt, y_vals, V_doppler):
    """
    Calculates the omega-ky spectrum, applying the Doppler shift
    as a phase rotation before the time FFT. (Corrected implementation)
    """
    Nt, Ny = data_ty.shape
    time_axis = np.arange(Nt) * dt
    ky_axis = 2 * np.pi * np.fft.fftfreq(Ny, d=(y_vals[1] - y_vals[0]))

    # 1. FFT along the spatial (y) dimension to get A(t, ky)
    # We subtract the time-mean of each spatial point first.
    data_ty_fluct = data_ty - np.mean(data_ty, axis=0)
    A_t_ky = np.fft.fft(data_ty_fluct, axis=1)

    # 2. Construct the phase shift matrix using a robust method (np.outer).
    #    The phase factor for the shift is exp(-i * ky * V_doppler * t).
    #    The outer product of time_axis and ky_axis creates the (t, ky) matrix needed.
    phase_matrix = np.outer(time_axis, ky_axis)
    phase_shift = np.exp(1j * V_doppler * phase_matrix)
    
    # 3. Apply the phase rotation.
    A_t_ky_shifted = A_t_ky * phase_shift
    print("\nApplied Doppler shift as phase rotation in time domain.")

    # 4. Apply a Hann window along the time axis of the phase-shifted data.
    hann_window = np.hanning(Nt)

    beta = 8  # A good value for strong side-lobe suppression. Try values from 5 to 12.
    kaiser_window = np.kaiser(Nt, beta)

    data_windowed = A_t_ky_shifted * hann_window[:, np.newaxis]

    # 5. FFT along the time dimension to get F(omega_wave, ky).
    dispersion_fft = np.fft.fft(data_windowed, axis=0)
    
    # 6. The power spectrum.
    power_spectrum = (dt / Nt) * np.abs(dispersion_fft)**2
    return power_spectrum

def plot_dispersion(power_spectrum, omega_axis, ky_axis, rho, metadata, theory_params=None):
    """Plots the omega-ky dispersion diagram with Doppler shift correction."""
    fig, ax = plt.subplots(figsize=(8, 7))

    power_shifted = np.fft.fftshift(power_spectrum)
    omega_lab_shifted = np.fft.fftshift(omega_axis)
    ky_shifted = np.fft.fftshift(ky_axis)
    
    log_power = np.log10(power_shifted + 1e-20)  # Avoid log(0) issues
    omega_max_plot = (np.abs(omega_lab_shifted).max())

    # Plot the location of the peak power vs. frequency
    peak_indices = np.argmax(power_shifted, axis=0)
    peak_frequencies = omega_lab_shifted[peak_indices]
    peak_ky = ky_shifted
    ax.plot(peak_ky * rho_s, peak_frequencies, 'r-', label='Peak Power Locations')

    # Plot the median frequency line
    cumulative_power = np.cumsum(power_spectrum, axis=0)
    # Normalize each column by its total power
    total_power = cumulative_power[-1, :]
    # Avoid division by zero for columns with no power
    total_power[total_power == 0] = 1.0
    normalized_cumsum = cumulative_power / total_power
            
    # Find the index where the cumulative sum crosses 0.5 for each column
    # np.searchsorted is efficient for this
    median_indices = np.zeros(power_spectrum.shape[1], dtype=int)
    for i in range(power_spectrum.shape[1]):
        median_indices[i] = np.searchsorted(normalized_cumsum[:, i], 0.5)
            
    # Clip indices to be within bounds
    median_indices = np.clip(median_indices, 0, len(omega_axis) - 1)
    dispersion_omega = omega_axis[median_indices]
    omega_for_line = np.fft.fftshift(dispersion_omega)  
    ax.plot(ky_shifted * rho_s, omega_for_line, 'k--', label='Median Frequency Line')

    vmax = np.max(log_power)
    vmin = vmax - 7

    im = ax.imshow(
        log_power,
        extent=[ky_shifted.min()*rho_s, ky_shifted.max()*rho_s, -omega_max_plot, omega_max_plot],
        origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
        cmap=mpl.colormaps.get_cmap('viridis'), 
    )

    if theory_params:
        print("Overlaying theoretical dispersion relations...")
        ky_rhos = ky_shifted * rho_s
        positive_k_mask = ky_rhos > 0
        ky_rhos_pos = ky_rhos[positive_k_mask]

        c_s = np.sqrt(theory_params['Te0_eV'] * ELEM_CHARGE / mi)
        omega_star_e = ky_rhos_pos * (c_s / theory_params['Ln'])
        
        # 1. TEM/RBM Frequency (propagates in electron direction, ω > 0)
        omega_tem = omega_star_e * (1 + theory_params['eta_e'])
        ax.plot(ky_rhos_pos, omega_tem, color='white', linestyle='--', linewidth=2.5, label='TEM')
        
        # 2. ITG Frequency (propagates in ion direction, ω < 0)
        tau = theory_params['Ti0_eV'] / theory_params['Te0_eV']
        omega_itg = -tau * omega_star_e * (1 + theory_params['eta_i'])
        ax.plot(ky_rhos_pos, omega_itg, color='magenta', linestyle=':', linewidth=2.5, label='ITG')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'log$_{10}$ |$F(\omega, k_y)$|$^2$')
    
    ax.set_xlabel(r'$k_y \rho_s$')
    ax.set_ylabel(r'$\omega_{plasma}$ [rad/s]')
    
    qty_name = metadata['quantity']
    rho_str = r'$\rho$'
    ax.set_title(f'Dispersion for {qty_name} at {rho_str}={rho:.3f}')
    
    ax.set_xlim(left=0)
    ax.set_ylim(-omega_max_plot, omega_max_plot)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_filename = f"dispersion_doppler_corr_{qty_name}_x{metadata['x_idx']}_{metadata['fstart']}to{metadata['fend']}.png"
    plt.savefig(output_filename)
    print(f"\nSaved plot to {output_filename}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Calculate Doppler-corrected omega-ky dispersion relation.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--fstart", type=int, required=True, help="Starting frame number.")
    parser.add_argument("--fend", type=int, required=True, help="Ending frame number.")
    parser.add_argument("--quantity", type=str, default='phi', choices=['phi', 'elcDens'], help="Quantity to analyze.")
    parser.add_argument("--xidx", type=int, default=None, help="Radial index to analyze. Default: middle of the domain.")
    parser.add_argument("--zstr", type=str, default="zmid", help="String identifier for the z-slice ('zmid' or 'zmin').")
    args = parser.parse_args()

    quantity_map = {'phi': ('field', 0), 'elcDens': ('elc_BiMaxwellianMoments', 0)}
    file_suffix, component = quantity_map[args.quantity]
    file_prefix = utils.find_prefix(f"-{file_suffix}_0.gkyl", '.')
    
    gdata_initial = pg.GData(f"{file_prefix}-{file_suffix}_{args.fstart}.gkyl")
    x_vals, _, z_vals = gdata_initial.get_grid()
    x_idx = args.xidx if args.xidx is not None else len(x_vals) // 2
    z_idx = 0 if args.zstr == 'zmin' else len(z_vals) // 2

    is_PT = input("triangularity [PT/NT]? ")
    if is_PT == "PT":
        R_axis = 1.6486461
        print("calculating dispersion for PT with R_axis = ", R_axis)
    else:
        R_axis = 1.7074685
        print("calculating dispersion for NT with R_axis = ", R_axis)
    x_val = x_vals[x_idx] - 0.1 # 0.1 is simulation domain inside LCFS
    R_LCFS = 2.17 # assumed for both PT/NT
    R = R_LCFS + x_val 
    rho = (R - R_axis)/(R_LCFS - R_axis)
    print(R, R_LCFS, rho)

    # Step 1: Calculate background profiles for normalization
    n0, Te0_joules = calculate_normalization(file_prefix, args.fstart, args.fend, x_idx, z_idx)

    # 1. Calculate the Doppler shift velocity
    V_doppler = calculate_doppler_velocity(file_prefix, args.fstart, args.fend, x_idx, z_idx)
    
    # 2. Assemble the 2D (time, y) data array for the chosen quantity
    data_ty, dt, y_vals = assemble_time_series(file_prefix, file_suffix, component, args.fstart, args.fend, x_idx, z_idx)
    if data_ty is None: exit("Could not assemble time series. Exiting.")

    if args.quantity == 'phi':
        normalization_factor = Te0_joules / ELEM_CHARGE
        data_ty_normalized = data_ty / normalization_factor
    elif args.quantity == 'elcDens':
        # For density, we analyze fluctuations around the mean, normalized by the mean
        data_ty_fluctuations = data_ty - np.mean(data_ty, axis=(0,1))
        data_ty_normalized = data_ty_fluctuations / n0

    # 3. Calculate the dispersion power spectrum
    power_spectrum = calculate_dispersion_spectrum(data_ty_normalized, dt, y_vals, V_doppler)

    # 4. Create omega and ky axes
    Nt, Ny = data_ty.shape
    omega_axis = 2 * np.pi * np.fft.fftfreq(Nt, d=dt)
    ky_axis = 2 * np.pi * np.fft.fftfreq(Ny, d=(y_vals[1] - y_vals[0]))

    theory_params = calculate_background_parameters(file_prefix, args.fstart, args.fend, x_idx, z_idx, x_vals)

    # 5. Plot the result with the Doppler correction
    metadata = {
        "fstart": args.fstart, "fend": args.fend, "x_idx": x_idx, 
        "z_str": args.zstr, "quantity": args.quantity
    }
    plot_dispersion(power_spectrum, omega_axis, ky_axis, rho, metadata, theory_params)

if __name__ == "__main__":
    main()