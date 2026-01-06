import os
import numpy as np
import postgkyl as pg
import h5py
import utils
from scipy.stats import skew, kurtosis

# Physical constants
MP = 1.672623e-27
AMU = 2.014
MI = MP * AMU
ME = 9.10938188e-31
EV = 1.602e-19

def calc_correlation_length(fluctuation_array, x_vals):
    """
    Calculates radial correlation length using autocorrelation.
    Input: fluctuation_array shape (Time*Y, X)
    """
    # 1. Calculate Autocorrelation Function (ACF) along X (axis 1)
    # We use valid mode convolution for efficiency
    n_samples, n_x = fluctuation_array.shape
    
    # Normalize fluctuations
    f_norm = fluctuation_array / (np.std(fluctuation_array, axis=1, keepdims=True) + 1e-10)
    
    # Compute ACF via FFT for speed, or manual dot product
    # Here we do a mean ACF across all time/y samples
    acf = np.zeros(n_x)
    for i in range(n_samples):
        row = f_norm[i, :]
        res = np.correlate(row, row, mode='full')
        res = res[res.size // 2:] # Keep positive lags
        res /= np.max(res) # Normalize to 1 at lag 0
        acf += res
    acf /= n_samples

    # 2. Fit exponential decay exp(-dx/L_corr) to find L_corr
    # We fit only the first part where ACF > 1/e to avoid tail noise
    dx = x_vals[1] - x_vals[0]
    lags = np.arange(n_x) * dx
    
    threshold_idx = np.argmax(acf < 1/np.e)
    if threshold_idx == 0: threshold_idx = 5 # Fallback if decorrelation is instant
    
    # Simple log linear fit: ln(ACF) = -x/L
    y_fit = np.log(np.clip(acf[:threshold_idx], 1e-5, 1.0))
    x_fit = lags[:threshold_idx]
    
    if len(x_fit) > 1:
        slope, _ = np.polyfit(x_fit, y_fit, 1)
        l_rad = -1.0 / slope
    else:
        l_rad = 0.0
        
    return l_rad, acf

def main():
    # --- Setup ---
    file_prefix = utils.find_prefix('-field_0.gkyl', '.')
    print(f"Using file prefix: {file_prefix}")

    try:
        fstart = int(input("fstart: "))
        fend = int(input("fend: "))
        step = int(input("step (default 1): ") or 1)
    except ValueError:
        print("Invalid input.")
        return

    z_idx = 1 # Mid-plane index
    z_str = 'zmid'

    # --- Geometry & Connection Length ---
    print("Calculating Geometry...")
    b_i_data = pg.GData(f"{file_prefix}-b_i.gkyl")
    _, _, z_vals, b_z = utils.func_data_3d(f"{file_prefix}-b_i.gkyl", 2)
    # Get mid-plane Y index
    mid_y = b_z.shape[1] // 2
    Lc = np.sum(b_z[:, mid_y, :], axis=1) * np.diff(z_vals)[0]
    Lc_ave = np.mean(Lc[len(Lc)//2:]) # Average over outer half
    print(f"Lc_ave = {Lc_ave:.4e}")

    # Load static data for VE calc
    jacgeo_data = pg.GData(f"{file_prefix}-jacobgeo.gkyl")
    bmag_data = pg.GData(f"{file_prefix}-bmag.gkyl")

    # --- Data Accumulation ---
    # We store raw 2D slices: List of arrays [Nx, Ny] -> will convert to [Nt, Nx, Ny]
    data_store = {
        'ne': [], 'ni': [], 'Te': [], 'Ti': [], 'phi': [],
        'VEx': [], 'VEy': [], 'p': [], 'Qpara': []
    }

    print(f"Processing frames {fstart} to {fend}...")
    
    # Store grid info only once
    x_vals, y_vals = None, None

    for tf in range(fstart, fend + 1, step):
        # Load GData objects
        elc_data = pg.data.GData(f"{file_prefix}-elc_BiMaxwellianMoments_{tf}.gkyl")
        ion_data = pg.data.GData(f"{file_prefix}-ion_BiMaxwellianMoments_{tf}.gkyl")
        phi_data = pg.data.GData(f"{file_prefix}-field_{tf}.gkyl")
        
        # Heat flux moments (assuming existence based on original script)
        q_data_files = [f"{file_prefix}-{s}_M3{d}_{tf}.gkyl" for s in ['elc','ion'] for d in ['par','perp']]
        q_moms = [pg.data.GData(f) for f in q_data_files]

        # 1. Get 2D Data (x, y) at z_idx
        # Note: func_data_2d returns (x, y, values)
        if x_vals is None:
             x_vals, y_vals, _ = utils.func_data_2d(elc_data, 0, z_idx)

        # Density
        _, _, ne_2d = utils.func_data_2d(elc_data, 0, z_idx)
        _, _, ni_2d = utils.func_data_2d(ion_data, 0, z_idx)

        # Temperature (Iso = (Tpar + 2Tperp)/3)
        _, _, Te_par = utils.func_data_2d(elc_data, 2, z_idx)
        _, _, Te_perp = utils.func_data_2d(elc_data, 3, z_idx)
        Te_2d = (Te_par + 2*Te_perp)/3.0 * ME / EV

        _, _, Ti_par = utils.func_data_2d(ion_data, 2, z_idx)
        _, _, Ti_perp = utils.func_data_2d(ion_data, 3, z_idx)
        Ti_2d = (Ti_par + 2*Ti_perp)/3.0 * MI / EV

        # Potential & ExB
        _, _, phi_2d = utils.func_data_2d(phi_data, 0, z_idx)
        VE_x, VE_y, _, _, _ = utils.func_calc_VE(phi_data, b_i_data, jacgeo_data, bmag_data, z_idx)

        # Heat Flux (Q) - Summing Par and Perp
        # Note: accessing -1 usually means average over z, or specific index in utils
        # Simplified: Just grab y-ave total Q for 1D profile, or 2D if needed. 
        # Keeping consistent with original logic: Q is stored for 1D profile
        _, qpara_elc = utils.func_data_yave(q_moms[0], 0, -1) 
        _, qpara_ion = utils.func_data_yave(q_moms[2], 0, -1)
        Q_total_1d = (ME/2 * qpara_elc) + (MI/2 * qpara_ion)

        # Append to storage
        data_store['ne'].append(ne_2d)
        data_store['ni'].append(ni_2d)
        data_store['Te'].append(Te_2d)
        data_store['Ti'].append(Ti_2d)
        data_store['phi'].append(phi_2d)
        data_store['VEx'].append(VE_x)
        data_store['VEy'].append(VE_y)
        data_store['p'].append((Te_2d + Ti_2d) * ne_2d)
        data_store['Qpara'].append(Q_total_1d)

    # --- Processing & Statistics ---
    print("Computing Statistics...")

    # Convert to Numpy Arrays: Shape [Time, X, Y]
    # Note: verify utils.func_data_2d output shape. Usually [X, Y].
    # Resulting arr shape: [Nt, Nx, Ny]
    arrs = {k: np.array(v) for k, v in data_store.items()}
    
    Nt, Nx, Ny = arrs['ne'].shape

    # 1. Mean Profiles (Average over Time (axis 0) and Y (axis 2))
    # Result shape: [Nx]
    means = {k: np.mean(v, axis=(0, 2)) for k, v in arrs.items() if v.ndim == 3}
    means['Qpara'] = np.mean(arrs['Qpara'], axis=0) # Already 1D

    # 2. Fluctuations (f_tilde = f - <f>_y,t)
    # We broadcast mean [Nx] across Time [Nt] and Y [Ny]
    flucs = {}
    for k in ['ne', 'Te', 'Ti', 'phi', 'VEx', 'VEy']:
        mean_profile = means[k][np.newaxis, :, np.newaxis] # Shape [1, Nx, 1]
        flucs[k] = arrs[k] - mean_profile

    # 3. RMS Levels (std dev over t, y)
    rms = {k: np.std(arrs[k], axis=(0, 2)) for k in ['ne', 'Te', 'Ti', 'phi']}
    
    # Normalized RMS
    norm_rms = {
        'dn': rms['ne'] / means['ne'],
        'dT': rms['Te'] / means['Te'],
        'dphi': rms['phi'] / means['Te'] # Normalized by Te usually
    }

    # 4. Transport Fluxes
    # Gamma = < ne_tilde * VEx_tilde >
    Gamma_x = np.mean(flucs['ne'] * flucs['VEx'], axis=(0, 2))
    
    # Q = n0 * < T_tilde * VEx_tilde > + T0 * Gamma
    # Electron Heat Flux
    Q_x_e = means['ne'] * np.mean(flucs['Te'] * flucs['VEx'], axis=(0, 2)) + means['Te'] * Gamma_x
    # Ion Heat Flux
    Q_x_i = means['ni'] * np.mean(flucs['Ti'] * flucs['VEx'], axis=(0, 2)) + means['Ti'] * Gamma_x

    # Reynolds Stress < VEx_tilde * VEy_tilde >
    Rey_stress = np.mean(flucs['VEx'] * flucs['VEy'], axis=(0, 2))
    Rey_force = -np.gradient(Rey_stress, x_vals)

    # 5. Higher Order Stats (Skewness, Kurtosis) & Correlation
    # Flatten Time and Y into one dimension for statistical functions: [Nt*Ny, Nx]
    dn_flat = flucs['ne'].transpose(0, 2, 1).reshape(-1, Nx) # [Samples, RadialPoints]
    
    skewness = skew(dn_flat, axis=0)
    kurt = kurtosis(dn_flat, axis=0)

    # Correlation Length
    l_rad, _ = calc_correlation_length(dn_flat, x_vals)

    # --- Output ---
    results = {
        # Profiles
        'neAve': means['ne'], 'TeAve': means['Te'], 'TiAve': means['Ti'], 
        'phiAve': means['phi'], 'QparaAve': means['Qpara'],
        # Fluxes
        'Gamma_x': Gamma_x, 'Qxe': Q_x_e, 'Qxi': Q_x_i,
        'Rey_stress': Rey_stress, 'Rey_force': Rey_force,
        # Fluctuations
        'dn_rms': rms['ne'], 'dn_norm': norm_rms['dn'],
        'dT_rms': rms['Te'], 'dT_norm': norm_rms['dT'],
        'dphi_rms': rms['phi'], 'dphi_norm': norm_rms['dphi'],
        # Stats
        'skew': skewness, 'kurt': kurt, 'l_rad': l_rad,
    }

    output_filename = f"diagnostics_{fstart}to{fend}_{z_str}.h5"
    metadata = {"fstart": fstart, "fend": fend, "zStr": z_str, "Lc": Lc_ave}
    
    print(f"Saving to {output_filename}...")
    utils.save_to_hdf5(output_filename, x_vals, results, metadata)
    print("Done.")

if __name__ == "__main__":
    main()