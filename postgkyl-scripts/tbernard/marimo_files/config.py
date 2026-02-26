import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import postgkyl as pg
from scipy.stats import skew, kurtosis, norm

# 1. Get the directory where THIS file (marimo_utils.py) lives
_script_dir = os.path.dirname(os.path.realpath(__file__))

# 2. Add that directory to the path so we can find 'utils.py' next to it
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# 3. Attempt to import
try:
    import utils
    print(f"Successfully loaded 'utils' from: {_script_dir}")
except ImportError as e:
    # This will now tell you the EXACT error (e.g. if a sub-dependency is missing)
    print(f"CRITICAL WARNING: 'utils.py' could not be imported from {_script_dir}.")
    print(f"Reason: {e}")

# --- Physical Constants ---
MP = 1.672623e-27
ME = 9.10938188e-31
EV = 1.602e-19

# --- Plotting Defaults ---
plt.rcParams.update({
    "font.size": 14,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (10, 6)
})

# --- Field Metadata (Master List) ---
# Maps internal keys to (Latex Label, Unit)
FIELD_INFO = {
    # Profiles
    'neAve':       {'lbl': r'$\langle n_e \rangle$', 'unit': '$m^{-3}$'},
    'TeAve':       {'lbl': r'$\langle T_e \rangle$', 'unit': 'eV'},
    'TiAve':       {'lbl': r'$\langle T_i \rangle$', 'unit': 'eV'},
    'phiAve':      {'lbl': r'$\langle \phi \rangle$', 'unit': 'V'},
    'QparaAve':    {'lbl': r'$\langle Q_\parallel \rangle$', 'unit': '$W/m^2$'},
    'VEyAve':      {'lbl': r'$\langle V_{E,y} \rangle$', 'unit': 'm/s'},
    'VEshearAve':  {'lbl': r'$| \langle \gamma_E \rangle|$', 'unit': '$s^{-1}$'},
    'ErAve':       {'lbl': r'$\langle E_r \rangle$', 'unit': 'V/m'},

    # Fluctuations
    'dn_norm':     {'lbl': r'$\delta n / n$', 'unit': ''},
    'dT_norm':     {'lbl': r'$\delta T_e / T_e$', 'unit': ''},
    'dphi_norm':   {'lbl': r'$e \delta \phi / T_e$', 'unit': ''},
    'dn_rms':      {'lbl': r'$\delta n_{rms}$', 'unit': '$m^{-3}$'},
    'dT_rms':      {'lbl': r'$\delta T_{e,rms}$', 'unit': 'eV'},
    'dphi_rms':    {'lbl': r'$\delta \phi_{rms}$', 'unit': 'V'},

    # Transport & Stats
    'Gamma_x':     {'lbl': r'$\Gamma_x$', 'unit': '$m^{-2}s^{-1}$'},
    'Qxe':         {'lbl': r'$Q_{e,x}$', 'unit': '$W/m^2$'},
    'Qxi':         {'lbl': r'$Q_{i,x}$', 'unit': '$W/m^2$'},
    'Rey_stress':  {'lbl': r'$\langle \tilde{v}_x \tilde{v}_y \rangle$', 'unit': '$m^2s^{-2}$'},
    'l_rad':       {'lbl': r'$L_{rad}$', 'unit': 'm'},
    'skew':        {'lbl': 'Skewness', 'unit': ''},
    'kurt':        {'lbl': 'Kurtosis', 'unit': ''},
    'I_flux':      {'lbl': r'$\langle \delta v_r \delta n^2 \rangle$', 'unit': ''},
    'gamma_mhd':   {'lbl': r'$\gamma_{MHD}$', 'unit': 's$^{-1}$'},
    'v_r':         {'lbl': r'$\langle v_r \rangle$', 'unit': 'm/s'},
    'v_eff':       {'lbl': r'$v_{eff},$', 'unit' : 'm/s'},
    'v_star_e':    {'lbl': r'$v_{*e}$', 'unit': 'm/s'},
    'v_star_i':    {'lbl': r'$v_{*i}$', 'unit': 'm/s'},
    'tau_ratio':   {'lbl': r'$\tau_\parallel / \tau_\perp$', 'unit': ''},

    # Geometry
    'Lc_ave':      {'lbl': r'$L_c$', 'unit': 'm'},
    'f_trap_mid':  {'lbl': r'Midplane Trapped Fraction $f_t$', 'unit': ''},

    # Integrated moments
    'int_n':       {'lbl': r'$\int n \, dV$', 'unit': 'Total Particles'},
    'int_en':      {'lbl': r'$\int \mathcal{E} \, dV$', 'unit': 'Total Energy [J]'},
    'int_nvz':     {'lbl': r'$\int n v_z \, dV$', 'unit': 'Momentum'},
}

# ==========================================
# Section 1: Mathematical Helpers
# ==========================================

def calc_radial_correlation(fluctuation_array, x_vals):
    """Calculates radial correlation length using autocorrelation."""
    dn = fluctuation_array
    dnSq = dn**2
    Nt, Nx = dn.shape
    
    # Define cutoffs (Trim edges)
    xi_min = Nx // 6
    xi_max = -xi_min
    
    # Radial grid spacing
    dx = x_vals[1] - x_vals[0]
    dxArray = np.arange(xi_min) * dx
    
    # Prepare slicing
    dn_cc = dn[:, xi_min:]
    sigma_sq_new = np.mean(dnSq[:, xi_min:xi_max], axis=0)
    
    # Calculate Correlation
    cc_dx = [
        np.mean(dn_cc[:, :xi_max] * dn_cc[:, dxi:(xi_max + dxi)], axis=0) / sigma_sq_new
        for dxi in range(xi_min)
    ]
    
    xVals_new = x_vals[xi_min:xi_max]
    cc_dx = np.array(cc_dx) 
    
    l_rad_trimmed = []
    
    for xi in range(len(xVals_new)):
        y = np.clip(cc_dx[:, xi], a_min=1e-3, a_max=None)
        try:
            # Weighted linear regression on log(ACF)
            coeffs = np.polyfit(dxArray, np.log(y), 1, w=np.sqrt(y))
            val = -1.0 / coeffs[0]
        except:
            val = np.nan
        l_rad_trimmed.append(val)

    # Pad result to match input dimension
    l_rad_full = np.full(Nx, np.nan)
    l_rad_full[xi_min:xi_max] = l_rad_trimmed
    
    return l_rad_full, cc_dx

def get_2d_fluctuations(data, subtract_mean='y'):
    """Calculates fluctuations: f - <f>."""
    if subtract_mean == 'y':
        # Assumes data is [X, Y]
        mean_profile = np.mean(data, axis=1, keepdims=True)
        fluc = data - mean_profile
        norm_fluc = fluc / (mean_profile + 1e-16)
    else:
        fluc = data - np.mean(data)
        norm_fluc = fluc / np.mean(data)
    return fluc, norm_fluc

# ==========================================
# Section 2: Core Processing Logic
# ==========================================

def get_max_frames_and_time(sim_def):
    """
    Scans the directory for the highest frame number, then opens
    ONLY that final file to extract the maximum physical time.
    """
    directory = sim_def.get('dir', '.')
    
    # Adjust this wildcard if your files end in .h5 instead of .bp
    files = glob.glob(os.path.join(directory, '*_[0-9]*.bp')) 
    
    if not files:
        return 0, 0.0
        
    max_frame = 0
    max_file = ""
    
    for f in files:
        try:
            # Extract the frame number from the filename
            frame_str = f.split('_')[-1].replace('.bp', '')
            frame_num = int(frame_str)
            if frame_num >= max_frame:
                max_frame = frame_num
                max_file = f
        except ValueError:
            continue
            
    # Open ONLY the final file to extract the physical time
    max_time = 0.0
    try:
        if max_file:
            # Load metadata using postgkyl
            data = pg.GData(max_file)
            # Extract time from the context dictionary
            max_time = data.ctx.get('time', 0.0) 
    except Exception as e:
        print(f"Warning: Could not read time from {max_file}: {e}")
        
    return max_frame, max_time

def process_simulation_run(sim_dir, fstart, fend, step=1, z_idx=1, amu=2.014, B_axis=2.0):
    """
    Processes a simulation directory to extract profiles, fluxes, and stats.
    Returns: x_vals, results_dict
    """
    MI_LOCAL = MP * amu # Calc local mass

    original_dir = os.getcwd()
    try:
        if not os.path.exists(sim_dir):
            print(f"Error: Directory {sim_dir} does not exist.")
            return None, None

        os.chdir(sim_dir)
        print(f"--> Analyzing: {sim_dir} | Frames: {fstart}-{fend} | AMU: {amu}")
        
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        
        # --- Geometry ---
        b_i_data = pg.GData(f"{file_prefix}-b_i.gkyl")
        _, _, z_vals, b_z = utils.func_data_3d(f"{file_prefix}-b_i.gkyl", 2)
        mid_y = b_z.shape[1] // 2
        Lc = np.sum(b_z[:, mid_y, :], axis=1) * np.diff(z_vals)[0]
        Lc_ave = np.mean(Lc[len(Lc)*2//3:])
        
        jacgeo_data = pg.GData(f"{file_prefix}-jacobgeo.gkyl")
        bmag_data = pg.GData(f"{file_prefix}-bmag.gkyl")

        # -- Calculate Local Shear ---
        try:
            gij_data = pg.GData(f"{file_prefix}-gij.gkyl")
            _, _, local_shear_xz = utils.func_calc_local_shear(gij_data)
        except Exception as e:
            print(f"Could not calculate local shear: {e}")
            local_shear_xz = None

        try:
            _, _, f_trap_xz = utils.func_calc_trapped_fraction(bmag_data)
            # Extract the outboard midplane value (z=0, which is the middle of the z-array)
            z_mid_idx = f_trap_xz.shape[1] // 2
            f_trap_mid = f_trap_xz[:, z_mid_idx]
        except Exception as e:
            print(f"Could not calculate trapped fraction: {e}")
            f_trap_xz = None
            f_trap_mid = None     

        # --- Storage Init ---
        data_store = {
            'ne': [], 'ni': [], 'Te': [], 'Ti': [], 'phi': [], 'phi_3d' : [],
            'VEx': [], 'VEy': [], 'VEshear': [], 'Er': [], 
            'p': [], 'Qpara': []
        }
        x_vals = None
        time_array = []

        # --- Time Loop ---
        for tf in range(fstart, fend + 1, step):
            try:
                # Load Objects
                elc_data = pg.data.GData(f"{file_prefix}-elc_BiMaxwellianMoments_{tf}.gkyl")
                ion_data = pg.data.GData(f"{file_prefix}-ion_BiMaxwellianMoments_{tf}.gkyl")
                phi_data = pg.data.GData(f"{file_prefix}-field_{tf}.gkyl")
                q_files = [f"{file_prefix}-{s}_M3{d}_{tf}.gkyl" for s in ['elc','ion'] for d in ['par','perp']]
                q_moms = [pg.data.GData(f) for f in q_files]

                try:
                    t_val = phi_data.ctx['time']
                except:
                    t_val = getattr(phi_data, 'time', tf)
                time_array.append(t_val)

                if x_vals is None: x_vals, _, _ = utils.func_data_2d(elc_data, 0, z_idx)

                # 2D Slices
                _, _, ne_2d = utils.func_data_2d(elc_data, 0, z_idx)
                _, _, ni_2d = utils.func_data_2d(ion_data, 0, z_idx)
                _, _, phi_2d = utils.func_data_2d(phi_data, 0, z_idx)

                # Temperatures
                _, _, Te_par = utils.func_data_2d(elc_data, 2, z_idx)
                _, _, Te_perp = utils.func_data_2d(elc_data, 3, z_idx)
                Te_2d = (Te_par + 2*Te_perp)/3.0 * ME / EV

                _, _, Ti_par = utils.func_data_2d(ion_data, 2, z_idx)
                _, _, Ti_perp = utils.func_data_2d(ion_data, 3, z_idx)
                Ti_2d = (Ti_par + 2*Ti_perp)/3.0 * MI_LOCAL / EV

                # phi
                _, _, _, phi_3d_vals = utils.func_data_3d(f"{file_prefix}-field_{tf}.gkyl", 0)

                # Flows & Shear
                VE_x, VE_y, _, VE_shear, Er = utils.func_calc_VE(phi_data, b_i_data, jacgeo_data, bmag_data, z_idx)

                # Heat Flux
                _, qpara_elc = utils.func_data_yave(q_moms[0], 0, -1) 
                _, qpara_ion = utils.func_data_yave(q_moms[2], 0, -1)
                # qpare_data = pg.data.GData(f"{file_prefix}-elc_M3par_{tf}.gkyl")
                # qpari_data = pg.data.GData(f"{file_prefix}-ion_M3par_{tf}.gkyl")
                # _, qpara_elc = utils.func_data_yave(qpare_data, 0, -1)
                # _, qpara_ion = utils.func_data_yave(qpari_data, 0, -1)
                Q_total_1d = (ME/2 * qpara_elc) + (MI_LOCAL/2 * qpara_ion)

                # Store
                data_store['ne'].append(ne_2d)
                data_store['ni'].append(ni_2d)
                data_store['Te'].append(Te_2d)
                data_store['Ti'].append(Ti_2d)
                data_store['phi'].append(phi_2d)
                data_store['phi_3d'].append(phi_3d_vals)
                data_store['VEx'].append(VE_x)
                data_store['VEy'].append(VE_y)
                data_store['VEshear'].append(np.abs(VE_shear))
                data_store['Er'].append(Er)
                data_store['Qpara'].append(Q_total_1d)
                
            except Exception as e:
                continue

        # --- Statistics ---
        if not data_store['ne']:
            print("No data processed.")
            return None, None

        arrs = {k: np.array(v) for k, v in data_store.items() if len(v) > 0}
        Nt, Nx, Ny = arrs['ne'].shape

        # Averages
        means = {}
        for k, v in arrs.items():
            if v.ndim == 3: means[k] = np.mean(v, axis=(0, 2))
            elif v.ndim == 2: means[k] = np.mean(v, axis=0)

        # Fluctuations
        flucs = {}
        for k in ['ne', 'Te', 'Ti', 'phi', 'VEx', 'VEy']:
            mean_profile = means[k][np.newaxis, :, np.newaxis]
            flucs[k] = arrs[k] - mean_profile

        # RMS
        rms = {k: np.std(arrs[k], axis=(0, 2)) for k in ['ne', 'Te', 'Ti', 'phi']}
        norm_rms = {
            'dn_norm': rms['ne'] / means['ne'],
            'dT_norm': rms['Te'] / means['Te'],
            'dphi_norm': rms['phi'] / means['Te']
        }

        Nt_fluc = flucs['ne'].shape[0]

        # --- NEW: Frequency Spectra (FFT over Time with Windowing) ---
        Nt_fluc = flucs['ne'].shape[0]
        
        # Calculate the physical time step (dt) between the loaded frames
        if len(time_array) > 1:
            dt = np.mean(np.diff(time_array))
        else:
            dt = 1.0 # Safe fallback if only 1 frame is loaded
            
        # Create a Hann window and reshape to [Nt, 1, 1] to broadcast across X and Y
        window = np.hanning(Nt_fluc).reshape(Nt_fluc, 1, 1)
        
        # Apply window to fluctuations BEFORE taking the FFT
        F_ne = np.fft.fft(flucs['ne'] * window, axis=0)
        F_Te = np.fft.fft(flucs['Te'] * window, axis=0)
        F_phi = np.fft.fft(flucs['phi'] * window, axis=0)
        
        # Power Spectra (Averaged over Y)
        P_ne = np.mean(np.abs(F_ne)**2, axis=2)
        P_Te = np.mean(np.abs(F_Te)**2, axis=2)
        P_phi = np.mean(np.abs(F_phi)**2, axis=2)
        
        # Cross Spectra (Averaged over Y)
        C_ne_phi = np.mean(F_ne * np.conj(F_phi), axis=2)
        C_Te_phi = np.mean(F_Te * np.conj(F_phi), axis=2)
        
        # Frequencies (Now with physical units!)
        freqs = np.fft.fftfreq(Nt_fluc, d=dt)

        # Calculate Parallel Mode Structure: RMS of phi over time and y-direction
        phi_3d_arr = arrs['phi_3d']
        phi_mode_structure = np.std(phi_3d_arr, axis=(0, 2)) # Result is [Nx, Nz]

        # Fluxes
        Gamma_x = np.mean(flucs['ne'] * flucs['VEx'], axis=(0, 2))
        Q_x_e = means['ne'] * np.mean(flucs['Te'] * flucs['VEx'], axis=(0, 2)) + means['Te'] * Gamma_x
        Q_x_i = means['ni'] * np.mean(flucs['Ti'] * flucs['VEx'], axis=(0, 2)) + means['Ti'] * Gamma_x
        Rey_stress = np.mean(flucs['VEx'] * flucs['VEy'], axis=(0, 2))
        Rey_force = -np.gradient(Rey_stress, x_vals)
        v_eff = Gamma_x/means['ne']

        # Higher Order Stats
        dn_flat = flucs['ne'].transpose(0, 2, 1).reshape(-1, Nx)
        dT_flat = flucs['Ti'].transpose(0, 2, 1).reshape(-1, Nx)
        dphi_flat = flucs['phi'].transpose(0, 2, 1).reshape(-1, Nx)
        print(np.shape(dn_flat))
        l_rad, _ = calc_radial_correlation(dn_flat, x_vals)
        skewness = skew(dn_flat, axis=0)
        kurt_val = kurtosis(dn_flat, axis=0)

        if 'VEx' in means:
            # If the mean was already calculated automatically
            v_r = means['VEx']
        elif 'VEx' in data_store:
            # Calculate the time-and-y average from the raw data
            v_r = np.mean(data_store['VEx'], axis=(0, 2))
        else:
            v_r = np.zeros(Nx) # Fallback if VEx isn't available

        # --- Turbulence Intensity Flux < dv_r * dn^2 > ---
        # Assuming VEx (radial ExB velocity) is stored in flucs
        if 'VEx' in flucs:
            dv_r = flucs['VEx']
            dn_sq = flucs['ne']**2
            # Average over time (axis 0) and binormal y (axis 2)
            I_flux = np.mean(dv_r * dn_sq, axis=(0, 2))
        else:
            I_flux = np.zeros(Nx) # Fallback if VEx isn't available

        # --- Interchange/Ballooning Growth Rate ---
        # gamma = c_s * sqrt(2 / (R * L_n))
        # 1. Calculate density gradient scale length L_n = - n / (dn/dx)
        ne_mean = means['ne']
        dx = x_vals[1] - x_vals[0]
        dne_dx = np.gradient(ne_mean, dx)
        
        # Avoid division by zero
        L_n = np.zeros_like(ne_mean)
        valid = np.abs(dne_dx) > 1e-10
        L_n[valid] = - ne_mean[valid] / dne_dx[valid]
        
        # 2. Calculate local sound speed (c_s)
        # Assuming EV and MI (ion mass) are defined constants in your utils
        c_s = np.sqrt(means['Te'] * 1.602e-19 / (2.014 * 1.672e-27)) # Deuterium
        
        # 3. Calculate Growth Rate
        # R is approximately the major radius (using x_vals as proxy if R_LCFS isn't explicitly passed here)
        # We use np.maximum to prevent sqrt of negative numbers where gradient is flat/reversed
        gamma_mhd = c_s * np.sqrt(np.maximum(0.0, 2.0 / (np.abs(x_vals) * np.abs(L_n) + 1e-8)))

        # ---  Diamagnetic Drift Velocities (v_*e and v_*i) ---
        Te_mean = means['Te']
        Ti_mean = means['Ti']
        
        # Calculate temperature gradients
        dTe_dx = np.gradient(Te_mean, dx)
        dTi_dx = np.gradient(Ti_mean, dx)
        
        valid_n = ne_mean > 1e-12
        v_star_e = np.zeros_like(ne_mean)
        v_star_i = np.zeros_like(ne_mean)
        
        # Now using the dynamically passed B_axis
        v_star_e[valid_n] = -(dTe_dx[valid_n] + (Te_mean[valid_n] / ne_mean[valid_n]) * dne_dx[valid_n]) / B_axis
        v_star_i[valid_n] =  (dTi_dx[valid_n] + (Ti_mean[valid_n] / ne_mean[valid_n]) * dne_dx[valid_n]) / B_axis

        # parallel and perp transit
        tau_para = Lc_ave / c_s
        tau_perp = L_n / v_eff
        tau_ratio = np.clip(tau_para / tau_perp, a_min=0, a_max=100)
        
        # Integrated moments
        t_elc, int_elc = load_integrated_moms(sim_dir, 'elc')
        t_ion, int_ion = load_integrated_moms(sim_dir, 'ion')

        # Pack Results
        results = {
            'neAve': means['ne'], 'TeAve': means['Te'], 'TiAve': means['Ti'], 
            'phiAve': means['phi'], 'QparaAve': means['Qpara'],
            'VEyAve': means['VEy'], 'VEshearAve': means['VEshear'], 'ErAve': means['Er'],
            'Gamma_x': Gamma_x, 'Qxe': Q_x_e, 'Qxi': Q_x_i, 'v_eff': v_eff,
            'Rey_stress': Rey_stress, 'Rey_force': Rey_force,
            'dn_rms': rms['ne'], 'dn_norm': norm_rms['dn_norm'],
            'dT_rms': rms['Te'], 'dT_norm': norm_rms['dT_norm'],
            'dphi_rms': rms['phi'], 'dphi_norm': norm_rms['dphi_norm'],
            'skew': skewness, 'kurt': kurt_val, 'l_rad': l_rad,
            'Lc_ave': Lc_ave,
            'z_vals': z_vals,
            'local_shear_xz': local_shear_xz,
            'phi_mode_structure': phi_mode_structure,
            'f_trap_xz': f_trap_xz,
            'f_trap_mid': f_trap_mid,
            'dn_flat': dn_flat,     
            'dT_flat': dT_flat,     
            'dphi_flat': dphi_flat,
            'I_flux': I_flux,
            'gamma_mhd': gamma_mhd,
            'v_r': v_r,
            'v_star_e': v_star_e,
            'v_star_i': v_star_i,
            'tau_ratio' : tau_ratio,

            'freqs': freqs,
            'P_ne': P_ne, 'P_Te': P_Te, 'P_phi': P_phi,
            'C_ne_phi': C_ne_phi, 'C_Te_phi': C_Te_phi,

            'time_series_t': t_elc,
            'int_n_elc': int_elc[:, 0] if int_elc is not None else None,
            'int_en_elc': int_elc[:, 2] if int_elc is not None else None,
            'int_n_ion': int_ion[:, 0] if int_ion is not None else None,
            'int_en_ion': int_ion[:, 2] if int_ion is not None else None,

            # Store dn_flat for PDF plotting
            'dn_flat': dn_flat, 
            'dT_flat' : dT_flat,
            'dphi_flat' : dphi_flat,
        }
        
        return x_vals, results

    except Exception as e:
        print(f"Error processing {sim_dir}: {e}")
        return None, None
    finally:
        os.chdir(original_dir)

def load_integrated_moms(sim_path, species='elc'):
    """
    Loads the time-series of integrated moments.
    Returns: time_steps, data_array (components: 0=n, 1=nvz, 2=energy)
    """
    original_dir = os.getcwd()
    try:
        os.chdir(sim_path)
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        fname = f"{file_prefix}-{species}_integrated_moms.gkyl"
        
        if not os.path.exists(fname):
            print(f"Warning: {fname} not found.")
            return None, None
            
        data = pg.GData(fname)
        time = data.get_grid()[0]
        values = data.get_values()
        
        return time, values
    except Exception as e:
        print(f"Error loading integrated moms: {e}")
        return None, None
    finally:
        os.chdir(original_dir)

def load_datasets(sim_dict, fstart, fend, step=1, amu=2.014, B_axis=2.0):
    """
    Batch processes all simulations in the dictionary.
    Returns a data dictionary ready for plot_1d_comparison.
    """
    processed_data = {}
    print(f"Batch processing {len(sim_dict)} simulations...")

    for label, meta in sim_dict.items():
        # Handle case where user just provided a path string instead of dict
        if isinstance(meta, str):
            path = meta
            color = None
            ls = '-'
        else:
            path = meta.get('path')
            color = meta.get('color', None)
            ls = meta.get('ls', '-')

        # Run the heavy calculation
        x, res = process_simulation_run(path, fstart, fend, step, amu=amu, B_axis=B_axis)
        
        if x is not None and res is not None:
            processed_data[label] = {
                'x': x,
                'results': res,
                'color': color,
                'ls': ls
            }
            
    print("Batch processing complete.")
    return processed_data

# ==========================================
# Section 3: Snapshot Loading
# ==========================================

def load_2d_snapshot(sim_path, frame, field_name, z_idx=1, amu=2.014):
    """Loads a single 2D snapshot."""
    MI_LOCAL = MP * amu 
    original_dir = os.getcwd()
    try:
        os.chdir(sim_path)
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        
        field_map = {
            'ne':  ('elc_BiMaxwellianMoments', 0, 'scalar'),
            'ni':  ('ion_BiMaxwellianMoments', 0, 'scalar'),
            'phi': ('field', 0, 'scalar'),
            'Te':  ('elc_BiMaxwellianMoments', [2, 3], 'iso_temp'),
            'Ti':  ('ion_BiMaxwellianMoments', [2, 3], 'iso_temp'),
        }
        
        suffix, comp, ftype = field_map.get(field_name, (None, None, None))
        if suffix is None: return None, None, None

        filename = f"{file_prefix}-{suffix}_{frame}.gkyl"
        if not os.path.exists(filename): return None, None, None
        
        data_obj = pg.data.GData(filename)
        
        if ftype == 'scalar':
            x, y, vals = utils.func_data_2d(data_obj, comp, z_idx)
            return x, y, vals
        elif ftype == 'iso_temp':
            x, y, val_par = utils.func_data_2d(data_obj, comp[0], z_idx)
            _, _, val_perp = utils.func_data_2d(data_obj, comp[1], z_idx)
            mass = ME if 'elc' in suffix else MI_LOCAL
            val_iso = (val_par + 2 * val_perp) / 3.0 * mass / EV
            return x, y, val_iso
            
    except Exception as e:
        print(f"Error loading snapshot: {e}")
        return None, None, None
    finally:
        os.chdir(original_dir)

# ==========================================
# Section 4: Plotting Functions
# ==========================================

def plot_saturation(data_dict, species='elc'):
    """
    Plots Total Particles and Total Energy vs Time to check for saturation.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for label, d in data_dict.items():
        res = d['results']
        t = res.get('time_series_t')
        
        if t is None: continue
        
        n_int = res.get(f'int_n_{species}')
        en_int = res.get(f'int_en_{species}')
        
        c = d.get('color', None)
        ls = d.get('ls', '-')

        if n_int is not None:
            # Normalize to initial value to see percentage change
            ax1.plot(t * 1e6, n_int / n_int[0], label=label, color=c, linestyle=ls)
            
        if en_int is not None:
            ax2.plot(t * 1e6, en_int / en_int[0], label=label, color=c, linestyle=ls)

    ax1.set_ylabel("Rel. Particle Count")
    ax1.set_title(f"Saturation Check ({species})")
    ax1.legend()
    
    ax2.set_ylabel("Rel. Total Energy")
    ax2.set_xlabel("Time ($\mu s$)")
    
    plt.tight_layout()
    plt.show()

def plot_1d_profiles(data_dict, fields_to_plot, lcfs_shift=0.0, r_lcfs=2.17, r_axis=1.65, trim_pts=0):
    """
    Plots 1D radial profiles mapped to normalized rho in a multi-column grid.
    """
    num_fields = len(fields_to_plot)
    if num_fields == 0: return None

    # Set up a 2-column grid layout
    num_cols = 2
    num_rows = int(np.ceil(num_fields / num_cols))
    
    # Scale figure height based on the number of rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 3.5 * num_rows), layout='constrained')
    
    # Force axs to be a flat 1D array for easy iteration, regardless of grid shape
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.flatten()

    # Calculate the physical minor radius for the denominator
    minor_radius = r_lcfs - r_axis
    if minor_radius <= 0:
        minor_radius = 1.0 # Fallback to prevent divide-by-zero

    for i, field in enumerate(fields_to_plot):
        ax = axs[i]
        
        # Use a fallback dictionary if FIELD_INFO isn't perfectly matched
        info = FIELD_INFO.get(field, {'lbl': field, 'unit': ''})
        ylabel = f"{info['lbl']} [{info['unit']}]" if info['unit'] else info['lbl']

        for label, d in data_dict.items():
            if 'results' in d and field in d['results']:
                raw_x = d['x']
                rho_norm = 1.0 + (raw_x - lcfs_shift) / minor_radius
                y_data = d['results'][field]

def plot_1d_profiles(data_dict, fields_to_plot, lcfs_shift=0.0, r_lcfs=2.17, r_axis=1.65, trim_pts=0):
    """
    Plots 1D radial profiles mapped to normalized rho in a multi-column grid.
    Includes boundary trimming to remove sponge layer artifacts.
    """
    num_fields = len(fields_to_plot)
    if num_fields == 0: return None

    num_cols = 2
    num_rows = int(np.ceil(num_fields / num_cols))
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 3.5 * num_rows), layout='constrained')
    
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.flatten()

    minor_radius = r_lcfs - r_axis
    if minor_radius <= 0:
        minor_radius = 1.0 

    for i, field in enumerate(fields_to_plot):
        ax = axs[i]
        
        info = FIELD_INFO.get(field, {'lbl': field, 'unit': ''})
        ylabel = f"{info['lbl']} [{info['unit']}]" if info['unit'] else info['lbl']

        for label, d in data_dict.items():
            if 'results' in d and field in d['results']:
                raw_x = d['x']
                rho_norm = 1.0 + (raw_x - lcfs_shift) / minor_radius
                y_data = d['results'][field]
                
                if trim_pts > 0:
                    rho_norm = rho_norm[trim_pts:-trim_pts]
                    y_data = y_data[trim_pts:-trim_pts]

                c = d.get('color', None)
                ls = d.get('ls', '-')
                ax.plot(rho_norm, y_data, color=c, linestyle=ls, label=label, lw=2)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(r'$\rho$') # Put x-axis label on EVERY plot
        ax.grid(True, alpha=0.3)
        
        # The LCFS is strictly at rho = 1.0
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.6, label='LCFS')
        
        if i == 0: ax.legend()

    # Turn off any unused, empty subplots if the number of fields is odd
    for j in range(num_fields, len(axs)):
        axs[j].set_visible(False)

    return fig

def plot_2d_comparison(sim_data, frame, field_name, mode='total', lcfs_shift=0.0, vlims=None, amu=2.014):
    """
    Plots side-by-side 2D heatmaps.
    sim_data: Dictionary {'Label': {'path': '...', ...}} OR {'Label': 'path'}
    """
    n_sims = len(sim_data)
    fig, axs = plt.subplots(1, n_sims, figsize=(6 * n_sims, 5), sharey=True)
    if n_sims == 1: axs = [axs]
    
    labels_map = {
        'ne': {'tot': r'$n_e$', 'fluc': r'$\delta n_e / n_e$'},
        'ni': {'tot': r'$n_e$', 'fluc': r'$\delta n_e / n_e$'},
        'Te': {'tot': r'$T_e$', 'fluc': r'$\delta T_e / T_e$'},
        'Ti': {'tot': r'$T_i$', 'fluc': r'$\delta T_i / T_i$'},
        'phi':{'tot': r'$\phi$', 'fluc': r'$\delta \phi$'}
    }
    lbl = labels_map.get(field_name, {'tot': field_name, 'fluc': field_name})

    for i, (label, item) in enumerate(sim_data.items()):
        ax = axs[i]
        
        # Extract Path (Handle Dict vs String)
        path = item.get('path') if isinstance(item, dict) else item
            
        x, y, data = load_2d_snapshot(path, frame, field_name, amu=amu)
        
        if x is None: 
            ax.text(0.5, 0.5, "Data Not Found", ha='center')
            continue
        
        if mode == 'fluctuation':
            fluc, norm_fluc = get_2d_fluctuations(data, subtract_mean='y')
            plot_data = fluc if field_name == 'phi' else norm_fluc
            title_str = f"{label}: {lbl['fluc']}"
            cmap = 'RdBu_r'
            if vlims is None:
                mx = np.max(np.abs(plot_data))
                vmin, vmax = -mx, mx
            else:
                vmin, vmax = vlims[0], vlims[1]
        else:
            plot_data = data
            title_str = f"{label}: {lbl['tot']}"
            cmap = 'inferno' if field_name != 'phi' else 'viridis'
            vmin, vmax = (None, None) if vlims is None else (vlims[0], vlims[1])

        im = ax.pcolormesh(x - lcfs_shift, y, plot_data.T, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title_str)
        ax.set_xlabel(r'$R - R_{sep}$ (m)')
        if i == 0: ax.set_ylabel('Y (m)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.grid(False)

    plt.tight_layout()
    plt.show()

def list_fields():
    """Prints a formatted list of all available fields."""
    print(f"{'Key':<15} | {'Unit':<15} | {'Description'}")
    print("-" * 50)
    
    # Sort keys alphabetically
    for key in sorted(FIELD_INFO.keys()):
        meta = FIELD_INFO[key]
        # Strip latex $ for cleaner printing
        clean_lbl = meta['lbl'].replace('$', '')
        clean_unit = meta['unit'].replace('$', '')
        print(f"{key:<15} | {clean_unit:<15} | {clean_lbl}")

def scan_for_negativity(sim_path, frame, field_name='ne', z_idx=1, amu=2.014):
    """
    Finds the minimum value of a field and its location.
    Returns: min_val, (r_loc, y_loc), (ix, iy), msg
    """
    # Load the field data using existing function
    x, y, data = load_2d_snapshot(sim_path, frame, field_name, z_idx, amu)
    
    if data is None:
        return None, None, None, "Data load failed."

    # Find Min Value and Index
    min_val = np.nanmin(data)
    flat_idx = np.nanargmin(data)
    # unravel_index returns tuple (idx_x, idx_y) for 2D array
    # Note: data from utils is usually [Nx, Ny] or [Ny, Nx]. 
    # Based on plotting code: pcolormesh(x, y, data.T) -> data is [x, y]
    ix, iy = np.unravel_index(flat_idx, data.shape)
    
    # Get Physical Coordinates
    r_loc = x[ix]
    y_loc = y[iy]
    
    msg = f"Min {field_name}: {min_val:.4e} at R={r_loc:.4f}, Y={y_loc:.4f} (Idx: {ix}, {iy})"
    
    return min_val, (r_loc, y_loc), (ix, iy), msg

def plot_qpara_sol(data_dict, lcfs_shift=0.0, x_trim=0, fit_rmin=0.0, fit_rmax=0.05):
    """
    Plots the parallel heat flux (Qpara) in the SOL region (R > R_LCFS).
    Fits the decay length (lambda_q) only within the specified [fit_rmin, fit_rmax] window.
    """
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    qpara_info = FIELD_INFO.get('QparaAve', {'lbl': 'Qpara', 'unit': ''})

    for label, d in data_dict.items():
        x_vals = d['x']
        results = d['results']
        
        qpara_data = results.get('QparaAve', None)
        if qpara_data is None:
            print(f"Warning: 'QparaAve' not found in results for {label}. Skipping.")
            continue
        
        x_plot = x_vals - lcfs_shift

        if x_trim > 0:
            x_plot = x_plot[x_trim:-x_trim]
            qpara_data = qpara_data[x_trim:-x_trim]

        sol_mask = x_plot > 0.0
        x_sol = x_plot[sol_mask]
        qpara_sol = qpara_data[sol_mask]

        if len(x_sol) == 0:
            continue

        c = d.get('color', None)
        ls = d.get('ls', '-')
        plot_label = label

        # --- NEW: Create a mask strictly for the user-defined fitting window ---
        fit_mask = (x_sol >= fit_rmin) & (x_sol <= fit_rmax)
        valid_q = qpara_sol > 0
        combined_mask = fit_mask & valid_q
        
        if np.sum(combined_mask) > 3:
            x_fit = x_sol[combined_mask]
            q_fit_data = qpara_sol[combined_mask]
            
            # Fit ln(q) = - x / lambda_q + C
            p = np.polyfit(x_fit, np.log(q_fit_data), 1)
            
            if p[0] < 0:
                lambda_q_m = -1.0 / p[0]
                lambda_q_mm = lambda_q_m * 1000.0
                
                # Generate and plot the fit line ONLY over the fitting window
                q_fit_line = np.exp(np.polyval(p, x_fit))
                ax.plot(x_fit, q_fit_line, color=c, linestyle=':', lw=3.5, alpha=1.0)
                
                plot_label = f"{label} ($\lambda_q$={lambda_q_mm:.1f} mm)"
            else:
                plot_label = f"{label} (No decay found)"

        # Plot the full SOL data
        ax.plot(x_sol, qpara_sol, label=plot_label, color=c, linestyle=ls, lw=2)

    ax.set_title(f"{qpara_info['lbl']} in SOL with Decay Fits")
    ax.set_xlabel(r"$R - R_{LCFS}$ (m)")
    ax.set_ylabel(f"{qpara_info['lbl']} ({qpara_info['unit']})")
    
    #ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    return fig

def plot_pdf_slice(data_dict, field_key='dn_flat', x_target_idx=None, r_target_val=None,
                   bins='auto', density=True, compare_gaussian=False,
                   x_trim=0, lcfs_shift=0.0):
    """
    Plots the Probability Density Function (PDF) of fluctuations at a specific
    radial location (x_target_idx or r_target_val) for one or more simulations.

    Args:
        data_dict (dict): Dictionary of processed simulation data.
                          Each entry should have 'x' and 'results' (containing field_key).
        field_key (str): Key for the flattened fluctuation data (e.g., 'dn_flat').
        x_target_idx (int, optional): The integer index of the radial point to plot.
                                      If None, r_target_val is used.
        r_target_val (float, optional): The approximate physical R-value to plot.
                                        If x_target_idx is None, finds the closest index.
        bins (int or str): Number of histogram bins or 'auto'.
        density (bool): If True, normalized histogram to form PDF.
        compare_gaussian (bool): If True, plots a standard normal PDF for comparison.
        x_trim (int): Number of points to trim from edges of x-axis (for plotting).
        lcfs_shift (float): Shift in x-axis for plotting (e.g., to align LCFS).
    """
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    for label, d in data_dict.items():
        x_vals = d['x']
        results = d['results']
        
        fluctuation_data_flat = results.get(field_key, None)
        if fluctuation_data_flat is None:
            print(f"Warning: '{field_key}' not found in results for {label}. Skipping.")
            continue
        
        if x_target_idx is None and r_target_val is not None:
            # Find the closest index to r_target_val
            # Ensure x_vals is shifted before finding index if lcfs_shift is applied to plots
            x_vals_shifted = x_vals - lcfs_shift
            idx_closest = np.argmin(np.abs(x_vals_shifted - r_target_val))
            current_x_idx = idx_closest
        elif x_target_idx is not None:
            current_x_idx = x_target_idx
        else:
            print("Error: Either 'x_target_idx' or 'r_target_val' must be provided.")
            continue

        if current_x_idx >= fluctuation_data_flat.shape[1] or current_x_idx < 0:
            print(f"Error: Radial index {current_x_idx} is out of bounds for {label}. Skipping.")
            continue

        # Extract the 1D array of fluctuations at the chosen radial index
        data_at_x = fluctuation_data_flat[:, current_x_idx]
        
        # Calculate mean and standard deviation for normalization for PDF
        # We normalize to mean=0 and std=1 for plotting PDFs for comparison
        mean_data_at_x = np.mean(data_at_x)
        std_data_at_x = np.std(data_at_x)
        
        # Avoid division by zero
        if std_data_at_x < 1e-10: 
            print(f"Warning: Standard deviation at R={x_vals[current_x_idx]-lcfs_shift:.3f} for {label} is near zero. Skipping PDF.")
            continue

        # Normalize the fluctuations to mean 0, std 1
        normalized_data = (data_at_x - mean_data_at_x) / std_data_at_x

        # Calculate the histogram (PDF)
        hist, bin_edges = np.histogram(normalized_data, bins=bins, density=density)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot the PDF
        deltR = r"$R - R_{LCFS}$"
        ax.plot(bin_centers, hist, label=f"{label}, {deltR}={x_vals[current_x_idx]-lcfs_shift:.3f}m",
                color=d.get('color', None), linestyle=d.get('ls', '-'))

    # Compare to a standard normal distribution if requested
    if compare_gaussian:
        xmin, xmax = ax.get_xlim()
        x_gaussian = np.linspace(xmin, xmax, 500)
        ax.plot(x_gaussian, norm.pdf(x_gaussian), 'k--', label='Standard Normal PDF')

    ax.set_title(f"PDF of Normalized Fluctuations ({FIELD_INFO.get(field_key, {}).get('lbl', field_key)})")
    ax.set_xlabel(r"$(\delta f - \langle \delta f \rangle) / \sigma_{\delta f}$") # Normalized fluctuation label
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.show()

def plot_distf_slice(sim_path, frame, species, ix, iy, z_idx=0):
    """
    Plots the cell-average distribution function in velocity space 
    at a specific spatial index (ix, iy).
    
    Assumes file naming: [prefix]-[species]_[frame].gkyl
    """
    original_dir = os.getcwd()
    try:
        os.chdir(sim_path)
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        
        # Construct distf filename
        filename_f = f"{file_prefix}-{species}_{frame}.gkyl"
        filename_vel = f"{file_prefix}-{species}_mapc2p_vel.gkyl"
        filename_jacobvel = f"{file_prefix}-{species}_jacobvel.gkyl"
        
        if not os.path.exists(filename_f):
            return None, f"Distribution file not found: {filename_f}"
            
        f_data = pg.GData(filename_f, mapc2p_vel_name=filename_vel)
        Jv_data = pg.GData(filename_jacobvel, mapc2p_vel_name=filename_vel)
        f_c = f_data.get_values()
        Jv_c = Jv_data.get_values()
        f_data._values = f_c/Jv_c
        dg = pg.GInterpModal(f_data, poly_order=1, basis_type="gkhyb")
        xInt, distf = dg.interpolate()

        Xnodal = [np.outer(xInt[3], np.ones(np.shape(xInt[4]))), \
            np.outer(np.ones(np.shape(xInt[3])), xInt[4])]

        if z_idx != 0:
            z_idx = distf.shape[2] // 2

        f_slice = np.squeeze(distf[ix, iy, z_idx, :, :])
        xlabel = r"$v_{\parallel}$"
        ylabel = r"$\mu$"
            
        # Create Figure
        fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
        
        # Plot Heatmap (Transpose for pcolormesh: [vpar, mu])
        # v1 is usually vpar coordinates, v2 is mu coordinates
        im = ax.pcolormesh(Xnodal[0], Xnodal[1], f_slice, cmap='plasma', shading='auto')
        
        ax.set_title(f"{species} $f(v)$ at x={xInt[0][ix]}, y={xInt[1][iy]}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, label='Cell Avg Amplitude')
        
        return fig, "Success"

    except Exception as e:
        return None, f"Error plotting distf: {e}"
    finally:
        os.chdir(original_dir)

def plot_parallel_mode_structure(data_dict, lcfs_shift=0.0, r_target_val=0.0):
    """Plots the local shear and parallel ITG mode structure for a specific R value."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    for label, d in data_dict.items():
        x = d['x'] - lcfs_shift
        res = d['results']
        
        if 'z_vals' not in res or 'phi_mode_structure' not in res:
            continue
            
        z = res['z_vals']
        phi_struct = res['phi_mode_structure']
        shear = res.get('local_shear_xz', None)
        
        # Find closest radial index
        x_idx = np.argmin(np.abs(x - r_target_val))
        actual_r = x[x_idx]
        
        c = d.get('color', None)
        ls = d.get('ls', '-')
        
        # Plot Mode Structure (Solid)
        ax1.plot(z, phi_struct[x_idx, :], color=c, linestyle=ls, label=f"{label} (R={actual_r:.2f})")
        
        # Plot Local Shear (Dashed)
        if shear is not None:
            ax2.plot(z, shear[x_idx, :], color=c, linestyle='--', alpha=0.6)

    ax1.set_xlabel(r"Parallel Coordinate $z$ (m)")
    ax1.set_ylabel(r"Mode Amplitude $\delta \phi_{rms}$", color='black')
    ax2.set_ylabel(r"Local Magnetic Shear $s_{loc}$", color='gray')
    ax1.set_title("Parallel ITG Mode Structure vs. Local Shear")
    
    # Add dummy lines for legend to explain line styles
    import matplotlib.lines as mlines
    mode_line = mlines.Line2D([], [], color='black', linestyle='-', label=r'$|\phi|$ Amplitude')
    shear_line = mlines.Line2D([], [], color='gray', linestyle='--', label=r'$s_{loc}$')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles=lines1 + [mode_line, shear_line], loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ==========================================
# Section 5: Spectral Analysis
# ==========================================
import h5py
from matplotlib.gridspec import GridSpec

def compute_fsa_spectra(data_yz, J_z):
    mean_y = np.mean(data_yz, axis=0, keepdims=True)
    fluc = data_yz - mean_y
    fft_k = np.fft.fft(fluc, axis=0)
    power_kz = np.abs(fft_k)**2
    if J_z is not None:
        numerator = np.sum(power_kz * J_z[np.newaxis, :], axis=1)
        return numerator / np.sum(J_z)
    return np.mean(power_kz, axis=1)

def compute_fsa_cross_spectra(data1_yz, data2_yz, J_z, weighting=None):
    f1 = data1_yz - np.mean(data1_yz, axis=0, keepdims=True)
    f2 = data2_yz - np.mean(data2_yz, axis=0, keepdims=True)
    F1 = np.fft.fft(f1, axis=0)
    F2 = np.fft.fft(f2, axis=0)
    cross = np.conj(F1) * F2
    if weighting is not None:
        cross = cross * weighting[np.newaxis, :]
    if J_z is not None:
        return np.sum(cross * J_z[np.newaxis, :], axis=1) / np.sum(J_z)
    return np.mean(cross, axis=1)

def fsa_mean(data_yz, J_z):
    zonal = np.mean(data_yz, axis=0)
    if J_z is not None:
        return np.sum(zonal * J_z) / np.sum(J_z)
    return np.mean(zonal)

def process_spectra_for_sim(sim_dir, fstart, fend, x_idx, step=1, amu=2.014):
    """Processes the k_y spectra for a single simulation at a specific radial index."""
    MI_LOCAL = MP * amu
    original_dir = os.getcwd()
    try:
        if not os.path.exists(sim_dir): return None
        os.chdir(sim_dir)
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        
        # Geometry
        try:
            jac_data = pg.GData(f"{file_prefix}-jacobgeo.gkyl")
            dg_j = pg.GInterpModal(jac_data, 1, 'ms')
            J_z = np.mean(dg_j._getRawModal(0)[x_idx, :, :, 0] / (2**1.5), axis=0)
            
            bmag_data = pg.GData(f"{file_prefix}-bmag.gkyl")
            dg_b = pg.GInterpModal(bmag_data, 1, 'ms')
            B_z = np.mean(dg_b._getRawModal(0)[x_idx, :, :, 0] / (2**1.5), axis=0)
            B0 = np.mean(B_z)
        except:
            J_z, B_z, B0 = None, None, 1.0

        acc = {'phi': [], 'n': [], 'Te': [], 'Ti': [], 'Tpar_e': [], 'Tperp_e': [], 
               'Qe_conv': [], 'Qe_cond': [], 'Qi_conv': [], 'Qi_cond': [],
               'cross_n_phi': [], 'cross_Ti_phi': [], 'cross_Tperp_phi': [], 'cross_Tpar_phi': [],
               'Te_bg': [], 'ne_bg': [], 'ni_bg': [], 'Ti_bg': []}

        # Setup spatial grid for ky
        gdata_init = pg.GData(f"{file_prefix}-field_{fstart}.gkyl")
        y_vals = gdata_init.get_grid()[1]
        Ly = abs(y_vals[-1] - y_vals[0])
        Ny = len(y_vals) - 1 if len(y_vals) > 1 else 1 # Number of cells
        ky_raw = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
        norm = 1.0 / Ny 

        for tf in range(fstart, fend + 1, step):
            try:
                phi_dat = pg.GData(f"{file_prefix}-field_{tf}.gkyl")
                elc_dat = pg.GData(f"{file_prefix}-elc_BiMaxwellianMoments_{tf}.gkyl")
                ion_dat = pg.GData(f"{file_prefix}-ion_BiMaxwellianMoments_{tf}.gkyl")
            except: continue

            def get_comp(gdat, c):
                dg = pg.GInterpModal(gdat, 1, 'ms')
                return dg._getRawModal(c)[x_idx, :, :, 0] / (2**1.5)

            phi = get_comp(phi_dat, 0)
            ne = get_comp(elc_dat, 0)
            Te_par = get_comp(elc_dat, 2) / EV * ME
            Te_perp = get_comp(elc_dat, 3) / EV * ME
            Te = (Te_par + 2*Te_perp)/3.0
            
            ni = get_comp(ion_dat, 0)
            Ti_par = get_comp(ion_dat, 2) / EV * MI_LOCAL
            Ti_perp = get_comp(ion_dat, 3) / EV * MI_LOCAL
            Ti = (Ti_par + 2*Ti_perp)/3.0

            acc['ne_bg'].append(fsa_mean(ne, J_z))
            acc['Te_bg'].append(fsa_mean(Te, J_z))
            acc['ni_bg'].append(fsa_mean(ni, J_z))
            acc['Ti_bg'].append(fsa_mean(Ti, J_z))

            for k, v in zip(['phi', 'n', 'Te', 'Ti', 'Tpar_e', 'Tperp_e'], [phi, ne, Te, Ti, Te_par, Te_perp]):
                acc[k].append(compute_fsa_spectra(v, J_z))

            inv_B = 1.0 / (B_z + 1e-16) if B_z is not None else None
            acc['Qe_conv'].append(compute_fsa_cross_spectra(ne, phi, J_z, inv_B))
            acc['Qe_cond'].append(compute_fsa_cross_spectra(Te, phi, J_z, inv_B))
            acc['Qi_conv'].append(compute_fsa_cross_spectra(ni, phi, J_z, inv_B))
            acc['Qi_cond'].append(compute_fsa_cross_spectra(Ti, phi, J_z, inv_B))

            acc['cross_n_phi'].append(compute_fsa_cross_spectra(ne, phi, J_z))
            acc['cross_Ti_phi'].append(compute_fsa_cross_spectra(Ti, phi, J_z))
            acc['cross_Tperp_phi'].append(compute_fsa_cross_spectra(Te_perp, phi, J_z))
            acc['cross_Tpar_phi'].append(compute_fsa_cross_spectra(Te_par, phi, J_z))

        if not acc['phi']: return None

        Te_bg_avg = np.mean(acc['Te_bg'])
        cs = np.sqrt(Te_bg_avg * EV / MI_LOCAL)
        omega_ci = (EV * B0) / MI_LOCAL if B0 != 0 else 1.0
        rho_s = cs / omega_ci

        final_data = {'ky': ky_raw, 'rho_s': rho_s}
        for k in ['phi', 'n', 'Te', 'Ti', 'Tpar_e', 'Tperp_e']:
            final_data[f'{k}_amp'] = np.sqrt(np.mean(acc[k], axis=0)) * norm

        factor_J = 1.5 * EV * norm**2
        def calc_flux(bg_avg, cross_list):
            return bg_avg * ky_raw * np.imag(np.mean(acc[cross_list], axis=0)) * factor_J

        final_data['Qe_tot'] = calc_flux(Te_bg_avg, 'Qe_conv') + calc_flux(np.mean(acc['ne_bg']), 'Qe_cond')
        final_data['Qi_tot'] = calc_flux(np.mean(acc['Ti_bg']), 'Qi_conv') + calc_flux(np.mean(acc['ni_bg']), 'Qi_cond')

        final_data['alpha_n_phi'] = np.mean(acc['cross_n_phi'], axis=0)
        final_data['alpha_Ti_phi'] = np.mean(acc['cross_Ti_phi'], axis=0)
        final_data['alpha_Tperp_phi'] = np.mean(acc['cross_Tperp_phi'], axis=0)
        final_data['alpha_Tpar_phi'] = np.mean(acc['cross_Tpar_phi'], axis=0)
        
        return final_data

    except Exception as e:
        print(f"Error processing spectra: {e}")
        return None
    finally:
        os.chdir(original_dir)

def plot_spectra_dashboard(spectra_dict, plot_mode='Amplitudes'):
    """Renders the spectra plots for the Marimo dashboard."""
    colors = ['red', 'blue', 'black', 'green', 'purple']
    
    if plot_mode == 'Fluxes & Phases':
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 2)
        
        def p(r, c, key, ylabel, is_alpha=True):
            ax = fig.add_subplot(gs[r, c])
            for i, (label, d) in enumerate(spectra_dict.items()):
                ky = np.fft.fftshift(d['ky']) * d['rho_s']
                y = np.fft.fftshift(d[key])
                mask = ky > 0
                
                ky_plot = ky[mask]
                y_plot = y[mask]
                
                c_idx = colors[i % len(colors)]
                
                if is_alpha:
                    # ---> FIX: Extract the actual phase angle <---
                    y_plot = np.angle(y_plot)
                    # Plot phases as dots to avoid ugly wrap-around lines
                    ax.plot(ky_plot, y_plot, label=label, color=c_idx, 
                            linestyle='none', marker='o', markersize=3, alpha=0.8)
                else:
                    y_plot /= 1e3 # Convert to kW/m^2
                    ax.plot(ky_plot, y_plot, label=label, color=c_idx, lw=1.5)
            
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            if is_alpha:
                ax.set_ylim(-np.pi, np.pi)
                ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
                ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
            if r == 2: ax.set_xlabel(r'$k_y \rho_s$')
            else: ax.set_xticklabels([])
            ax.grid(True, which='both', linestyle='-', alpha=0.3)
            return ax

        ax_leg = p(0, 0, 'Qi_tot', r'$k_y \hat{Q}_i / (kW m^{-2})$', False)
        ax_leg.legend(loc='upper left', fontsize=10)
        p(0, 1, 'Qe_tot', r'$k_y \hat{Q}_e / (kW m^{-2})$', False)
        p(1, 0, 'alpha_n_phi', r'$\alpha(\hat{n}, \hat{\phi})$')
        p(1, 1, 'alpha_Ti_phi', r'$\alpha(\hat{T}_i, \hat{\phi})$')
        p(2, 0, 'alpha_Tperp_phi', r'$\alpha(\hat{T}_{\perp,e}, \hat{\phi})$')
        p(2, 1, 'alpha_Tpar_phi', r'$\alpha(\hat{T}_{\parallel,e}, \hat{\phi})$')

    else:
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 2)
        
        def plot_panel(r, c, key, ylabel):
            ax = fig.add_subplot(gs[r, c])
            for i, (label, d) in enumerate(spectra_dict.items()):
                ky = np.fft.fftshift(d['ky']) * d['rho_s']
                y = np.fft.fftshift(d[key])
                mask = ky > 0
                y = y[mask]
                if 'n_amp' in key: y /= 1e17
                ax.plot(ky[mask], y, label=label, color=colors[i % len(colors)], lw=1.5)
            
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.set_ylim(bottom=0)
            if r == 2: ax.set_xlabel(r'$k_y \rho_s$')
            else: ax.set_xticklabels([])
            ax.grid(True, which='both', linestyle='-', alpha=0.3)
            return ax

        ax_leg = plot_panel(0, 0, 'n_amp', r'$\hat{n} / (10^{17} m^{-3})$')
        ax_leg.legend(fontsize=10)
        plot_panel(0, 1, 'Ti_amp', r'$\hat{T}_i$ / eV')
        plot_panel(1, 0, 'Te_amp', r'$\hat{T}_e$ / eV')
        plot_panel(1, 1, 'phi_amp', r'$\hat{\phi}$ / V')
        plot_panel(2, 0, 'Tperp_e_amp', r'$\hat{T}_{\perp,e}$ / eV')
        plot_panel(2, 1, 'Tpar_e_amp', r'$\hat{T}_{\parallel,e}$ / eV')

    plt.tight_layout()
    return fig

def plot_frequency_spectra(data_dict, field_key='ne', r_target_val=0.0, lcfs_shift=0.0):
    """Plots Frequency Cross-Power, Cross-Phase, and Coherence."""
    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True, layout='constrained')
    
    for label, d in data_dict.items():
        x = d['x'] - lcfs_shift
        res = d['results']
        
        if 'freqs' not in res: continue
        
        # Find closest radial index
        x_idx = np.argmin(np.abs(x - r_target_val))
        actual_r_shift = x[x_idx]
        
        freqs = res['freqs']
        mask = freqs > 0 # Only plot positive frequencies
        f_plot = freqs[mask]
        f_plot /= 1e3
        
        if field_key == 'ne':
            P_f = res['P_ne'][mask, x_idx]
            P_phi = res['P_phi'][mask, x_idx]
            C_f_phi = res['C_ne_phi'][mask, x_idx]
            lbl_f = r'\delta n'  # Removed the $ signs here
        else:
            P_f = res['P_Te'][mask, x_idx]
            P_phi = res['P_phi'][mask, x_idx]
            C_f_phi = res['C_Te_phi'][mask, x_idx]
            lbl_f = r'\delta T_e' # Removed the $ signs here
            
        c = d.get('color', None)
        ls = d.get('ls', '-')
        
        # 1. Cross-Power (Magnitude of Cross Spectrum)
        cross_power = np.abs(C_f_phi)
        axs[0].plot(f_plot, cross_power, color=c, linestyle=ls, label=f"{label} ({actual_r_shift:.2f}m)")
        
        # 2. Cross-Phase (Angle of Cross Spectrum)
        phase = np.angle(C_f_phi)
        axs[1].plot(f_plot, phase, color=c, linestyle=ls)
        
        # 3. Coherence
        coherence = cross_power / np.sqrt(P_f * P_phi + 1e-16)
        axs[2].plot(f_plot, coherence, color=c, linestyle=ls)
        
    axs[0].set_ylabel(r'Cross-Power $|P_{' + lbl_f + r', \phi}|$')
    axs[0].set_yscale('log')
    axs[0].legend(fontsize=10)
    axs[0].set_title('Frequency Spectra (Cross-Phase & Coherence)')
    
    axs[1].set_ylabel(f'Phase Angle (rad)')
    axs[1].set_ylim(-np.pi, np.pi)
    axs[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axs[1].set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    
    axs[2].set_ylabel(f'Coherence $\gamma$')
    axs[2].set_ylim(0, 1.05)
    axs[2].set_xlabel('Freq (kHz)')
    
    for ax in axs: ax.grid(True, alpha=0.3)
    
    return fig