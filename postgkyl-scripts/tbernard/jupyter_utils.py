import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import postgkyl as pg
from scipy.stats import skew, kurtosis

# Attempt to import local utils
try:
    import utils
except ImportError:
    sys.path.append(os.getcwd())
    try:
        import utils
    except ImportError:
        print("WARNING: 'utils.py' not found. Data loading functions may fail.")

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
    'VEshearAve':  {'lbl': r'$\langle \gamma_E \rangle$', 'unit': '$s^{-1}$'},
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
    'Lc_ave':      {'lbl': r'$L_c$', 'unit': 'm'},
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

def process_simulation_run(sim_dir, fstart, fend, step=1, z_idx=1, amu=2.014):
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
        Lc_ave = np.mean(Lc[len(Lc)//2:])
        
        jacgeo_data = pg.GData(f"{file_prefix}-jacobgeo.gkyl")
        bmag_data = pg.GData(f"{file_prefix}-bmag.gkyl")

        # --- Storage Init ---
        data_store = {
            'ne': [], 'ni': [], 'Te': [], 'Ti': [], 'phi': [],
            'VEx': [], 'VEy': [], 'VEshear': [], 'Er': [], 
            'p': [], 'Qpara': []
        }
        x_vals = None

        # --- Time Loop ---
        for tf in range(fstart, fend + 1, step):
            try:
                # Load Objects
                elc_data = pg.data.GData(f"{file_prefix}-elc_BiMaxwellianMoments_{tf}.gkyl")
                ion_data = pg.data.GData(f"{file_prefix}-ion_BiMaxwellianMoments_{tf}.gkyl")
                phi_data = pg.data.GData(f"{file_prefix}-field_{tf}.gkyl")
                q_files = [f"{file_prefix}-{s}_M3{d}_{tf}.gkyl" for s in ['elc','ion'] for d in ['par','perp']]
                q_moms = [pg.data.GData(f) for f in q_files]

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

                # Flows & Shear
                VE_x, VE_y, _, VE_shear, Er = utils.func_calc_VE(phi_data, b_i_data, jacgeo_data, bmag_data, z_idx)

                # Heat Flux
                _, qpara_elc = utils.func_data_yave(q_moms[0], 0, -1) 
                _, qpara_ion = utils.func_data_yave(q_moms[2], 0, -1)
                Q_total_1d = (ME/2 * qpara_elc) + (MI_LOCAL/2 * qpara_ion)

                # Store
                data_store['ne'].append(ne_2d)
                data_store['ni'].append(ni_2d)
                data_store['Te'].append(Te_2d)
                data_store['Ti'].append(Ti_2d)
                data_store['phi'].append(phi_2d)
                data_store['VEx'].append(VE_x)
                data_store['VEy'].append(VE_y)
                data_store['VEshear'].append(VE_shear)
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

        # Fluxes
        Gamma_x = np.mean(flucs['ne'] * flucs['VEx'], axis=(0, 2))
        Q_x_e = means['ne'] * np.mean(flucs['Te'] * flucs['VEx'], axis=(0, 2)) + means['Te'] * Gamma_x
        Q_x_i = means['ni'] * np.mean(flucs['Ti'] * flucs['VEx'], axis=(0, 2)) + means['Ti'] * Gamma_x
        Rey_stress = np.mean(flucs['VEx'] * flucs['VEy'], axis=(0, 2))
        Rey_force = -np.gradient(Rey_stress, x_vals)

        # Higher Order Stats
        dn_flat = flucs['ne'].transpose(0, 2, 1).reshape(-1, Nx)
        l_rad, _ = calc_radial_correlation(dn_flat, x_vals)
        skewness = skew(dn_flat, axis=0)
        kurt_val = kurtosis(dn_flat, axis=0)

        # Pack Results
        results = {
            'neAve': means['ne'], 'TeAve': means['Te'], 'TiAve': means['Ti'], 
            'phiAve': means['phi'], 'QparaAve': means['Qpara'],
            'VEyAve': means['VEy'], 'VEshearAve': means['VEshear'], 'ErAve': means['Er'],
            'Gamma_x': Gamma_x, 'Qxe': Q_x_e, 'Qxi': Q_x_i,
            'Rey_stress': Rey_stress, 'Rey_force': Rey_force,
            'dn_rms': rms['ne'], 'dn_norm': norm_rms['dn_norm'],
            'dT_rms': rms['Te'], 'dT_norm': norm_rms['dT_norm'],
            'dphi_rms': rms['phi'], 'dphi_norm': norm_rms['dphi_norm'],
            'skew': skewness, 'kurt': kurt_val, 'l_rad': l_rad,
            'Lc_ave': Lc_ave
        }
        
        return x_vals, results

    except Exception as e:
        print(f"Error processing {sim_dir}: {e}")
        return None, None
    finally:
        os.chdir(original_dir)

def load_datasets(sim_dict, fstart, fend, step=1, amu=2.014):
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
        x, res = process_simulation_run(path, fstart, fend, step, amu=amu)
        
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

def plot_1d_comparison(data_dict, field_keys, units=None, lcfs_shift=0.0, x_trim=0):
    """
    Plots 1D profiles.
    If 'units' list is None, looks up default units from FIELD_INFO.
    """
    n_fields = len(field_keys)
    cols = min(3, n_fields)
    rows = (n_fields + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharex=True)
    if n_fields == 1: axs = [axs]
    else: axs = axs.flatten()
    
    for i, field in enumerate(field_keys):
        ax = axs[i]
        
        # --- LOOKUP METADATA ---
        if units is None and field in FIELD_INFO:
            unit_str = FIELD_INFO[field]['unit']
            title_str = FIELD_INFO[field]['lbl']
        elif units is not None and i < len(units):
            unit_str = units[i]
            title_str = field
        else:
            unit_str = ""
            title_str = field
        # -----------------------

        for label, d in data_dict.items():
            x = d['x'] - lcfs_shift
            y = d['results'].get(field, None)
            x = x[x_trim:-x_trim] if x_trim > 0 else x
            y = y[x_trim:-x_trim] if x_trim > 0 else y
            
            c = d.get('color', None)
            ls = d.get('ls', '-')

            if y is not None:
                if np.isscalar(y) or y.ndim==0:
                    ax.axhline(y, label=f"{label} (Global)", color=c, linestyle='--')
                else:
                    ax.plot(x, y, label=label, color=c, linestyle=ls)
        
        ax.set_title(title_str)
        ax.set_ylabel(unit_str)
        if i == 0: ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        
        if i >= (rows - 1) * cols:
            ax.set_xlabel(r"$R - R_{sep}$ (m)")

    for i in range(n_fields, len(axs)):
        fig.delaxes(axs[i])
        
    plt.tight_layout()
    plt.show()

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
        'Te': {'tot': r'$T_e$', 'fluc': r'$\delta T_e / T_e$'},
        'Te': {'tot': r'$T_i$', 'fluc': r'$\delta T_i / T_i$'},
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