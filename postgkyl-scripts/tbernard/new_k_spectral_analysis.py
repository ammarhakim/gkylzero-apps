import os
import argparse
import numpy as np
import postgkyl as pg
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Physical constants
ELEM_CHARGE = 1.60217662e-19
MASS_ELC = 9.10938356e-31
MASS_ION = 1.6726219e-27 * 2.014  # Deuterium
EV_TO_J = 1.60217662e-19

# Try importing utils
try:
    import utils
except ImportError:
    utils = None

# --- Plotting Style ---
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 1.5
})

# ==========================================
# Data Extraction Functions
# ==========================================

def get_flux_surface_data(gdata, x_idx):
    """Extracts (y, z) surface data at radial index x_idx."""
    dg = pg.GInterpModal(gdata, poly_order=1, basis_type='ms')
    raw_coeffs = dg._getRawModal(0) 
    cell_avg = raw_coeffs[..., 0] / (2**1.5)
    return cell_avg[x_idx, :, :]

def get_geometry_data(file_prefix, x_idx):
    """Loads Jacobian and B_field."""
    try:
        jac_data = pg.GData(f"{file_prefix}-jacobgeo.gkyl")
        J_slice = get_flux_surface_data(jac_data, x_idx)
        J_z = np.mean(J_slice, axis=0)

        bmag_data = pg.GData(f"{file_prefix}-bmag.gkyl")
        B_slice = get_flux_surface_data(bmag_data, x_idx)
        B_z = np.mean(B_slice, axis=0)
        
        return J_z, B_z
    except Exception as e:
        print(f"Geometry warning: {e}. Using flat metric.")
        return None, None

def compute_fsa_spectra(data_yz, J_z):
    """Computes Flux-Surface Averaged Power Spectrum."""
    mean_y = np.mean(data_yz, axis=0, keepdims=True)
    fluc = data_yz - mean_y
    fft_k = np.fft.fft(fluc, axis=0)
    power_kz = np.abs(fft_k)**2
    
    if J_z is not None:
        numerator = np.sum(power_kz * J_z[np.newaxis, :], axis=1)
        return numerator / np.sum(J_z)
    else:
        return np.mean(power_kz, axis=1)

def compute_fsa_cross_spectra(data1_yz, data2_yz, J_z, weighting=None):
    """Computes FSA Cross Spectrum < f1 * f2* >_FSA."""
    f1 = data1_yz - np.mean(data1_yz, axis=0, keepdims=True)
    f2 = data2_yz - np.mean(data2_yz, axis=0, keepdims=True)
    F1 = np.fft.fft(f1, axis=0)
    F2 = np.fft.fft(f2, axis=0)
    cross = np.conj(F1) * F2
    
    if weighting is not None:
        cross = cross * weighting[np.newaxis, :]
        
    if J_z is not None:
        return np.sum(cross * J_z[np.newaxis, :], axis=1) / np.sum(J_z)
    else:
        return np.mean(cross, axis=1)

def fsa_mean(data_yz, J_z):
    """Calculates Flux Surface Average of the background profile."""
    # Average over Y first (zonal mean), then weight by Jz
    zonal = np.mean(data_yz, axis=0)
    if J_z is not None:
        return np.sum(zonal * J_z) / np.sum(J_z)
    return np.mean(zonal)

# ==========================================
# Main Logic
# ==========================================

def analyze_simulation(args):
    if utils is None:
        print("Error: 'utils.py' is required."), exit()

    try:
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        print(f"Using file prefix: {file_prefix}")
    except FileNotFoundError as e:
        print(e), exit()

    # --- 1. Grid & Geometry ---
    gdata_init = pg.GData(f"{file_prefix}-field_{args.fstart}.gkyl")
    x_vals = gdata_init.get_grid()[0]
    x_vals = (x_vals[1:] + x_vals[:-1]) / 2
    x_idx = args.xidx if args.xidx is not None else len(x_vals) // 2
    
    # Geometry for Normalization
    R_loc = args.R_LCFS - args.x_inner + x_vals[x_idx]
    rho_val = (R_loc - args.R_axis) / (args.R_LCFS - args.R_axis)
    J_z, B_z = get_geometry_data(file_prefix, x_idx)
    B0 = np.mean(B_z) if B_z is not None else 1.0

    print(f"Analyzing x_idx={x_idx}, rho={rho_val:.3f}")

    # --- 2. Initialize Accumulators ---
    acc = {
        'phi': [], 'n': [], 'Te': [], 'Ti': [], 
        'Tpar_e': [], 'Tperp_e': [], 
        # Flux Components
        'Qe_conv': [], 'Qe_cond': [],
        'Qi_conv': [], 'Qi_cond': [],
        # Cross Phases
        'cross_n_phi': [], 'cross_Ti_phi': [], 
        'cross_Tperp_phi': [], 'cross_Tpar_phi': [],
        # Backgrounds (to average later)
        'Te_bg': [], 'ne_bg': [], 'ni_bg': [], 'Ti_bg': []
    }

    # --- 3. Time Loop ---
    for tf in range(args.fstart, args.fend + 1):
        print(f"Processing frame {tf}...", end='\r')
        try:
            phi_dat = pg.GData(f"{file_prefix}-field_{tf}.gkyl")
            elc_dat = pg.GData(f"{file_prefix}-elc_BiMaxwellianMoments_{tf}.gkyl")
            ion_dat = pg.GData(f"{file_prefix}-ion_BiMaxwellianMoments_{tf}.gkyl")
        except: continue

        # Extract Slices
        phi = get_flux_surface_data(phi_dat, x_idx)
        
        # Helper to get specific moment component
        def get_comp(gdat, c):
            dg = pg.GInterpModal(gdat, 1, 'ms')
            return dg._getRawModal(c)[x_idx, :, :, 0] / (2**1.5)

        ne = get_comp(elc_dat, 0)
        Te_par = get_comp(elc_dat, 2) / EV_TO_J * MASS_ELC
        Te_perp = get_comp(elc_dat, 3) / EV_TO_J * MASS_ELC
        Te = (Te_par + 2*Te_perp)/3.0
        
        ni = get_comp(ion_dat, 0)
        Ti_par = get_comp(ion_dat, 2) / EV_TO_J * MASS_ION
        Ti_perp = get_comp(ion_dat, 3) / EV_TO_J * MASS_ION
        Ti = (Ti_par + 2*Ti_perp)/3.0

        # --- Calculate Backgrounds (FSA) ---
        ne0 = fsa_mean(ne, J_z)
        Te0 = fsa_mean(Te, J_z)
        ni0 = fsa_mean(ni, J_z)
        Ti0 = fsa_mean(Ti, J_z)
        
        acc['ne_bg'].append(ne0); acc['Te_bg'].append(Te0)
        acc['ni_bg'].append(ni0); acc['Ti_bg'].append(Ti0)

        # --- Spectra ---
        for k, v in zip(['phi', 'n', 'Te', 'Ti', 'Tpar_e', 'Tperp_e'], [phi, ne, Te, Ti, Te_par, Te_perp]):
            acc[k].append(compute_fsa_spectra(v, J_z))

        # --- Flux Decomposition ---
        # Formula: Q_conv = 1.5 * T0 * ky * Im(n * phi*) / B
        #          Q_cond = 1.5 * n0 * ky * Im(T * phi*) / B
        
        inv_B = 1.0 / (B_z + 1e-16) if B_z is not None else None
        
        # Electron Components
        # Note: We compute the cross-spectrum term < A * phi* / B > here
        acc['Qe_conv'].append(compute_fsa_cross_spectra(ne, phi, J_z, weighting=inv_B))
        acc['Qe_cond'].append(compute_fsa_cross_spectra(Te, phi, J_z, weighting=inv_B))
        
        # Ion Components
        acc['Qi_conv'].append(compute_fsa_cross_spectra(ni, phi, J_z, weighting=inv_B))
        acc['Qi_cond'].append(compute_fsa_cross_spectra(Ti, phi, J_z, weighting=inv_B))

        # --- Cross Phases ---
        acc['cross_n_phi'].append(compute_fsa_cross_spectra(ne, phi, J_z))
        acc['cross_Ti_phi'].append(compute_fsa_cross_spectra(Ti, phi, J_z))
        acc['cross_Tperp_phi'].append(compute_fsa_cross_spectra(Te_perp, phi, J_z))
        acc['cross_Tpar_phi'].append(compute_fsa_cross_spectra(Te_par, phi, J_z))

    # --- 4. Final Processing ---
    print("\nAveraging time series...")
    
    # Time-averaged Backgrounds
    Te_bg_avg = np.mean(acc['Te_bg'])
    ne_bg_avg = np.mean(acc['ne_bg'])
    Ti_bg_avg = np.mean(acc['Ti_bg'])
    ni_bg_avg = np.mean(acc['ni_bg'])

    # Calculate rho_s
    cs = np.sqrt(Te_bg_avg * EV_TO_J / MASS_ION)
    omega_ci = (ELEM_CHARGE * B0) / MASS_ION
    rho_s = cs / omega_ci
    print(f"  Avg Te={Te_bg_avg:.1f} eV, rho_s={rho_s*1000:.2f} mm")

    y_vals = gdata_init.get_grid()[1]
    Ly = abs(y_vals[-1] - y_vals[0])
    y_vals = (y_vals[1:] + y_vals[:-1]) / 2
    Ny = len(y_vals)
    ky_raw = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    norm = 1.0 / Ny 

    final_data = {}
    
    # Amplitudes
    for k in ['phi', 'n', 'Te', 'Ti', 'Tpar_e', 'Tperp_e']:
        final_data[f'{k}_amp'] = np.sqrt(np.mean(acc[k], axis=0)) * norm

    # Fluxes Calculation
    # Q(k) = 1.5 * Coeff * ky * Im(Cross)
    # Unit conversion: 
    #   density (m^-3), Temp (eV), phi (V), B (T)
    #   Need Result in Watts/m^2 -> Then / 1e6 for MW
    #   Q ~ [1.5 * n * T * k * phi / B] * e (for eV->J)
    
    factor_J = 1.5 * EV_TO_J * norm**2
    
    def calc_flux(bg, cross_list):
        # Mean complex cross-spectrum
        cross_avg = np.mean(acc[cross_list], axis=0)
        # ky * Im(Cross)
        return bg * ky_raw * np.imag(cross_avg) * factor_J

    final_data['Qe_conv'] = calc_flux(Te_bg_avg, 'Qe_conv')
    final_data['Qe_cond'] = calc_flux(ne_bg_avg, 'Qe_cond')
    final_data['Qe_tot']  = final_data['Qe_conv'] + final_data['Qe_cond']
    
    final_data['Qi_conv'] = calc_flux(Ti_bg_avg, 'Qi_conv')
    final_data['Qi_cond'] = calc_flux(ni_bg_avg, 'Qi_cond')
    final_data['Qi_tot']  = final_data['Qi_conv'] + final_data['Qi_cond']

    # Phases
    final_data['alpha_n_phi'] = np.angle(np.mean(acc['cross_n_phi'], axis=0))
    final_data['alpha_Ti_phi'] = np.angle(np.mean(acc['cross_Ti_phi'], axis=0))
    final_data['alpha_Tperp_phi'] = np.angle(np.mean(acc['cross_Tperp_phi'], axis=0))
    final_data['alpha_Tpar_phi'] = np.angle(np.mean(acc['cross_Tpar_phi'], axis=0))

    # Save
    with h5py.File(args.output, 'w') as f:
        f.attrs['label'] = args.label
        f.attrs['rho_s'] = rho_s
        f.attrs['rho_val'] = rho_val
        f.create_dataset('ky', data=ky_raw)
        for key, val in final_data.items():
            f.create_dataset(key, data=val)
    print(f"Saved to {args.output}")

# ==========================================
# Plotting
# ==========================================

def plot_comparison(file_list, output_file, override_labels=None, plot_fluxes=False):
    print("Loading data for comparison...")
    data_store = []
    
    for i, fname in enumerate(file_list):
        with h5py.File(fname, 'r') as f:
            lbl = override_labels[i] if override_labels and i < len(override_labels) else f.attrs['label']
            d = {
                'label': lbl,
                'rho_s': f.attrs['rho_s'],
                'rho_val': f.attrs.get('rho_val', 0.0),
                'ky': f['ky'][:],
            }
            for k in f.keys():
                if k != 'ky': d[k] = f[k][:]
            data_store.append(d)

    colors = ['red', 'blue', 'green', 'purple']
    
    # --- FLUX PLOT ---
    if plot_fluxes:
        print("Plotting fluxes...")
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(3, 2)
        # Top Left: Electron Conductive vs Convective
        # Bottom Left: Ion Conductive vs Convective
        # Right column could be totals
        
        def p(r, c, key, ylabel, is_alpha=True):
            ax = fig.add_subplot(gs[r, c])
            for i, d in enumerate(data_store):
                ky = np.fft.fftshift(d['ky']) * d['rho_s']
                y = np.fft.fftshift(d[key])
                mask = ky > 0
                y = y[mask]
                if not is_alpha:    
                    y /= 1e3  # Convert to kW/m^2 for plotting
                lbl = f"{d['label']} ($\\rho$={d['rho_val']:.2f})"
                ax.plot(ky[mask], y, label=lbl, color=colors[i], lw=1.5)
            
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            if is_alpha:
                ax.set_ylim(bottom=-np.pi, top=np.pi)
                # normalize y-axis to -pi to pi with pi as labels
                ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
                ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
            if r == 2: ax.set_xlabel(r'$k_y \rho_s$')
            else: ax.set_xticklabels([])
            ax.grid(True, which='both', linestyle='-', alpha=0.3)
            return ax

        ax_leg = p(0, 0, 'Qi_tot', r'$k_y \hat{Q}_i / (kW m^{-2})$',is_alpha=False)
        ax_leg.legend(loc='upper left', ncol=1, frameon=True)
        p(0, 1, 'Qe_tot', r'$k_y \hat{Q}_e / (kW m^{-2})$',is_alpha=False)
        p(1, 0, 'alpha_n_phi', r'$\alpha(\hat{n}, \hat{\phi})$')
        p(1, 1, 'alpha_Ti_phi', r'$\alpha(\hat{T}_i, \hat{\phi})$')
        p(2, 0, 'alpha_Tperp_phi', r'$\alpha(\hat{T}_{\perp,e}, \hat{\phi})$')
        p(2, 1, 'alpha_Tpar_phi', r'$\alpha(\hat{T}_{\parallel,e}, \hat{\phi})$')

        plt.tight_layout()
        print(f"Saving plot to comparison_fluxes.png...")
        plt.savefig("comparison_fluxes.png", dpi=300)
        
    else:
        # --- STANDARD PLOT ---
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(3, 2)
        
        def plot_panel(r, c, key, ylabel):
            ax = fig.add_subplot(gs[r, c])
            for i, d in enumerate(data_store):
                ky = np.fft.fftshift(d['ky']) * d['rho_s']
                y = np.fft.fftshift(d[key])
                mask = ky > 0
                y = y[mask]
                if 'n_amp' in key: y /= 1e17

                lbl = f"{d['label']} ($\\rho$={d['rho_val']:.2f})"
                ax.plot(ky[mask], y, label=lbl, color=colors[i], lw=1.5)
            
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.set_ylim(bottom=0)
            if r == 2: ax.set_xlabel(r'$k_y \rho_s$')
            else: ax.set_xticklabels([])
            ax.grid(True, which='both', linestyle='-', alpha=0.3)
            return ax

        ax_leg = plot_panel(0, 0, 'n_amp', r'$\hat{n} / (10^{17} m^{-3})$')
        ax_leg.legend(ncol=1, frameon=True)
        plot_panel(0, 1, 'Ti_amp', r'$\hat{T}_i$ / eV')
        plot_panel(1, 0, 'Te_amp', r'$\hat{T}_e$ / eV')
        plot_panel(1, 1, 'phi_amp', r'$\hat{\phi}$ / V')
        plot_panel(2, 0, 'Tperp_e_amp', r'$\hat{T}_{\perp,e}$ / eV')
        plot_panel(2, 1, 'Tpar_e_amp', r'$\hat{T}_{\parallel,e}$ / eV')

        plt.tight_layout()
        print(f"Saving plot to {output_file}...")
        plt.savefig(output_file, dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot_fluxes', action='store_true')
    parser.add_argument('--fstart', type=int, default=0)
    parser.add_argument('--fend', type=int, default=10)
    parser.add_argument('--xidx', type=int)
    parser.add_argument('--zstr', type=str, default='zmid')
    parser.add_argument('--label', type=str, default='Sim')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--files', nargs='+')
    parser.add_argument('--labels', nargs='+')
    
    # Geometry defaults
    parser.add_argument('--R_axis', type=float, default=2.3695)
    parser.add_argument('--R_LCFS', type=float, default=2.8209)
    parser.add_argument('--x_inner', type=float, default=0.10)

    args = parser.parse_args()

    # --- INPUT PROMPT LOGIC ---
    if args.analyze and '--analyze' in sys.argv:
        try:
            print("\n--- Geometry Setup ---")
            device = input("Enter device (e.g., TCV, DIII-D, WEST) [Hit Enter for default]: ").strip().upper()
            if device:
                tri = input("PT / NT? ").strip().upper()
                if device == 'TCV':
                    args.x_inner = 0.04
                    if tri == 'PT':
                        args.R_axis = 0.8727315068; args.R_LCFS = 1.0968432365
                    else: 
                        args.R_axis = 0.8867856264; args.R_LCFS = 1.0870056099
                elif device == 'DIII-D' or device == 'D3D':
                    args.x_inner = 0.10
                    args.R_axis = 1.6486; args.R_LCFS = 2.17
                
                print(f"  -> Configured for {device} ({tri})")
        except KeyboardInterrupt:
            print("\nCancelled."), exit()

    if args.analyze:
        if args.output is None: args.output = 'spectra.h5'
        analyze_simulation(args)
    elif args.plot or args.plot_fluxes:
        if args.output is None: args.output = 'comparison.png'
        if not args.files: print("Provide --files"), exit()
        plot_comparison(args.files, args.output, args.labels, args.plot_fluxes)
    else:
        parser.print_help()