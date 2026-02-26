"""
plotting.py
Handles all Matplotlib visualizations for the Gkeyll dashboard.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import postgkyl as pg

# Import our local modules
import config
import math_utils
import data_loader

# Safely import the external 'utils.py' provided by the Gkeyll team
_script_dir = os.path.dirname(os.path.realpath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
try:
    import utils
except ImportError:
    pass


def plot_saturation(data_dict, species='elc'):
    """Plots Total Particles and Total Energy vs Time to check for saturation."""
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
            ax1.plot(t * 1e6, n_int / n_int[0], label=label, color=c, linestyle=ls)
            
        if en_int is not None:
            ax2.plot(t * 1e6, en_int / en_int[0], label=label, color=c, linestyle=ls)

    ax1.set_ylabel("Rel. Particle Count")
    ax1.set_title(f"Saturation Check ({species})")
    ax1.legend()
    
    ax2.set_ylabel("Rel. Total Energy")
    ax2.set_xlabel("Time ($\mu s$)")
    
    plt.tight_layout()
    return fig


def plot_1d_profiles(data_dict, fields_to_plot, lcfs_shift=0.0, r_lcfs=2.17, r_axis=1.65, trim_pts=0):
    """Plots 1D radial profiles mapped to normalized rho in a multi-column grid."""
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
        
        info = config.FIELD_INFO.get(field, {'lbl': field, 'unit': ''})
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
        ax.set_xlabel(r'$\rho$') 
        ax.grid(True, alpha=0.3)
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.6, label='LCFS')
        
        if i == 0: ax.legend()

    for j in range(num_fields, len(axs)):
        axs[j].set_visible(False)

    return fig


def plot_2d_comparison(sim_data, frame, field_name, mode='total', lcfs_shift=0.0, vlims=None, amu=2.014):
    """Plots side-by-side 2D heatmaps."""
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
        path = item.get('path') if isinstance(item, dict) else item
            
        x, y, data = data_loader.load_2d_snapshot(path, frame, field_name, amu=amu)
        
        if x is None: 
            ax.text(0.5, 0.5, "Data Not Found", ha='center')
            continue
        
        if mode == 'fluctuation':
            fluc, norm_fluc = math_utils.get_2d_fluctuations(data, subtract_mean='y')
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
    return fig


def plot_qpara_sol(data_dict, lcfs_shift=0.0, x_trim=0, fit_rmin=0.0, fit_rmax=0.05):
    """Plots Qpara in the SOL region and fits the decay length."""
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    qpara_info = config.FIELD_INFO.get('QparaAve', {'lbl': 'Qpara', 'unit': ''})

    for label, d in data_dict.items():
        x_vals = d['x']
        results = d['results']
        
        qpara_data = results.get('QparaAve', None)
        if qpara_data is None: continue
        
        x_plot = x_vals - lcfs_shift
        if x_trim > 0:
            x_plot = x_plot[x_trim:-x_trim]
            qpara_data = qpara_data[x_trim:-x_trim]

        sol_mask = x_plot > 0.0
        x_sol = x_plot[sol_mask]
        qpara_sol = qpara_data[sol_mask]

        if len(x_sol) == 0: continue

        c = d.get('color', None)
        ls = d.get('ls', '-')
        plot_label = label

        fit_mask = (x_sol >= fit_rmin) & (x_sol <= fit_rmax)
        valid_q = qpara_sol > 0
        combined_mask = fit_mask & valid_q
        
        if np.sum(combined_mask) > 3:
            x_fit = x_sol[combined_mask]
            q_fit_data = qpara_sol[combined_mask]
            
            p = np.polyfit(x_fit, np.log(q_fit_data), 1)
            if p[0] < 0:
                lambda_q_m = -1.0 / p[0]
                lambda_q_mm = lambda_q_m * 1000.0
                q_fit_line = np.exp(np.polyval(p, x_fit))
                ax.plot(x_fit, q_fit_line, color=c, linestyle=':', lw=3.5, alpha=1.0)
                plot_label = f"{label} ($\lambda_q$={lambda_q_mm:.1f} mm)"
            else:
                plot_label = f"{label} (No decay found)"

        ax.plot(x_sol, qpara_sol, label=plot_label, color=c, linestyle=ls, lw=2)

    ax.set_title(f"{qpara_info['lbl']} in SOL with Decay Fits")
    ax.set_xlabel(r"$R - R_{LCFS}$ (m)")
    ax.set_ylabel(f"{qpara_info['lbl']} ({qpara_info['unit']})")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    return fig


def plot_pdf_slice(data_dict, field_key='dn_flat', x_target_idx=None, r_target_val=None,
                   bins='auto', density=True, compare_gaussian=False,
                   x_trim=0, lcfs_shift=0.0):
    """Plots the PDF of fluctuations at a specific radial location."""
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    for label, d in data_dict.items():
        x_vals = d['x']
        results = d['results']
        
        fluctuation_data_flat = results.get(field_key, None)
        if fluctuation_data_flat is None: continue
        
        if x_target_idx is None and r_target_val is not None:
            x_vals_shifted = x_vals - lcfs_shift
            current_x_idx = np.argmin(np.abs(x_vals_shifted - r_target_val))
        elif x_target_idx is not None:
            current_x_idx = x_target_idx
        else:
            continue

        if current_x_idx >= fluctuation_data_flat.shape[1] or current_x_idx < 0: continue

        data_at_x = fluctuation_data_flat[:, current_x_idx]
        mean_data_at_x = np.mean(data_at_x)
        std_data_at_x = np.std(data_at_x)
        
        if std_data_at_x < 1e-10: continue

        normalized_data = (data_at_x - mean_data_at_x) / std_data_at_x
        hist, bin_edges = np.histogram(normalized_data, bins=bins, density=density)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        deltR = r"$R - R_{LCFS}$"
        ax.plot(bin_centers, hist, label=f"{label}, {deltR}={x_vals[current_x_idx]-lcfs_shift:.3f}m",
                color=d.get('color', None), linestyle=d.get('ls', '-'))

    if compare_gaussian:
        xmin, xmax = ax.get_xlim()
        x_gaussian = np.linspace(xmin, xmax, 500)
        ax.plot(x_gaussian, norm.pdf(x_gaussian), 'k--', label='Standard Normal PDF')

    ax.set_title(f"PDF of Normalized Fluctuations ({config.FIELD_INFO.get(field_key, {}).get('lbl', field_key)})")
    ax.set_xlabel(r"$(\delta f - \langle \delta f \rangle) / \sigma_{\delta f}$")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return fig


def plot_distf_slice(sim_path, frame, species, ix, iy, z_idx=0):
    """Plots the cell-average distribution function in velocity space."""
    original_dir = os.getcwd()
    try:
        os.chdir(sim_path)
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        
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

        Xnodal = [np.outer(xInt[3], np.ones(np.shape(xInt[4]))), 
                  np.outer(np.ones(np.shape(xInt[3])), xInt[4])]

        if z_idx != 0:
            z_idx = distf.shape[2] // 2

        f_slice = np.squeeze(distf[ix, iy, z_idx, :, :])
            
        fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
        im = ax.pcolormesh(Xnodal[0], Xnodal[1], f_slice, cmap='plasma', shading='auto')
        ax.set_title(f"{species} $f(v)$ at x={xInt[0][ix]}, y={xInt[1][iy]}")
        ax.set_xlabel(r"$v_{\parallel}$")
        ax.set_ylabel(r"$\mu$")
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
        if 'z_vals' not in res or 'phi_mode_structure' not in res: continue
            
        z = res['z_vals']
        phi_struct = res['phi_mode_structure']
        shear = res.get('local_shear_xz', None)
        
        x_idx = np.argmin(np.abs(x - r_target_val))
        actual_r = x[x_idx]
        
        c = d.get('color', None)
        ls = d.get('ls', '-')
        
        ax1.plot(z, phi_struct[x_idx, :], color=c, linestyle=ls, label=f"{label} (R={actual_r:.2f})")
        if shear is not None:
            ax2.plot(z, shear[x_idx, :], color=c, linestyle='--', alpha=0.6)

    ax1.set_xlabel(r"Parallel Coordinate $z$ (m)")
    ax1.set_ylabel(r"Mode Amplitude $\delta \phi_{rms}$", color='black')
    ax2.set_ylabel(r"Local Magnetic Shear $s_{loc}$", color='gray')
    ax1.set_title("Parallel ITG Mode Structure vs. Local Shear")
    
    import matplotlib.lines as mlines
    mode_line = mlines.Line2D([], [], color='black', linestyle='-', label=r'$|\phi|$ Amplitude')
    shear_line = mlines.Line2D([], [], color='gray', linestyle='--', label=r'$s_{loc}$')
    lines1, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=lines1 + [mode_line, shear_line], loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    return fig


def plot_spectra_dashboard(spectra_dict, plot_mode='Amplitudes'):
    """Renders the spectra plots for the Marimo dashboard."""
    colors = ['red', 'blue', 'black', 'green', 'purple']
    
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2)
    
    if plot_mode == 'Fluxes & Phases':
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
                    y_plot = np.angle(y_plot)
                    ax.plot(ky_plot, y_plot, label=label, color=c_idx, 
                            linestyle='none', marker='o', markersize=3, alpha=0.8)
                else:
                    y_plot /= 1e3 
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
        
        x_idx = np.argmin(np.abs(x - r_target_val))
        actual_r_shift = x[x_idx]
        
        freqs = res['freqs']
        mask = freqs > 0 
        f_plot = freqs[mask]
        f_plot /= 1e3
        
        if field_key == 'ne':
            P_f = res['P_ne'][mask, x_idx]
            P_phi = res['P_phi'][mask, x_idx]
            C_f_phi = res['C_ne_phi'][mask, x_idx]
            lbl_f = r'\delta n' 
        else:
            P_f = res['P_Te'][mask, x_idx]
            P_phi = res['P_phi'][mask, x_idx]
            C_f_phi = res['C_Te_phi'][mask, x_idx]
            lbl_f = r'\delta T_e'
            
        c = d.get('color', None)
        ls = d.get('ls', '-')
        
        cross_power = np.abs(C_f_phi)
        axs[0].plot(f_plot, cross_power, color=c, linestyle=ls, label=f"{label} ({actual_r_shift:.2f}m)")
        
        phase = np.angle(C_f_phi)
        axs[1].plot(f_plot, phase, color=c, linestyle=ls)
        
        coherence = cross_power / np.sqrt(P_f * P_phi + 1e-16)
        axs[2].plot(f_plot, coherence, color=c, linestyle=ls)
        
    axs[0].set_ylabel(r'Cross-Power $|P_{' + lbl_f + r', \phi}|$')
    axs[0].set_yscale('log')
    axs[0].legend(fontsize=10)
    axs[0].set_title('Frequency Spectra (Cross-Phase & Coherence)')
    
    axs[1].set_ylabel('Phase Angle (rad)')
    axs[1].set_ylim(-np.pi, np.pi)
    axs[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axs[1].set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    
    axs[2].set_ylabel(r'Coherence $\gamma$')
    axs[2].set_ylim(0, 1.05)
    axs[2].set_xlabel('Freq (kHz)')
    
    for ax in axs: ax.grid(True, alpha=0.3)
    
    return fig


def list_fields():
    """Prints a formatted list of all available fields."""
    print(f"{'Key':<15} | {'Unit':<15} | {'Description'}")
    print("-" * 50)
    for key in sorted(config.FIELD_INFO.keys()):
        meta = config.FIELD_INFO[key]
        clean_lbl = meta['lbl'].replace('$', '')
        clean_unit = meta['unit'].replace('$', '')
        print(f"{key:<15} | {clean_unit:<15} | {clean_lbl}")


def scan_for_negativity(sim_path, frame, field_name='ne', z_idx=1, amu=2.014):
    """Finds the minimum value of a field and its location."""
    x, y, data = data_loader.load_2d_snapshot(sim_path, frame, field_name, z_idx, amu)
    if data is None:
        return None, None, None, "Data load failed."

    min_val = np.nanmin(data)
    flat_idx = np.nanargmin(data)
    ix, iy = np.unravel_index(flat_idx, data.shape)
    r_loc = x[ix]
    y_loc = y[iy]
    msg = f"Min {field_name}: {min_val:.4e} at R={r_loc:.4f}, Y={y_loc:.4f} (Idx: {ix}, {iy})"
    
    return min_val, (r_loc, y_loc), (ix, iy), msg