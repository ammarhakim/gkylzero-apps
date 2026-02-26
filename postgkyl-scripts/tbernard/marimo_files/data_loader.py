"""
data_loader.py
Handles reading Gkeyll data from disk, extracting fields, and calculating core profiles/stats.
"""
import os
import sys
import glob
import numpy as np
import postgkyl as pg
from scipy.stats import skew, kurtosis

# Import our new local modules
import config
import math_utils

# Safely import the external 'utils.py' provided by the Gkeyll team
_script_dir = os.path.dirname(os.path.realpath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
try:
    import utils
except ImportError as e:
    print(f"CRITICAL WARNING: 'utils.py' could not be imported from {_script_dir}.")
    print(f"Reason: {e}")


def get_max_frames_and_time(sim_def):
    """Scans the directory for the highest frame number, extracts max physical time."""
    directory = sim_def.get('dir', '.')
    files = glob.glob(os.path.join(directory, '*_[0-9]*.bp')) 
    
    if not files:
        return 0, 0.0
        
    max_frame = 0
    max_file = ""
    for f in files:
        try:
            frame_str = f.split('_')[-1].replace('.bp', '')
            frame_num = int(frame_str)
            if frame_num >= max_frame:
                max_frame = frame_num
                max_file = f
        except ValueError:
            continue
            
    max_time = 0.0
    try:
        if max_file:
            data = pg.GData(max_file)
            max_time = data.ctx.get('time', 0.0) 
    except Exception as e:
        print(f"Warning: Could not read time from {max_file}: {e}")
        
    return max_frame, max_time


def load_integrated_moms(sim_path, species='elc'):
    """Loads the time-series of integrated moments."""
    original_dir = os.getcwd()
    try:
        os.chdir(sim_path)
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        fname = f"{file_prefix}-{species}_integrated_moms.gkyl"
        if not os.path.exists(fname):
            return None, None
        data = pg.GData(fname)
        return data.get_grid()[0], data.get_values()
    except Exception as e:
        print(f"Error loading integrated moms: {e}")
        return None, None
    finally:
        os.chdir(original_dir)


def load_2d_snapshot(sim_path, frame, field_name, z_idx=1, amu=2.014):
    """Loads a single 2D snapshot for plotting."""
    MI_LOCAL = config.MP * amu 
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
            mass = config.ME if 'elc' in suffix else MI_LOCAL
            val_iso = (val_par + 2 * val_perp) / 3.0 * mass / config.EV
            return x, y, val_iso
            
    except Exception as e:
        print(f"Error loading snapshot: {e}")
        return None, None, None
    finally:
        os.chdir(original_dir)


def process_spectra_for_sim(sim_dir, fstart, fend, x_idx, step=1, amu=2.014):
    """Processes the k_y spectra for a single simulation at a specific radial index."""
    MI_LOCAL = config.MP * amu
    original_dir = os.getcwd()
    try:
        if not os.path.exists(sim_dir): return None
        os.chdir(sim_dir)
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        
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

        gdata_init = pg.GData(f"{file_prefix}-field_{fstart}.gkyl")
        y_vals = gdata_init.get_grid()[1]
        Ly = abs(y_vals[-1] - y_vals[0])
        Ny = len(y_vals) - 1 if len(y_vals) > 1 else 1 
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
            Te_par = get_comp(elc_dat, 2) / config.EV * config.ME
            Te_perp = get_comp(elc_dat, 3) / config.EV * config.ME
            Te = (Te_par + 2*Te_perp)/3.0
            
            ni = get_comp(ion_dat, 0)
            Ti_par = get_comp(ion_dat, 2) / config.EV * MI_LOCAL
            Ti_perp = get_comp(ion_dat, 3) / config.EV * MI_LOCAL
            Ti = (Ti_par + 2*Ti_perp)/3.0

            acc['ne_bg'].append(math_utils.fsa_mean(ne, J_z))
            acc['Te_bg'].append(math_utils.fsa_mean(Te, J_z))
            acc['ni_bg'].append(math_utils.fsa_mean(ni, J_z))
            acc['Ti_bg'].append(math_utils.fsa_mean(Ti, J_z))

            for k, v in zip(['phi', 'n', 'Te', 'Ti', 'Tpar_e', 'Tperp_e'], [phi, ne, Te, Ti, Te_par, Te_perp]):
                acc[k].append(math_utils.compute_fsa_spectra(v, J_z))

            inv_B = 1.0 / (B_z + 1e-16) if B_z is not None else None
            acc['Qe_conv'].append(math_utils.compute_fsa_cross_spectra(ne, phi, J_z, inv_B))
            acc['Qe_cond'].append(math_utils.compute_fsa_cross_spectra(Te, phi, J_z, inv_B))
            acc['Qi_conv'].append(math_utils.compute_fsa_cross_spectra(ni, phi, J_z, inv_B))
            acc['Qi_cond'].append(math_utils.compute_fsa_cross_spectra(Ti, phi, J_z, inv_B))

            acc['cross_n_phi'].append(math_utils.compute_fsa_cross_spectra(ne, phi, J_z))
            acc['cross_Ti_phi'].append(math_utils.compute_fsa_cross_spectra(Ti, phi, J_z))
            acc['cross_Tperp_phi'].append(math_utils.compute_fsa_cross_spectra(Te_perp, phi, J_z))
            acc['cross_Tpar_phi'].append(math_utils.compute_fsa_cross_spectra(Te_par, phi, J_z))

        if not acc['phi']: return None

        Te_bg_avg = np.mean(acc['Te_bg'])
        cs = np.sqrt(Te_bg_avg * config.EV / MI_LOCAL)
        omega_ci = (config.EV * B0) / MI_LOCAL if B0 != 0 else 1.0
        rho_s = cs / omega_ci

        final_data = {'ky': ky_raw, 'rho_s': rho_s}
        for k in ['phi', 'n', 'Te', 'Ti', 'Tpar_e', 'Tperp_e']:
            final_data[f'{k}_amp'] = np.sqrt(np.mean(acc[k], axis=0)) * norm

        factor_J = 1.5 * config.EV * norm**2
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


def process_simulation_run(sim_dir, fstart, fend, step=1, z_idx=1, amu=2.014, B_axis=2.0):
    """Processes a simulation directory to extract profiles, fluxes, and stats."""
    MI_LOCAL = config.MP * amu 
    original_dir = os.getcwd()
    try:
        if not os.path.exists(sim_dir):
            print(f"Error: Directory {sim_dir} does not exist.")
            return None, None

        os.chdir(sim_dir)
        print(f"--> Analyzing: {sim_dir} | Frames: {fstart}-{fend} | AMU: {amu}")
        
        file_prefix = utils.find_prefix('-field_0.gkyl', '.')
        
        b_i_data = pg.GData(f"{file_prefix}-b_i.gkyl")
        _, _, z_vals, b_z = utils.func_data_3d(f"{file_prefix}-b_i.gkyl", 2)
        mid_y = b_z.shape[1] // 2
        Lc = np.sum(b_z[:, mid_y, :], axis=1) * np.diff(z_vals)[0]
        Lc_ave = np.mean(Lc[len(Lc)*2//3:])
        
        jacgeo_data = pg.GData(f"{file_prefix}-jacobgeo.gkyl")
        bmag_data = pg.GData(f"{file_prefix}-bmag.gkyl")

        try:
            gij_data = pg.GData(f"{file_prefix}-gij.gkyl")
            _, _, local_shear_xz = utils.func_calc_local_shear(gij_data)
        except Exception:
            local_shear_xz = None

        try:
            _, _, f_trap_xz = utils.func_calc_trapped_fraction(bmag_data)
            z_mid_idx = f_trap_xz.shape[1] // 2
            f_trap_mid = f_trap_xz[:, z_mid_idx]
        except Exception:
            f_trap_xz = None
            f_trap_mid = None     

        data_store = {
            'ne': [], 'ni': [], 'Te': [], 'Ti': [], 'phi': [], 'phi_3d' : [],
            'VEx': [], 'VEy': [], 'VEshear': [], 'Er': [], 
            'p': [], 'Qpara': []
        }
        x_vals = None
        time_array = []

        for tf in range(fstart, fend + 1, step):
            try:
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

                _, _, ne_2d = utils.func_data_2d(elc_data, 0, z_idx)
                _, _, ni_2d = utils.func_data_2d(ion_data, 0, z_idx)
                _, _, phi_2d = utils.func_data_2d(phi_data, 0, z_idx)

                _, _, Te_par = utils.func_data_2d(elc_data, 2, z_idx)
                _, _, Te_perp = utils.func_data_2d(elc_data, 3, z_idx)
                Te_2d = (Te_par + 2*Te_perp)/3.0 * config.ME / config.EV

                _, _, Ti_par = utils.func_data_2d(ion_data, 2, z_idx)
                _, _, Ti_perp = utils.func_data_2d(ion_data, 3, z_idx)
                Ti_2d = (Ti_par + 2*Ti_perp)/3.0 * MI_LOCAL / config.EV

                _, _, _, phi_3d_vals = utils.func_data_3d(f"{file_prefix}-field_{tf}.gkyl", 0)
                VE_x, VE_y, _, VE_shear, Er = utils.func_calc_VE(phi_data, b_i_data, jacgeo_data, bmag_data, z_idx)

                _, qpara_elc = utils.func_data_yave(q_moms[0], 0, -1) 
                _, qpara_ion = utils.func_data_yave(q_moms[2], 0, -1)
                Q_total_1d = (config.ME/2 * qpara_elc) + (MI_LOCAL/2 * qpara_ion)

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

        if not data_store['ne']:
            return None, None

        arrs = {k: np.array(v) for k, v in data_store.items() if len(v) > 0}
        Nt, Nx, Ny = arrs['ne'].shape

        means = {}
        for k, v in arrs.items():
            if v.ndim == 3: means[k] = np.mean(v, axis=(0, 2))
            elif v.ndim == 2: means[k] = np.mean(v, axis=0)

        flucs = {}
        for k in ['ne', 'Te', 'Ti', 'phi', 'VEx', 'VEy']:
            mean_profile = means[k][np.newaxis, :, np.newaxis]
            flucs[k] = arrs[k] - mean_profile

        rms = {k: np.std(arrs[k], axis=(0, 2)) for k in ['ne', 'Te', 'Ti', 'phi']}
        norm_rms = {
            'dn_norm': rms['ne'] / means['ne'],
            'dT_norm': rms['Te'] / means['Te'],
            'dphi_norm': rms['phi'] / means['Te']
        }

        Nt_fluc = flucs['ne'].shape[0]
        dt = np.mean(np.diff(time_array)) if len(time_array) > 1 else 1.0
        window = np.hanning(Nt_fluc).reshape(Nt_fluc, 1, 1)
        
        F_ne = np.fft.fft(flucs['ne'] * window, axis=0)
        F_Te = np.fft.fft(flucs['Te'] * window, axis=0)
        F_phi = np.fft.fft(flucs['phi'] * window, axis=0)
        
        P_ne = np.mean(np.abs(F_ne)**2, axis=2)
        P_Te = np.mean(np.abs(F_Te)**2, axis=2)
        P_phi = np.mean(np.abs(F_phi)**2, axis=2)
        C_ne_phi = np.mean(F_ne * np.conj(F_phi), axis=2)
        C_Te_phi = np.mean(F_Te * np.conj(F_phi), axis=2)
        freqs = np.fft.fftfreq(Nt_fluc, d=dt)

        phi_mode_structure = np.std(arrs['phi_3d'], axis=(0, 2))

        Gamma_x = np.mean(flucs['ne'] * flucs['VEx'], axis=(0, 2))
        Q_x_e = means['ne'] * np.mean(flucs['Te'] * flucs['VEx'], axis=(0, 2)) + means['Te'] * Gamma_x
        Q_x_i = means['ni'] * np.mean(flucs['Ti'] * flucs['VEx'], axis=(0, 2)) + means['Ti'] * Gamma_x
        Rey_stress = np.mean(flucs['VEx'] * flucs['VEy'], axis=(0, 2))
        Rey_force = -np.gradient(Rey_stress, x_vals)
        v_eff = Gamma_x / means['ne']

        dn_flat = flucs['ne'].transpose(0, 2, 1).reshape(-1, Nx)
        dT_flat = flucs['Ti'].transpose(0, 2, 1).reshape(-1, Nx)
        dphi_flat = flucs['phi'].transpose(0, 2, 1).reshape(-1, Nx)
        
        l_rad, _ = math_utils.calc_radial_correlation(dn_flat, x_vals)
        skewness = skew(dn_flat, axis=0)
        kurt_val = kurtosis(dn_flat, axis=0)

        v_r = means.get('VEx', np.mean(data_store.get('VEx', np.zeros(Nx)), axis=(0,2)))
        I_flux = np.mean(flucs.get('VEx', np.zeros_like(flucs['ne'])) * flucs['ne']**2, axis=(0, 2))

        ne_mean = means['ne']
        dx = x_vals[1] - x_vals[0]
        dne_dx = np.gradient(ne_mean, dx)
        L_n = np.zeros_like(ne_mean)
        valid = np.abs(dne_dx) > 1e-10
        L_n[valid] = - ne_mean[valid] / dne_dx[valid]
        
        c_s = np.sqrt(means['Te'] * 1.602e-19 / (2.014 * 1.672e-27))
        gamma_mhd = c_s * np.sqrt(np.maximum(0.0, 2.0 / (np.abs(x_vals) * np.abs(L_n) + 1e-8)))

        Te_mean = means['Te']
        Ti_mean = means['Ti']
        dTe_dx = np.gradient(Te_mean, dx)
        dTi_dx = np.gradient(Ti_mean, dx)
        valid_n = ne_mean > 1e-12
        v_star_e = np.zeros_like(ne_mean)
        v_star_i = np.zeros_like(ne_mean)
        v_star_e[valid_n] = -(dTe_dx[valid_n] + (Te_mean[valid_n] / ne_mean[valid_n]) * dne_dx[valid_n]) / B_axis
        v_star_i[valid_n] =  (dTi_dx[valid_n] + (Ti_mean[valid_n] / ne_mean[valid_n]) * dne_dx[valid_n]) / B_axis

        tau_para = Lc_ave / c_s
        tau_perp = L_n / v_eff
        tau_ratio = np.clip(tau_para / tau_perp, a_min=0, a_max=100)
        
        t_elc, int_elc = load_integrated_moms(sim_dir, 'elc')
        t_ion, int_ion = load_integrated_moms(sim_dir, 'ion')

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
            'Lc_ave': Lc_ave, 'z_vals': z_vals, 'local_shear_xz': local_shear_xz,
            'phi_mode_structure': phi_mode_structure,
            'f_trap_xz': f_trap_xz, 'f_trap_mid': f_trap_mid,
            'I_flux': I_flux, 'gamma_mhd': gamma_mhd, 'v_r': v_r,
            'v_star_e': v_star_e, 'v_star_i': v_star_i, 'tau_ratio' : tau_ratio,
            'freqs': freqs, 'P_ne': P_ne, 'P_Te': P_Te, 'P_phi': P_phi,
            'C_ne_phi': C_ne_phi, 'C_Te_phi': C_Te_phi,
            'time_series_t': t_elc,
            'int_n_elc': int_elc[:, 0] if int_elc is not None else None,
            'int_en_elc': int_elc[:, 2] if int_elc is not None else None,
            'int_n_ion': int_ion[:, 0] if int_ion is not None else None,
            'int_en_ion': int_ion[:, 2] if int_ion is not None else None,
            'dn_flat': dn_flat, 'dT_flat' : dT_flat, 'dphi_flat' : dphi_flat,
        }
        return x_vals, results

    except Exception as e:
        print(f"Error processing {sim_dir}: {e}")
        return None, None
    finally:
        os.chdir(original_dir)


def load_datasets(sim_dict, fstart, fend, step=1, amu=2.014, B_axis=2.0):
    """Batch processes all simulations in the dictionary."""
    processed_data = {}
    print(f"Batch processing {len(sim_dict)} simulations...")

    for label, meta in sim_dict.items():
        if isinstance(meta, str):
            path = meta
            color = None
            ls = '-'
        else:
            path = meta.get('path')
            color = meta.get('color', None)
            ls = meta.get('ls', '-')

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