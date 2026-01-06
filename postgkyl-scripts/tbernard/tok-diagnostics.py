import os
import re
import fnmatch
import numpy as np
import postgkyl as pg
import matplotlib as mpl
import h5py
import utils

# convert x_vals to rho, parsing information from the input file
def main():
    mp = 1.672623e-27
    AMU = 2.014
    mi = mp * AMU
    me = 9.10938188e-31
    eV = 1.602e-19

    file_prefix = utils.find_prefix('-field_0.gkyl', '.')
    print(f"Using file prefix: {file_prefix}")

    fstart = int(input("fstart? "))
    fend = int(input("fend? "))
    step = int(input("step? "))
    z_idx = 1 # when 0 use first index, else use mid-plane
    z_str = 'zmid'

    bmag_data = pg.GData(f"{file_prefix}-bmag.gkyl")
    _, y_vals, bmag_2d = utils.func_data_2d(bmag_data, 0, z_idx)
    b_i_data = pg.GData(f"{file_prefix}-b_i.gkyl")
    _, _, z_vals, b_z = utils.func_data_3d(f"{file_prefix}-b_i.gkyl", 2)
    
    b_z = b_z[:,len(y_vals)//2,:]
    Lc = np.sum(b_z,axis=1)*np.diff(z_vals)[0]
    Lc_ave = np.mean(Lc[80:])
    print(f"Lc_ave = {Lc_ave}")
    jacgeo_data = pg.GData(f"{file_prefix}-jacobgeo.gkyl")

    diag_names = ['phi', 'elcDens', 'ionDens', 'elcTemp', 'ionTemp', 'VEy', 'VEshear', 'p', 'Er', 'Qpara']
    diagnostics = {name: [] for name in diag_names}
    elcDens2dTot, elcTemp2dTot, ionTemp2dTot, phi2dTot = [], [], [], []
    VEx2dTot, VEy2dTot = [], []

    for tf in range(fstart, fend + 1, step):
        elc_data = pg.data.GData(f"{file_prefix}-elc_BiMaxwellianMoments_{tf}.gkyl")
        ion_data = pg.data.GData(f"{file_prefix}-ion_BiMaxwellianMoments_{tf}.gkyl")
        phi_data = pg.data.GData(f"{file_prefix}-field_{tf}.gkyl")
        elc_qpara_data = pg.data.GData(f"{file_prefix}-elc_M3par_{tf}.gkyl")
        ion_qpara_data = pg.data.GData(f"{file_prefix}-ion_M3par_{tf}.gkyl")
        elc_qperp_data = pg.data.GData(f"{file_prefix}-elc_M3perp_{tf}.gkyl")
        ion_qperp_data = pg.data.GData(f"{file_prefix}-ion_M3perp_{tf}.gkyl")

        x_vals, elc_dens = utils.func_data_yave(elc_data, 0, z_idx)
        x_vals, ion_dens = utils.func_data_yave(ion_data, 0, z_idx)

        x_vals, elc_Tpar = utils.func_data_yave(elc_data, 2, z_idx)
        x_vals, elc_Tperp = utils.func_data_yave(elc_data, 3, z_idx)
        elc_temp = (elc_Tpar + 2 * elc_Tperp) / 3 * me / eV

        x_vals, ion_Tpar = utils.func_data_yave(ion_data, 2, z_idx)
        x_vals, ion_Tperp = utils.func_data_yave(ion_data, 3, z_idx)
        ion_temp = (ion_Tpar + 2 * ion_Tperp) / 3 * mi / eV

        x_vals, phi_vals = utils.func_data_yave(phi_data, 0, z_idx)
        VE_x, VE_y, VE_y_1d, VE_shear, Er = utils.func_calc_VE(phi_data, b_i_data, jacgeo_data, bmag_data, z_idx)

        _ , qpara_ion = utils.func_data_yave(ion_qpara_data, 0, -1)
        _, qpara_elc = utils.func_data_yave(elc_qpara_data, 0, -1)
        _ , qperp_ion = utils.func_data_yave(ion_qperp_data, 0, -1)
        _, qperp_elc = utils.func_data_yave(elc_qperp_data, 0, -1)

        Qion = mi/2 * (qpara_ion) # + qperp_ion)
        Qelc = me/2 * (qpara_elc) # + qperp_elc)

        diagnostics['elcDens'].append(elc_dens)
        diagnostics['ionDens'].append(ion_dens)
        diagnostics['elcTemp'].append(elc_temp)
        diagnostics['ionTemp'].append(ion_temp)
        diagnostics['phi'].append(phi_vals)
        diagnostics['VEy'].append(VE_y_1d)
        diagnostics['VEshear'].append(VE_shear)
        diagnostics['Er'].append(Er)
        diagnostics['p'].append((elc_temp + ion_temp) * elc_dens)
        diagnostics['Qpara'].append(Qelc + Qion)

        # Transpose the following data to make it easier to do turbulence statistics
        VEx2dTot.append(VE_x.T)
        VEy2dTot.append(VE_y.T)

        _, _, elc_dens2d = utils.func_data_2d(elc_data, 0, z_idx)
        elcDens2dTot.append(elc_dens2d.T)

        _, _, elc_temp2dpar = utils.func_data_2d(elc_data, 2, z_idx)
        _, _, elc_temp2dperp = utils.func_data_2d(elc_data, 2, z_idx)
        elcTemp2dTot.append((elc_temp2dpar.T + 2*elc_temp2dperp) * me / eV)

        _, _, ion_temp2dpar = utils.func_data_2d(ion_data, 2, z_idx)
        _, _, ion_temp2dperp = utils.func_data_2d(ion_data, 2, z_idx)
        ionTemp2dTot.append((ion_temp2dpar.T + 2*ion_temp2dperp) * mi / eV)

        _, _, phi2d = utils.func_data_2d(phi_data, 0, z_idx)
        phi2dTot.append(phi2d.T)

    Nt, Nx, Ny = len(elcDens2dTot), len(x_vals), len(y_vals)
    elcDensAve = utils.func_time_ave(diagnostics['elcDens'])
    elcTempAve = utils.func_time_ave(diagnostics['elcTemp'])
    ionDensAve = utils.func_time_ave(diagnostics['ionDens'])
    ionTempAve = utils.func_time_ave(diagnostics['ionTemp'])
    phiAve = utils.func_time_ave(diagnostics['phi'])
    qparaAve = utils.func_time_ave(diagnostics['Qpara'])

    elcDens2dTot = np.array(elcDens2dTot)
    elcTemp2dTot = np.array(elcTemp2dTot)
    ionTemp2dTot = np.array(ionTemp2dTot)
    phi2dTot = np.array(phi2dTot)
    VEx2dTot = np.array(VEx2dTot)
    VEy2dTot = np.array(VEy2dTot)

    nAve2d = np.mean(elcDens2dTot, axis=(0, 1), keepdims=True)
    TeAve2d = np.mean(elcTemp2dTot, axis=(0, 1), keepdims=True)
    TiAve2d = np.mean(ionTemp2dTot, axis=(0, 1), keepdims=True)
    VExAve2d = np.mean(VEx2dTot, axis=(0, 1), keepdims=True)
    VEyAve2d = np.mean(VEy2dTot, axis=(0, 1), keepdims=True)

    dVEx = VEx2dTot - VExAve2d
    dVEy = VEy2dTot - VEyAve2d
    dnGr = elcDens2dTot - nAve2d
    dTeQr = elcTemp2dTot - TeAve2d
    dTiQr = ionTemp2dTot - TiAve2d

    Gx = np.mean(dnGr * dVEx, axis=(0, 1))
    Qxe = elcDensAve * np.mean(dTeQr * dVEx, axis=(0, 1)) + Gx * elcTempAve
    Qxi = ionDensAve * np.mean(dTiQr * dVEx, axis=(0, 1)) + Gx * ionTempAve
    reynolds_stress = np.mean(dVEx * dVEy, axis=(0, 1))
    reynolds_force = -np.gradient(reynolds_stress, x_vals)

    n_flat = elcDens2dTot.transpose(0, 2, 1).reshape(-1, Nx)
    T_flat = elcTemp2dTot.transpose(0, 2, 1).reshape(-1, Nx)
    phi_flat = phi2dTot.transpose(0, 2, 1).reshape(-1, Nx)

    # Time-Averaged Profiles (Mean)
    n_mean = np.mean(n_flat, axis=0)
    T_mean = np.mean(T_flat, axis=0)
    phi_mean = np.mean(phi_flat, axis=0)

    # RMS Fluctuations (Std Dev). To normalize, uncomment "/ Mean "
    # dn or dn/n
    dn_rms = np.std(n_flat, axis=0) #/ np.abs(n_mean)
    # dT or dT/T
    dT_rms = np.std(T_flat, axis=0) #/ np.abs(T_mean)
    # dPhi or e*dPhi/Te (Normalized by Te)
    dphi_rms = np.std(phi_flat, axis=0) #/ np.abs(T_mean)

    dn_norm = dn_rms / elcDensAve
    dT_norm = dT_rms / elcTempAve
    dphi_norm = dphi_rms / elcTempAve

    dn, _ = utils.func_calc_norm_fluc(elcDens2dTot, elcDensAve, elcDensAve, Nt, Ny, Nx)
    dT, _ = utils.func_calc_norm_fluc(elcTemp2dTot, elcTempAve, elcTempAve, Nt, Ny, Nx)
    dphi, _ = utils.func_calc_norm_fluc(phi2dTot, phiAve, elcTempAve, Nt, Ny, Nx)

    dnSq = dn * dn
    skew, kurt = [], []
    for xs in range(Nx):
        dn_loc = dn[:, xs].flatten()
        sigma_loc = np.mean(np.sqrt(dnSq[:, xs]))
        pdf_input = dn_loc / sigma_loc
        hist, edges = np.histogram(pdf_input, bins='auto', density=True)
        dbin = np.diff(edges)
        bin_center = edges[:-1] + dbin / 2
        skew_val = np.sum(sigma_loc**3 * bin_center**3 * hist * dbin) / sigma_loc**3
        kurt_val = np.sum(sigma_loc**4 * bin_center**4 * hist * dbin) / sigma_loc**4 - 3
        skew.append(skew_val)
        kurt.append(kurt_val)

    xi_min = Nx // 6
    xi_max = -xi_min
    dx = x_vals[1] - x_vals[0]
    dxArray = np.arange(xi_min) * dx
    dn_cc = dn[:, xi_min:]
    sigma_sq_new = np.mean(dnSq[:, xi_min:xi_max], axis=0)
    cc_dx = [
        np.mean(dn_cc[:, :xi_max] * dn_cc[:, dxi:(xi_max + dxi)], axis=0) / sigma_sq_new
        for dxi in range(xi_min)
    ]
    xVals_new = x_vals[xi_min:xi_max]
    cc_dx = np.array(cc_dx)
    l_rad, l_rad_err = [], []
    for xi in range(len(xVals_new)):
        y = np.clip(cc_dx[:, xi], a_min=1e-3, a_max=None)
        coeffs = np.polyfit(dxArray, np.log(y), 1, w=np.sqrt(y))
        l_rad.append(-1.0 / coeffs[0])
        l_rad_err.append(coeffs[1])

    averaged = {name + "Ave": utils.func_time_ave(data) for name, data in diagnostics.items()}
    all_results = {**averaged, "Gx": Gx, "Qxe": Qxe, "Qxi": Qxi, 
                   "reynolds_stress": reynolds_stress, "reynolds_force": reynolds_force,
                   "dn_norm": dn_norm, "dT_norm": dT_norm, "dphi_norm": dphi_norm,
                   "dn": dn_rms, "dT": dT_rms, "dphi": dphi_rms,
                   "skew": skew, "kurt": kurt, "l_rad": l_rad, "l_rad_err": l_rad_err,
                   "xVals_new": xVals_new}

    metadata = {"fstart": fstart, "fend": fend, "zStr": z_str}
    utils.save_to_hdf5(f"diagnostics_{fstart}to{fend}_{z_str}.h5", x_vals, all_results, metadata)

if __name__ == "__main__":
    main()
