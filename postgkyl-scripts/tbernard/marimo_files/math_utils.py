"""
math_utils.py
Pure mathematical functions for spectral analysis, correlation, and fluctuations.
Including "fsa" flux-surface-averaged quantities.
"""
import numpy as np

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