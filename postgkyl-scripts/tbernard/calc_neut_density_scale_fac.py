import os
import re
import fnmatch
import numpy as np
import postgkyl as pg
import matplotlib as mpl
import h5py
import utils

# Simulation parameters and physical constants
mp = 1.67262192369e-27  # Proton mass [kg]
me = 9.1093837015e-31  # Electron mass [kg]
eV = 1.602176634e-19  # Elementary charge [C]
neut = 'D0' # Neutral species name, e.g., 'D0' for deuterium
rec_frac = 0.99  # Fraction of neutrals that are ionized in the reaction

def main():
    
    file_prefix = utils.find_prefix('-jacobgeo.gkyl', '.')
    fn_ion_bflux_xu = f"{file_prefix}-ion_bflux_xupper_integrated_HamiltonianMoments.gkyl"
    fn_ion_bflux_zu = f"{file_prefix}-ion_bflux_zupper_integrated_HamiltonianMoments.gkyl"
    fn_ion_bflux_zl = f"{file_prefix}-ion_bflux_zlower_integrated_HamiltonianMoments.gkyl"

    tvals, ion_bflux_xu = utils.integrated_moms(fn_ion_bflux_xu)
    _, ion_bflux_zu = utils.integrated_moms(fn_ion_bflux_zu)
    _, ion_bflux_zl = utils.integrated_moms(fn_ion_bflux_zl)

    # Find index where tvals is closest to t
    frame = input("Frame number? ")
    # Calculate time from frame number. Frames output every 1e-5
    t = float(frame)*1e-6
    print(f"t = {t}")
    idx = np.argmin(np.abs(tvals - t))
    print(tvals)
    print(f"Index of closest t is {idx} for t = {tvals[idx]}")

    # Total flux at t
    ion_bflux_total = ion_bflux_xu[idx] + ion_bflux_zu[idx] + ion_bflux_zl[idx]
    print(f"Total ion flux at t = {tvals[idx]} is {ion_bflux_total}")

    # Calculate integral of ionization source term over entire volume
    # S_e,iz = n_neut * react_rate
    # Interpolate 3d data
    x, y, z, m0_neut = utils.func_data_3d(f"{file_prefix}-{neut}_{frame}.gkyl", 0) # neutrals are static
    _, _, _, m0_elc = utils.func_data_3d(f"{file_prefix}-elc_BiMaxwellianMoments_{frame}.gkyl", 0)
    _, _, _, react_rate = utils.func_data_3d(f"{file_prefix}-ion_elc_react_iz_{neut}_{frame}.gkyl", 0)
    _, _, _, jac = utils.func_data_3d(f"{file_prefix}-jacobgeo.gkyl", 0)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dV = dx * dy * dz  # Volume element
    m0_neut = m0_neut/(mp*2.014)

    # Multiply by Jacobian to get volume element
    source_iz = jac * m0_neut * m0_elc * react_rate
    source_iz_int = np.sum(source_iz) * dV  # Integrate over the entire volume

    print(f"Integral of ionization source term over entire volume: {source_iz_int}")

    alpha_frac = rec_frac * ion_bflux_total / source_iz_int
    print(f"Alpha fraction: {alpha_frac}")

if __name__ == "__main__":
    main()