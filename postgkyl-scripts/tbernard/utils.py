import os
import re
import fnmatch
import numpy as np
import postgkyl as pg
import matplotlib as mpl
import h5py

# Physical constants in SI units
elem_charge = 1.602176634e-19  # Elementary charge [C]
mass_proton = 1.67262192369e-27  # Proton mass [kg]
mass_elc = 9.1093837015e-31     # Electron mass [kg]

def find_prefix(pattern, path):
    for name in os.listdir(path):
        if fnmatch.fnmatch(name, '*' + pattern):
            return re.sub(pattern, '', name)
    raise FileNotFoundError("ERROR: file prefix not found!")

def get_center_coords(interp_grid):
    return [(grid[1:] + grid[:-1]) / 2 for grid in interp_grid]

def interpolate_field(field3d, comp):
    interp = pg.data.GInterpModal(field3d, 1, 'ms')
    return interp.interpolate(comp)

def get_slice_index(z_vals, z_idx):
    return 0 if z_idx == 0 else len(z_vals) // 2

def func_data_yave(field3d, comp, z_idx):
    grid, values = interpolate_field(field3d, comp)
    CCC = get_center_coords(grid)
    z_slice = get_slice_index(CCC[2], z_idx)
    return CCC[0], np.mean(values[:, :, z_slice, 0], axis=1)

def func_data_2d(field3d, comp, z_idx):
    grid, values = interpolate_field(field3d, comp)
    CCC = get_center_coords(grid)
    z_slice = get_slice_index(CCC[2], z_idx)
    return CCC[0], CCC[1], values[:, :, z_slice, 0]

def func_calc_VE(phi3d, bmag2d, z_idx):
    grid, values = interpolate_field(phi3d, 0)
    CCC = get_center_coords(grid)
    x_vals, y_vals, z_vals = CCC
    z_slice = get_slice_index(z_vals, z_idx)
    phi = values[:, :, z_slice, 0]
    dx, dy = x_vals[1] - x_vals[0], y_vals[1] - y_vals[0]
    Ex = -np.gradient(phi, dx, axis=0)
    Ey = -np.gradient(phi, dy, axis=1)
    VE_x = Ey / bmag2d
    VE_y = -np.mean(Ex, axis=1) / np.mean(bmag2d, axis=1)
    VE_shear = np.gradient(VE_y, dx, axis=0)
    return VE_x, VE_y, VE_shear, np.mean(Ex, axis=1)

def func_time_ave(data_list):
    return np.mean(np.array(data_list), axis=0)

def func_calc_norm_fluc(data2d, dataAve, dataNorm, Nt, Ny, Nx):
    data2dTot = np.reshape(data2d, (Nt * Ny, Nx))
    dataAve2d = np.array([dataAve] * (Nt * Ny))
    delt = data2dTot - dataAve2d
    sigma = np.sqrt(np.mean(delt ** 2, axis=0))
    delt_norm = sigma / dataNorm
    return delt, delt_norm

def save_to_hdf5(filename, x_vals, diagnostics, metadata):
    with h5py.File(filename, "w") as f:
        f.create_dataset("x_vals", data=x_vals)
        for name, data in diagnostics.items():
            f.create_dataset(name, data=data)
        for key, value in metadata.items():
            f.attrs[key] = value