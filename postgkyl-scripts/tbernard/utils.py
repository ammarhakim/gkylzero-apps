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

def integrated_moms(filename, comp=0):
    """
    Load integrated Hamiltonian moments from a file.
    """
    gdata = pg.data.GData(filename)
    grid = gdata.get_grid()
    values = gdata.get_values()
    return np.array(grid[0]), np.array(values[:,comp]) if values.ndim > 1 else values

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

def func_data_3d(filename, comp=0):
    gdata = pg.data.GData(filename)
    grid, values = interpolate_field(gdata, comp)
    CCC = get_center_coords(grid)
    return CCC[0], CCC[1], CCC[2], values[:, :, :, 0]

def func_phase_data(filename, comp=0):
    gdata = pg.data.GData(filename)
    grid, values = interpolate_field(gdata, comp)
    CCC = get_center_coords(grid)
    return CCC[0], CCC[1], CCC[2], CCC[3], CCC[4], values[:, :, :, :, :, 0]

def func_calc_VE(phi_data, b_i_data, jac_data, bmag_data, z_idx):
    grid, phi = interpolate_field(phi_data, 0)
    _, b_x = interpolate_field(b_i_data, 0)
    _, b_y = interpolate_field(b_i_data, 1)
    _, b_z = interpolate_field(b_i_data, 2)
    _, jac = interpolate_field(jac_data, 0)
    _, bmag = interpolate_field(bmag_data, 0)
    CCC = get_center_coords(grid)
    x_vals, y_vals, z_vals = CCC
    z_slice = get_slice_index(z_vals, z_idx)
    phi = phi[:, :, :, 0]
    b_x = b_x[:, :, :, 0]
    b_y = b_y[:, :, :, 0]
    b_z = b_z[:, :, :, 0]
    jac = jac[:, :, :, 0]
    bmag = bmag[:, :, :, 0]
    dx, dy, dz = x_vals[1] - x_vals[0], y_vals[1] - y_vals[0], z_vals[1] - z_vals[0]
    Ex = -np.gradient(phi, dx, axis=0)
    dphi_dx = np.gradient(phi, dx, axis=0)/jac
    dphi_dy = np.gradient(phi, dy, axis=1)/jac
    dphi_dz = np.gradient(phi, dz, axis=2)/jac
    VE_x = (b_y*dphi_dz - b_z*dphi_dy)/bmag
    VE_y = (b_z*dphi_dx - b_x*dphi_dz)/bmag
    VE_shear = np.gradient(VE_y, dx, axis=0)
    VE_x = VE_x[:, :, z_slice] # return 2d
    VE_y = VE_y[:, :, z_slice]
    VE_y_1d = np.mean(VE_y, axis=1)
    VE_shear = np.mean(VE_shear[:, :, z_slice], axis=1)
    Ex = np.mean(Ex[:, :, z_slice], axis=1)
    return VE_x, VE_y, VE_y_1d, VE_shear, Ex

def func_calc_reynolds_stress(phi_data, b_i_data, jac_data, bmag_data, z_idx):
    grid, phi = interpolate_field(phi_data, 0)
    _, b_x = interpolate_field(b_i_data, 0)
    _, b_y = interpolate_field(b_i_data, 1)
    _, b_z = interpolate_field(b_i_data, 2)
    _, jac = interpolate_field(jac_data, 0)
    _, bmag = interpolate_field(bmag_data, 0)
    CCC = get_center_coords(grid)
    x_vals, y_vals, z_vals = CCC
    z_slice = get_slice_index(z_vals, z_idx)
    phi = phi[:, :, :, 0]
    b_x = b_x[:, :, :, 0]
    b_y = b_y[:, :, :, 0]
    b_z = b_z[:, :, :, 0]
    jac = jac[:, :, :, 0]
    bmag = bmag[:, :, :, 0]
    dx, dy, dz = x_vals[1] - x_vals[0], y_vals[1] - y_vals[0], z_vals[1] - z_vals[0]
    dphi_dx = np.gradient(phi, dx, axis=0)/jac
    dphi_dy = np.gradient(phi, dy, axis=1)/jac
    dphi_dz = np.gradient(phi, dz, axis=2)/jac
    VE_x = (b_y*dphi_dz - b_z*dphi_dy)/bmag
    VE_y = (b_z*dphi_dx - b_x*dphi_dz)/bmag
    VE_x_fluc = VE_x - np.mean(VE_x, axis=(1), keepdims=True)
    VE_y_fluc = VE_y - np.mean(VE_y, axis=(1), keepdims=True)
    RS_xy = np.mean(VE_x_fluc*VE_y_fluc, axis=(1))
    return RS_xy

def func_time_ave(data_list):
    return np.mean(np.array(data_list), axis=0)

def func_calc_norm_fluc(data2d, dataAve, dataNorm, Nt, Ny, Nx):
    data2dTot = np.reshape(data2d, (Nt * Ny, Nx))
    dataAve2d = np.array([dataAve] * (Nt * Ny))
    delt = data2dTot - dataAve2d
    sigma = np.sqrt(np.mean(delt ** 2, axis=0))
    #delt = np.mean(delt, axis=0)
    delt_norm = sigma / dataNorm
    return delt, delt_norm

def save_to_hdf5(filename, x_vals, diagnostics, metadata):
    with h5py.File(filename, "w") as f:
        f.create_dataset("x_vals", data=x_vals)
        for name, data in diagnostics.items():
            f.create_dataset(name, data=data)
        for key, value in metadata.items():
            f.attrs[key] = value