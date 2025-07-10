import math
import os
import re
import fnmatch
import numpy as np
import postgkyl as pg

def miller_plasma_volume(R, a, kappa, delta, sol_thickness=0.0):
    """
    Estimate tokamak plasma volume using Miller geometry.

    Parameters:
        R (float): Major radius (m)
        a (float): Minor radius (m)
        kappa (float): Elongation
        delta (float): Triangularity (0 <= delta < 1)
        sol_thickness (float): Thickness of scrape-off layer (m)

    Returns:
        dict: Volumes in cubic meters
            - core_volume
            - total_volume (core + SOL)
            - sol_volume (just the SOL region)
    """
    f_delta = 1 - 0.5 * delta**2
    a_inner = a - 0.1

    # Core plasma volume
    core_volume = 2 * math.pi**2 * R * a**2 * kappa * f_delta

    core_volume_inner = 2 * math.pi**2 * R * a_inner**2 * kappa * f_delta
    core_volume -= core_volume_inner

    # Effective minor radius with SOL
    a_eff = a + sol_thickness
    total_volume = 2 * math.pi**2 * R * a_eff**2 * kappa * f_delta

    # SOL volume is the difference
    sol_volume = total_volume - core_volume

    # LCFS surface area (outer surface of the plasma core)
    surface_area_lcfs = 4 * math.pi**2 * R * a * kappa * f_delta

    return {
        "core_volume_m3": core_volume,
        "total_volume_m3": total_volume,
        "sol_volume_m3": sol_volume,
        "surface_area_lcfs_m2": surface_area_lcfs
    }

def find_prefix(pattern, path):
    for name in os.listdir(path):
        if fnmatch.fnmatch(name, '*' + pattern):
            return re.sub(pattern, '', name)
    raise FileNotFoundError("ERROR: file prefix not found!")

def interpolate_field(field3d, comp):
    interp = pg.data.GInterpModal(field3d, 1, 'ms')
    return interp.interpolate(comp)

def get_center_coords(interp_grid):
    return [(grid[1:] + grid[:-1]) / 2 for grid in interp_grid]

def qprofile_PT(r):
    R = r + R_axis
    a = [407.582626469394, -2468.613680167604, 4992.660489790657, -3369.710290916853]
    return a[0]*R**3 + a[1]*R**2 + a[2]*R + a[3]
def qprofile_NT(r):
    R = r + R_axis
    a = [154.51071835546747,  -921.8584472748003, 1842.1077075366113, -1231.619813170522]
    return a[0]*R**3 + a[1]*R**2 + a[2]*R + a[3]

def calc_core_volume(jac, g_ij):
    grid, jac_vals = interpolate_field(jac, 0)
    _ , g_xx_vals = interpolate_field(g_ij, 0)
    CCC = get_center_coords(grid)
    dx = CCC[0][1] - CCC[0][0]  
    dy = CCC[1][1] - CCC[1][0]
    dz = CCC[2][1] - CCC[2][0]
    lcfs_idx = int(len(CCC[1])*2/3) 
    volume = np.sum(jac_vals[:,:,:,0]) * dx * dy * dz
    area = np.sum(jac_vals[lcfs_idx,:,:,0]/np.sqrt(g_xx_vals[lcfs_idx,:,:,0])) * dy * dz
    Ly = CCC[1][-1] - CCC[1][0]
    ntoroidal = 2*np.pi*a/(Ly*q0)
    return volume, area, ntoroidal

# Example use
if __name__ == "__main__":
    R_axis = 1.7074685      # Major radius (m) # PT 1.6486461 # NT 1.7074685
    a = 2.17-R_axis         # Minor radius (m)
    kappa = 1.35     # Elongation
    delta = -0.4     # Triangularity
    sol = 0.05      # SOL thickness (m)
    q0 = qprofile_PT(a)

    volumes = miller_plasma_volume(R_axis, a, kappa, delta, sol)

    print("Core Plasma Volume: {:.4f} m³".format(volumes["core_volume_m3"]))
    print("SOL Volume: {:.4f} m³".format(volumes["sol_volume_m3"]))
    print("Total Plasma + SOL Volume: {:.4f} m³".format(volumes["total_volume_m3"]))
    
    # Load the data
    file_prefix = find_prefix('-field_0.gkyl', '.')
    jacgeo_data = pg.GData(f"{file_prefix}-jacobgeo.gkyl")
    g_ij_data = pg.GData(f"{file_prefix}-g_ij.gkyl")
    sim_core_vol, sim_lcfs_area, ntoroidal = calc_core_volume(jacgeo_data, g_ij_data)

    print("Simulated Volume: {:.4f} m³".format(sim_core_vol))
    print("1/ntoroidal = ", 1/ntoroidal)
