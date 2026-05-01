import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fnmatch
import os
import re
import postgkyl as pg

def get_center_coords(interp_grid):
    return [(grid[1:] + grid[:-1]) / 2 for grid in interp_grid]

def interpolate_field(field3d, comp):
    interp = pg.data.GInterpModal(field3d, 1, 'ms')
    return interp.interpolate(comp)

def func_data_3d(filename, comp=0):
    gdata = pg.data.GData(filename)
    grid, values = interpolate_field(gdata, comp)
    CCC = get_center_coords(grid)
    return CCC[0], CCC[1], CCC[2], values[:, :, :, 0]

def find_prefix(pattern, path):
    for name in os.listdir(path):
        if fnmatch.fnmatch(name, '*' + pattern):
            return re.sub(pattern, '', name)
    raise FileNotFoundError("ERROR: file prefix not found!")

# =============================================================================
# 1. Physical Constants & Plasma Parameters
# =============================================================================
m_i = 3.34e-27        # Deuterium ion mass (kg)
m_e = 9.109e-31      # Electron mass (kg)
q_e = 1.602e-19       # Elementary charge (C) to convert eV to Joules
E_n = 10.0            # Assumed neutral energy (eV)
v_n = np.sqrt(E_n * q_e / m_i)  # Characteristic neutral velocity (~17 km/s)

# Find prefix
prefix = find_prefix("-field_0.gkyl", ".")
frame = 50

# --- Load and Interpolate Ionization Rate ---
# load data with frame and prefix
data_ion = pg.GData(f"{prefix}-ion_elc_react_iz_D0_{frame}.gkyl")
# Use 'ms' (Modal Serendipity) since this is a spatial field, not 5D phase space
interp_ion = pg.GInterpModal(data_ion, poly_order=1, basis_type="ms") 
grid, sv_iz = interp_ion.interpolate()
xc, yc, zc = get_center_coords(grid)

# Squeeze removes empty dimensions and slice the 2D plane you need (e.g., z2=0)
sv_ion = np.squeeze(sv_iz[:, 0, :, 0])  # Shape should match (Nx, Nz)

# --- Load and Interpolate Charge Exchange Rate ---
data_cx = pg.GData(f"{prefix}-ion_react_cx_D0_{frame}.gkyl")
interp_cx = pg.GInterpModal(data_cx, poly_order=1, basis_type="ms")
_, sv_cx = interp_cx.interpolate()

sv_cx = np.squeeze(sv_cx[:, 0, :, 0])

# =============================================================================
# 2. Grid Setup & field data
# =============================================================================
Nx, Nz = len(xc), len(zc)
dx = xc[1] - xc[0]
dz = zc[1] - zc[0]

# Read density and temperatures
data_elc_moms = pg.GData(f"{prefix}-elc_BiMaxwellianMoments_{frame}.gkyl")
data_ion_moms = pg.GData(f"{prefix}-ion_BiMaxwellianMoments_{frame}.gkyl")
interp_elc = pg.GInterpModal(data_elc_moms, poly_order=1, basis_type="ms")
interp_ion = pg.GInterpModal(data_ion_moms, poly_order=1, basis_type="ms")
_, ne = interp_elc.interpolate(0)
_, vte_par = interp_elc.interpolate(2) 
_, vte_perp = interp_elc.interpolate(3)
_, vti_par = interp_ion.interpolate(1)
_, vti_perp = interp_ion.interpolate(2)
vte_sq = (vte_par + 2*vte_perp)/3.0 
vti_sq = (vti_par + 2*vti_perp)/3.0
ne = np.squeeze(ne[:, 0, :, 0])
vte_sq = np.squeeze(vte_sq[:, 0, :, 0])
vti_sq = np.squeeze(vti_sq[:, 0, :, 0]) 
vte = np.sqrt(vte_sq)
vti = np.sqrt(vti_sq)
cs = vte * np.sqrt(m_e / m_i)

# Curvilinear metrics (assuming uniform for this test, replace with your gkyl data)
data_jac = pg.GData(f"{prefix}-jacobgeo.gkyl")
interp_jac = pg.GInterpModal(data_jac, poly_order=1, basis_type="ms")
_, J = interp_jac.interpolate()
J = np.squeeze(J[:, 0, :, 0])  # Shape should match (

data_gij = pg.GData(f"{prefix}-gij.gkyl")
interp_gij = pg.GInterpModal(data_gij, poly_order=1, basis_type="ms")
_, gxx = interp_gij.interpolate(0)
_, gzz = interp_gij.interpolate(5)
gxx = np.squeeze(gxx[:, 0, :, 0])
gzz = np.squeeze(gzz[:, 0, :, 0])

# Calculate derived physics arrays
Dn = (vti**2) / (ne * sv_cx)  # Neutral diffusion coeff

# =============================================================================
# 3. Define the Immersed Limiter Mask
# =============================================================================
# The limiters are at the first and last poloidal indices
k_lim_low = 0
k_lim_up = Nz - 1
limiter_i_tip = int((2.0 / 3.0) * Nx) 

# Calculate the recycling source at BOTH limiter faces
# (Using the local plasma parameters specifically at k=0 and k=Nz3-1)
n_recycle_low = ne[:, k_lim_low] * (cs[:, k_lim_low] / v_n)
n_recycle_up = ne[:, k_lim_up] * (cs[:, k_lim_up] / v_n)

z_min, z_max = zc[0], zc[-1]
L_pol = 0.15 # Characteristic
pol_left = np.cosh((zc - z_min) / L_pol)**-2 + 1e-5
pol_right = np.cosh((zc - z_max) / L_pol)**-2 + 1e-5
n_wall_bg = n_recycle_low[-1]*pol_left + n_recycle_up[-1]*pol_right    # Background main wall neutral density (m^-3)

# =============================================================================
# 4. Build the Sparse Matrix
# =============================================================================
N_total = Nx * Nz
A = sp.lil_matrix((N_total, N_total))
b = np.zeros(N_total)

# Helper function to map 2D (i, k) to 1D flat index and handle periodic Poloidal BCs
def get_idx(i, k):
    return i * Nz + (k % Nz)

print("Building 2D curvilinear sparse matrix...")

for i in range(Nx):
    for k in range(Nz):
        row = get_idx(i, k)
        
    # --- Boundary Condition 1: Inner Core (Neumann) ---
        if i == 0:
            A[row, row] = -1.0
            A[row, get_idx(i+1, k)] = 1.0
            b[row] = 0.0
            
        # --- Boundary Condition 2: Main Wall (Dirichlet n=n_wall) ---
        if i == Nx - 1:
            if k == k_lim_low:
                A[row, row] = 1.0
                b[row] = n_recycle_low[i]
            elif k == k_lim_up:
                A[row, row] = 1.0
                b[row] = n_recycle_up[i]
            else:
                A[row, row] = 1.0
                b[row] = n_wall_bg[k]
                # A[row, row] = -1.0
                # A[row, get_idx(i-1, k)] = 1.0
                # b[row] = 0.0
            continue
            
        if i >= limiter_i_tip:
            if k == k_lim_low:
                A[row, row] = 1.0
                b[row] = n_recycle_low[i]
                continue
            if k == k_lim_up:
                A[row, row] = 1.0
                b[row] = n_recycle_up[i]
                continue

        if i < limiter_i_tip:
            # CLOSED CORE: Field lines wrap continuously around the tokamak
            k_plus = (k + 1) % Nz
            k_minus = (k - 1) % Nz
        else:
            # OPEN SOL: Field lines hit the limiter (No periodic wrap)
            k_plus = k + 1
            k_minus = k - 1
            
        # --- The PDE: Standard Plasma Cell ---
        # Calculate face-centered transport coefficients (averaging adjacent cell centers)
        # Radial fluxes
        JDgxx_plus  = 0.5 * (J[i+1, k]*Dn[i+1, k]*gxx[i+1, k] + J[i, k]*Dn[i, k]*gxx[i, k])
        JDgxx_minus = 0.5 * (J[i, k]*Dn[i, k]*gxx[i, k] + J[i-1, k]*Dn[i-1, k]*gxx[i-1, k])
        c_rad_plus  = JDgxx_plus  / (J[i, k] * dx**2)
        c_rad_minus = JDgxx_minus / (J[i, k] * dx**2)
        
        # Poloidal fluxes (with periodic wrapping)
        k_plus = (k + 1) % Nz
        k_minus = (k - 1) % Nz
        JDgzz_plus  = 0.5 * (J[i, k_plus]*Dn[i, k_plus]*gzz[i, k_plus] + J[i, k]*Dn[i, k]*gzz[i, k])
        JDgzz_minus = 0.5 * (J[i, k]*Dn[i, k]*gzz[i, k] + J[i, k_minus]*Dn[i, k_minus]*gzz[i, k_minus])
        c_pol_plus  = JDgzz_plus  / (J[i, k] * dz**2)
        c_pol_minus = JDgzz_minus / (J[i, k] * dz**2)
        
        # Center coefficient includes the ionization sink
        c_center = -(c_rad_plus + c_rad_minus + c_pol_plus + c_pol_minus) - (ne[i, k] * sv_ion[i, k])
        
        # Populate matrix row
        A[row, get_idx(i+1, k)] = c_rad_plus
        A[row, get_idx(i-1, k)] = c_rad_minus
        A[row, get_idx(i, k_plus)] = c_pol_plus
        A[row, get_idx(i, k_minus)] = c_pol_minus
        A[row, row] = c_center
        
        b[row] = 0.0

# =============================================================================
# 5. Solve the Linear System
# =============================================================================
print("Solving system...")
# Convert to Compressed Sparse Row (CSR) format for fast math
A_csr = A.tocsr()
n_flat = spla.spsolve(A_csr, b)

# Reshape back to 2D
n_neutral = n_flat.reshape((Nx, Nz))

# Save the 2D array to a binary .npy file
np.save("n_neutral_profile.npy", n_neutral)
print("Saved neutral profile to 'n_neutral_profile.npy'")
print("Shape of neutral density array:", n_neutral.shape)

# =============================================================================
# 6. Plot the Results
# # =============================================================================
# fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')

# # Plot on a log scale since neutral density drops exponentially
# im = ax.pcolormesh(np.log10(n_neutral).T, cmap='magma', shading='auto')
# ax.set_title("Neutral Density")
# ax.set_xlabel("Radial Index (i)")
# ax.set_ylabel("Poloidal Index (k)")

# # Draw limiter location on last third of x-axis
# limiter_x = np.arange(limiter_i_tip, Nx)
# ax.plot(limiter_x, np.full_like(limiter_x, k_lim_low), 'w--', label='Limiter Location')

# fig.colorbar(im, ax=ax, label=r"$\log_{10}(n_n)$ [$m^{-3}$]")
# ax.legend()
#plt.show()

# 1. Load your numerical data
# n_neutral = np.load("n_neutral_profile.npy")
# (Assuming n_neutral, Nz1, Nz3, limiter_i_tip are already in memory)

# =============================================================================
# 1. Map Logical Indices to Physical Coordinates
# =============================================================================
# --- UPDATE THESE TO MATCH YOUR GKEYLL DOMAIN ---
x_core, x_wall = xc[0], xc[-1]        # Physical radial domain (e.g., meters)
z_min, z_max = zc[0], zc[-1]         # Physical poloidal domain (e.g., meters)
Lz = z_max - z_min
print(x_core, x_wall, z_min, z_max)

x_phys = np.linspace(x_core, x_wall, Nx)
z_phys = np.linspace(z_min, z_max, Nz)
x_LCFS = x_phys[limiter_i_tip]   # Physical location of the limiter tip

# Create 2D coordinate grids
X, Z = np.meshgrid(x_phys, z_phys, indexing='ij')

# Flatten everything for the curve fitter
x_flat = X.ravel()
z_flat = Z.ravel()
n_flat = n_neutral.ravel()

# =============================================================================
# 3. Perform the Curve Fit (IN LOG SPACE)
# =============================================================================
def log_analytic_model(coords, amp, L_rad, L_pol, floor):
    x, z = coords
    
    # Radial step function
    radial = 0.5 * (1.0 - np.tanh((x_LCFS - x) / L_rad))
    
    # Poloidal sech^2 pulses
    pol_left = np.cosh((z - z_min) / L_pol)**-2
    pol_right = np.cosh((z - z_max) / L_pol)**-2
    
    # Calculate linear density, then take log10
    # Added a tiny epsilon to prevent log10(0) just in case
    n = amp * radial * (pol_left + pol_right) + floor
    return np.log10(n + 1e-10)

# Provide initial guesses based on your axes
# [Amplitude, L_rad (~0.01), L_pol (~0.5), Floor (~1e13)]
p0 = [np.max(n_neutral), 0.01, 0.5, 1e13]

# Set bounds to keep the physics reasonable
bounds = (
    [0, 1e-4, 1e-4, 1e13],          # Lower bounds
    [np.inf, 0.1, 5.0, 1e19]        # Upper bounds
)

print("Fitting 2D surface in log-space...")
# Notice we pass np.log10(n_flat) as the target data now!
popt, pcov = curve_fit(log_analytic_model, (x_flat, z_flat), np.log10(n_flat), p0=p0, bounds=bounds)

amp_fit, L_rad_fit, L_pol_fit, floor_fit = popt

popt_plt = [amp_fit, L_rad_fit, L_pol_fit, floor_fit]

print("\n--- FITTED PARAMETERS FOR C CODE ---")
print(f"alpha_neut (Amplitude) = {amp_fit:.4e}")
print(f"L_rad (Radial Denom)   = {L_rad_fit:.4f}")
print(f"L_pol (Poloidal Denom) = {L_pol_fit:.4f}")
print(f"Floor                  = {floor_fit:.4e}")

def log_radial_model(x, amp, L_rad, bg):
    # Pure radial tanh step
    radial = 0.5 * (-np.tanh((x_LCFS - x) / L_rad) + 1.0)
    
    # Calculate density and return log10
    n = amp * radial + bg
    return np.log10(n)

# =============================================================================
# 3. Perform the 1D Curve Fit
# =============================================================================
n_1d = n_neutral[:, k_lim_low]  # Take the slice at the limiter
min_val = np.min(n_1d)
max_val = np.max(n_1d)

# Guesses: [Amplitude, L_rad, Background]
p0 = [max_val, 0.01, min_val]

# Bounds
bounds = (
    [0,      1e-4, min_val * 0.1],  # Lower bounds 
    [np.inf, 0.1,  min_val * 10.0]  # Upper bounds
)

print("Fitting 1D radial slice at the limiter...")
popt, pcov = curve_fit(log_radial_model, xc, np.log10(n_1d), p0=p0, bounds=bounds)

amp_fit, L_rad_fit, bg_fit = popt

print("\n--- FITTED RADIAL PARAMETERS ---")
print(f"alpha_neut (Amplitude) = {amp_fit:.4e}")
print(f"L_rad (Radial Denom)   = {L_rad_fit:.4f}")
print(f"Background             = {bg_fit:.4e}")

popt_plt[1] = 0.05  # Update the radial length scale in the 2D model with the 1D fit result

# =============================================================================
# 4. Verify the Fit Visually
# =============================================================================
# We need to exponentiate the output of the log model to plot it linearly
n_fit_log = log_analytic_model((X, Z), *popt_plt)
n_fit = 10**n_fit_log

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')

# Use vmin and vmax so both plots share the exact same color scale!
vmin = np.log10(np.min(n_neutral))
vmax = np.log10(np.max(n_neutral))

im1 = ax1.pcolormesh(X, Z, np.log10(n_neutral), cmap='magma', shading='auto', vmin=vmin, vmax=vmax)
ax1.set_title("Numerical Diffusion Solution")
ax1.set_xlabel("Radial (x)")
ax1.set_ylabel("Poloidal (z)")
fig.colorbar(im1, ax=ax1)

im2 = ax2.pcolormesh(X, Z, n_fit_log, cmap='magma', shading='auto', vmin=vmin, vmax=vmax)
ax2.set_title("Analytic Fit")
ax2.set_xlabel("Radial (x)")
ax2.set_ylabel("Poloidal (z)")
fig.colorbar(im2, ax=ax2)

plt.show()