import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utils
import postgkyl as pg

# =============================================================================
# 1. DEFINE THE ANALYTIC MODEL FUNCTIONS
# =============================================================================

def model_c0(Z, amp1, center1, width1, amp2, center2, width2):
    """A sum of two independent Gaussian functions to model two spikes."""
    gauss1 = amp1 * np.exp(-((Z - center1)**2) / (2 * width1**2))
    gauss2 = amp2 * np.exp(-((Z - center2)**2) / (2 * width2**2))
    return gauss1 + gauss2

def model_c1(Z, amplitude, width):
    """An odd cubic polynomial function: A * Z * (1 - (Z/W)**2)."""
    return amplitude * Z * (1 - (Z / width)**2)

def model_c4(Z, amplitude, center, width):
    """A standard Gaussian (bell curve) function."""
    return amplitude * np.exp(-((Z - center)**2) / (2 * width**2))

# =============================================================================
# 2. AUTOMATIC GUESSING FUNCTIONS
# =============================================================================

def _find_guesses_c0(Z_data, y_data):
    """Analyzes data to provide smart initial guesses for a two-Gaussian fit."""
    mid_idx = len(Z_data) // 2
    Z_left, y_left = Z_data[:mid_idx], y_data[:mid_idx]
    Z_right, y_right = Z_data[mid_idx:], y_data[mid_idx:]

    # Left Peak
    idx1 = np.argmax(y_left)
    amp1, center1 = y_left[idx1], Z_left[idx1]
    try:
        #w1_indices = np.where(y_left > amp1 / 2)[0]
        width1 = (Z_left[1] - Z_left[0]) / 2.355
    except IndexError:
        width1 = 1.0

    # Right Peak
    idx2 = np.argmax(y_right)
    amp2, center2 = y_right[idx2], Z_right[idx2]
    try:
        #w2_indices = np.where(y_right > amp2 / 2)[0]
        width2 = (Z_right[-2] - Z_right[-1]) / 2.355
    except IndexError:
        width2 = 1.0
        
    p0 = [amp1, center1, width1, amp2, center2, width2]
    print("--- c0 Initial Guesses ---")
    print(f"  Left:  Amp≈{p0[0]:.2e}, Center≈{p0[1]:.2f}, Width≈{p0[2]:.2f}")
    print(f"  Right: Amp≈{p0[3]:.2e}, Center≈{p0[4]:.2f}, Width≈{p0[5]:.2f}")
    return p0

def _find_guesses_c1(Z_data, y_data):
    """Provides smart initial guesses for the odd cubic polynomial fit."""
    print("--- Finding c1 Initial Guesses ---")
    # Find the main positive peak
    peak_idx = np.argmax(y_data)
    peak_y, peak_z = y_data[peak_idx], Z_data[peak_idx]
    
    # For f(Z) = A*Z*(1-(Z/W)**2), the peak is at Z = W/sqrt(3).
    # So, a good guess for the zero-crossing width W is peak_z * sqrt(3).
    width_guess = peak_z * np.sqrt(3)
    
    # From the formula, the peak value is A * (2*W**3)/(3*sqrt(3)*W).
    # So, A = PeakValue * (3*sqrt(3)) / (2*W**2) -- this is wrong.
    # The peak value is A*peak_z*(1-(peak_z/width_guess)**2)
    # A = peak_y / (peak_z * (1 - (peak_z/width_guess)**2))
    amp_guess = peak_y / (peak_z * (1 - (peak_z/width_guess)**2))
    
    p0 = [amp_guess, width_guess]
    print("--- c1 Initial Guesses ---")
    print(f"  Amp≈{p0[0]:.2f}, Width≈{p0[1]:.2f}")
    return p0

def _find_guesses_c4(Z_data, y_data):
    """Provides smart initial guesses for a single Gaussian fit."""
    peak_idx = np.argmax(y_data)
    amp_guess = y_data[peak_idx]
    center_guess = Z_data[peak_idx]
    
    try:
        width_indices = np.where(y_data > amp_guess / 2)[0]
        width_guess = (Z_data[width_indices[-1]] - Z_data[width_indices[0]]) / 2.355
    except IndexError:
        width_guess = (Z_data[-1] - Z_data[0]) / 4 # Fallback
        
    p0 = [amp_guess, center_guess, width_guess]
    print("--- c4 Initial Guesses ---")
    print(f"  Amp≈{p0[0]:.2e}, Center≈{p0[1]:.2f}, Width≈{p0[2]:.2f}")
    return p0

def print_c_functions(params_dict, var_name="Z0"):
    """
    Prints the fitted functions in a C-style format for easy copy-pasting.
    """
    print("\n" + "="*50)
    print("--- C-Style Analytic Functions ---")
    print(f"--- (independent variable is '{var_name}') ---")
    print("="*50)

    # --- c0 function ---
    if 'c0' in params_dict:
        p = params_dict['c0']
        # Use .8e for high precision in scientific notation
        c0_str = (f"c0({var_name}) = {p[0]:.8e} * exp(-pow({var_name} - ({p[1]:.8e}), 2.0) / (2.0 * pow({p[2]:.8e}, 2.0))) + "
                  f"{p[3]:.8e} * exp(-pow({var_name} - ({p[4]:.8e}), 2.0) / (2.0 * pow({p[5]:.8e}, 2.0)));")
        print("c0 (Two-Gaussian Spikes):\n" + c0_str + "\n")

    # --- c1 function ---
    if 'c1' in params_dict:
        p = params_dict['c1']
        c1_str = f"c1({var_name}) = {p[0]:.8e} * {var_name} * (1.0 - pow({var_name} / {p[1]:.8e}, 2.0));"
        print("c1 (Cubic Polynomial):\n" + c1_str + "\n")

    # --- c4 function ---
    if 'c4' in params_dict:
        p = params_dict['c4']
        c4_str = f"c4({var_name}) = {p[0]:.8e} * exp(-pow({var_name} - ({p[1]:.8e}), 2.0) / (2.0 * pow({p[2]:.8e}, 2.0)));"
        print("c4 (Gaussian):\n" + c4_str + "\n")
        
    print("="*50)

# =============================================================================
# 3. Read the data
# =============================================================================

file_prefix = utils.find_prefix('-field_0.gkyl', '.')
print(f"Using file prefix: {file_prefix}")

frame = int(input("frame? "))

eV = 1.602e-19
mp = 1.672623e-27
moms_data = pg.GData(f"{file_prefix}-H0_LTEMoments_{frame}.gkyl")
grid, c0_data = utils.interpolate_field(moms_data, 0)
_, c1_data = utils.interpolate_field(moms_data, 1)
_, c4_data = utils.interpolate_field(moms_data, 4)
c0_data = c0_data[:,0]
c1_data = c1_data[:,0]
c4_data = c4_data[:,0]/eV*mp  # Convert to eV units
Z_data = utils.get_center_coords(grid)[0]

# =============================================================================
# 4. PERFORM THE CURVE FITTING
# =============================================================================
print("\n--- Performing Curve Fits ---")
all_params = {}
try:
    p0_c0 = _find_guesses_c0(Z_data, c0_data)
    params_c0, _ = curve_fit(model_c0, Z_data, c0_data, p0=p0_c0)
    all_params['c0'] = params_c0
    print("  c0 fit successful.")
except Exception as e:
    print(f"  c0 fit FAILED: {e}")

try:
    p0_c1 = _find_guesses_c1(Z_data, c1_data)
    params_c1, _ = curve_fit(model_c1, Z_data, c1_data, p0=p0_c1)
    all_params['c1'] = params_c1
    print("  c1 fit successful.")
except Exception as e:
    print(f"  c1 fit FAILED: {e}")

try:
    p0_c4 = _find_guesses_c4(Z_data, c4_data)
    params_c4, _ = curve_fit(model_c4, Z_data, c4_data, p0=p0_c4)
    all_params['c4'] = params_c4
    print("  c4 fit successful.")
except Exception as e:
    print(f"  c4 fit FAILED: {e}")

# =============================================================================
# 5. PLOT THE RESULTS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Analytic Fits to Data", fontsize=18)

# --- Plot c0 fit ---
ax = axes[0]
ax.plot(Z_data, c0_data, 'o', label='Data', markersize=4, alpha=0.5)
if 'c0' in all_params:
    p = all_params['c0']
    ax.plot(Z_data, model_c0(Z_data, *p), 'r-', lw=2, label=f'Fit: A₁={p[0]:.2e}, C₁={p[1]:.2f}\nA₂={p[3]:.2e}, C₂={p[4]:.2f}')
ax.set_title("c0 (Two-Gaussian Fit)", fontsize=14)
ax.set_xlabel("$Z_0$"), ax.legend(), ax.grid(True, linestyle=':')

# --- Plot c1 fit ---
ax = axes[1]
ax.plot(Z_data, c1_data, 'o', label='Data', markersize=4, alpha=0.5)
if 'c1' in all_params:
    p = all_params['c1']
    ax.plot(Z_data, model_c1(Z_data, *p), 'r-', lw=2, label=f'Fit: A={p[0]:.2f}, W={p[1]:.2f}')
ax.set_title("c1 (Cubic Polynomial Fit)", fontsize=14)
ax.set_xlabel("$Z_0$"), ax.legend(), ax.grid(True, linestyle=':')

# --- Plot c4 fit ---
ax = axes[2]
ax.plot(Z_data, c4_data, 'o', label='Data', markersize=4, alpha=0.5)
if 'c4' in all_params:
    p = all_params['c4']
    ax.plot(Z_data, model_c4(Z_data, *p), 'r-', lw=2, label=f'Fit: A={p[0]:.2e}, Ctr={p[1]:.2f}, W={p[2]:.2f}')
ax.set_title("c4 (Gaussian Fit)", fontsize=14)
ax.set_xlabel("$Z_0$"), ax.legend(), ax.grid(True, linestyle=':')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"analytic_fits_frame{frame}.png", dpi=300)
plt.show()

if all_params:
    print_c_functions(all_params, var_name="Z0")