# -----------------------------------------------------------------------------
# plot_geqdsk_equilibrium.py
#
# A script to read a GEQDSK file and generate two plots:
#   1. The poloidal cross-section of the magnetic equilibrium, with an
#      optional overlay of an approximate Miller geometry equilibrium.
#   2. The safety factor (q) profile as a function of the major radius.
#
# Usage:
#   python plot_geqdsk_equilibrium.py /path/to/your/gfile.geqdsk
#
# For more options, run:
#   python plot_geqdsk_equilibrium.py --help
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from freeqdsk import geqdsk
from scipy.interpolate import CubicSpline
import sys
import argparse  # Import the argument parsing library
import os        # Import for path manipulation

# --- Plotting Style (can be left as a global setting) ---
plt.rcParams.update({
    "font.size": 14,
    "lines.linewidth": 2.5,
    "image.cmap": 'viridis',
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ========================= HELPER FUNCTIONS ================================

def get_r_lcfs(gfile_data):
    """
    Calculates R_LCFS at the Outer Midplane (height of magnetic axis).
    
    Args:
        gfile_data (dict): Dictionary containing 'rbdry', 'zbdry', and 'zmaxis'.
        
    Returns:
        float: R coordinate of the LCFS at the outer midplane.
    """
    R_b = gfile_data['rbdry']
    Z_b = gfile_data['zbdry']
    Z_axis = gfile_data['zmaxis'] # Use 0.0 if you strictly want geometric midplane
    
    # We want to find R where Z_b crosses Z_axis.
    # Since the boundary is a loop, we iterate through segments.
    
    intersections = []
    
    n_points = len(R_b)
    for i in range(n_points):
        # Get points for current segment (wrapping around at the end)
        z1 = Z_b[i]
        z2 = Z_b[(i + 1) % n_points]
        r1 = R_b[i]
        r2 = R_b[(i + 1) % n_points]
        
        # Check if the segment crosses Z_axis
        if (z1 - Z_axis) * (z2 - Z_axis) <= 0:
            # Avoid division by zero if horizontal segment (unlikely)
            if z1 == z2:
                intersections.append(r1)
                continue
            
            # Linear interpolation to find R at Z_axis
            slope = (r2 - r1) / (z2 - z1)
            r_cross = r1 + slope * (Z_axis - z1)
            intersections.append(r_cross)
    
    if not intersections:
        raise ValueError("Could not find intersection with midplane.")
        
    # The Outer Midplane is the Maximum R intersection
    return max(intersections)

def calculate_miller_parameters(gfile_data):
    """Calculates key geometric parameters (a, kappa, delta) from g-file LCFS."""
    rbdry, zbdry = gfile_data["rbdry"], gfile_data["zbdry"]
    r_max, r_min = np.max(rbdry), np.min(rbdry)
    a = (r_max - r_min) / 2.0
    z_max, z_min = np.max(zbdry), np.min(zbdry)
    kappa = (z_max - z_min) / (r_max - r_min)
    r_geometric_center = (r_max + r_min) / 2.0
    r_at_top_of_lcfs = rbdry[np.argmax(zbdry)]
    delta = (r_geometric_center - r_at_top_of_lcfs) / a
    
    return {
        'R_axis': gfile_data["rmagx"], 'Z_axis': gfile_data["zmagx"],
        'a': a, 'kappa': kappa, 'delta': delta
    }

def generate_miller_lcfs(miller_params, shafranov_param, a_override=None):
    """Generates R, Z coordinates for a Miller LCFS, optionally with a different minor radius."""
    theta = np.linspace(0, 2 * np.pi, 200)
    R_axis, Z_axis = miller_params['R_axis'], miller_params['Z_axis']
    a = miller_params['a'] if a_override is None else a_override
    kappa = miller_params['kappa']
    delta = miller_params['delta']
    R_miller = R_axis - shafranov_param * a**2 / (2. * R_axis) + a * np.cos(theta + np.arcsin(delta) * np.sin(theta))
    Z_miller = Z_axis + kappa * a * np.sin(theta)
    return R_miller, Z_miller

# ========================= CORE ANALYSIS SCRIPT ============================

def analyze_and_plot(gfile_path, plot_miller, shafranov_param, save_plots, output_dir, figure_format, x_in, x_out):
    """
    Main function to perform all analysis and plotting.
    """
    # --- 1. Load Data from G-file ---
    try:
        with open(gfile_path, "r") as f:
            print(f"Reading g-file: {gfile_path}")
            gfile_data = geqdsk.read(f)
    except FileNotFoundError:
        print(f"Error: The file '{gfile_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the g-file: {e}", file=sys.stderr)
        sys.exit(1)

    psi_RZ = gfile_data["psi"]
    R_grid = np.linspace(gfile_data["rleft"], gfile_data["rleft"] + gfile_data["rdim"], gfile_data["nx"])
    Z_grid = np.linspace(gfile_data["zmid"] - gfile_data["zdim"]/2, gfile_data["zmid"] + gfile_data["zdim"]/2, gfile_data["ny"])
    RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing='ij')

    psi_sep = gfile_data["sibdry"]
    print(f"Psi at separatrix (LCFS): {psi_sep}")

    # --- 2. Plot the 2D Equilibrium ---
    print("Generating equilibrium plot...")
    fig1, ax1 = plt.subplots(figsize=(8, 10))
    levels = np.linspace(psi_RZ.min(), psi_RZ.max(), 20)
    contour = ax1.contourf(RR, ZZ, psi_RZ, levels=levels, vmin=psi_RZ.min(), vmax=psi_RZ.max()*.75)
    fig1.colorbar(contour, ax=ax1, label=r'$\psi$ (Wb/rad)')
    
    ax1.plot(gfile_data["rlim"], gfile_data["zlim"], 'k-', linewidth=2.0, label='Vessel Wall')
    ax1.plot(gfile_data["rbdry"], gfile_data["zbdry"], 'w-', linewidth=2.0, label='Experimental LCFS')
    ax1.plot(gfile_data["rmagx"], gfile_data["zmagx"], 'wx', markersize=10, mew=2.5, label='Magnetic Axis')

    # print magnetic axis location
    print(f"Magnetic Axis Location: R = {gfile_data['rmagx']:.4f} m, Z = {gfile_data['zmagx']:.4f} m")

    formatted_r_coords = ", ".join(map(str, gfile_data['rlim']))
    print(f"\nVessel Wall R Coordinates: {formatted_r_coords}")
    formatted_z_coords = ", ".join(map(str, gfile_data['zlim']))
    print(f"Vessel Wall Z Coordinates: {formatted_z_coords}")

    # Assuming 'g' is your loaded gfile object/dictionary
    r_axis = gfile_data['rmaxis']      # R position of magnetic axis [m]
    f_axis = gfile_data['fpol'][0]     # F = RB_phi at the axis [T*m]

    B_axis = gfile_data['bcentr'] #f_axis / r_axis  # B field on axis [T]

    print(f"B on axis: {B_axis:.4f} T")

    # Assuming you have loaded your gfile into a dict called 'g'
    # (using OMFIT, freegs, or a custom parser)

    r_lcfs = get_r_lcfs(gfile_data)
    print(f"R_LCFS (Outer Midplane): {r_lcfs:.4f} m")   

    if plot_miller:
        print("\n--- Miller Geometry Analysis ---")
        miller_params = calculate_miller_parameters(gfile_data)
        print(f"  a={miller_params['a']:.4f} m, κ={miller_params['kappa']:.4f}, δ={miller_params['delta']:.4f}")
        # Main Miller LCFS
        R_miller, Z_miller = generate_miller_lcfs(miller_params, shafranov_param)
        ax1.plot(R_miller, Z_miller, 'r--', linewidth=2.5, label=f'Miller LCFS')
        # 10cm inside (a - 0.10)
        R_miller_inner, Z_miller_inner = generate_miller_lcfs(miller_params, shafranov_param, a_override=miller_params['a'] - x_in)
        ax1.plot(R_miller_inner, Z_miller_inner, color='C1', linestyle=':', linewidth=3, label='inner/outer boundary')
        # 5cm outside (a + 0.05)
        R_miller_outer, Z_miller_outer = generate_miller_lcfs(miller_params, shafranov_param, a_override=miller_params['a'] + x_out)
        ax1.plot(R_miller_outer, Z_miller_outer, color='C1', linestyle=':', linewidth=3)

    gfile_basename = os.path.basename(gfile_path)
    ax1.set_title(f"Magnetic Equilibrium: {gfile_basename}")
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_aspect('equal')
    ax1.legend()
    plt.tight_layout()

    if save_plots:
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
        output_filename = os.path.join(output_dir, f"equilibrium_{gfile_basename}{figure_format}")
        plt.savefig(output_filename)
        print(f"\nSaved equilibrium plot to {output_filename}")
    plt.show()

    # --- 3. Calculate and Plot the Q-Profile vs. Major Radius ---
    print("\nGenerating q-profile plot...")
    q_profile = gfile_data["qpsi"]
    z_axis_idx = np.abs(Z_grid - gfile_data["zmagx"]).argmin()
    r_axis_idx = np.abs(R_grid - gfile_data["rmagx"]).argmin()
    psi_outboard_midplane = psi_RZ[r_axis_idx:, z_axis_idx]
    R_outboard_midplane = R_grid[r_axis_idx:]

    # 1. Sort the arrays so psi is strictly increasing
    sort_indices = np.argsort(psi_outboard_midplane)
    psi_sorted = psi_outboard_midplane[sort_indices]
    R_sorted   = R_outboard_midplane[sort_indices]

    # 2. Create the interpolator with sorted data
    R_of_psi_interpolator = CubicSpline(psi_sorted, R_sorted, extrapolate=False)
    psi_normalized = np.linspace(0, 1, len(q_profile))
    # Ensure your target psi_values are also within the sorted range
    psi_values = gfile_data["simagx"] + psi_normalized * (gfile_data["sibdry"] - gfile_data["simagx"])

    R_for_q_profile = R_of_psi_interpolator(psi_values)
    coeffs_q = np.polyfit(R_for_q_profile[~np.isnan(R_for_q_profile)], q_profile[~np.isnan(R_for_q_profile)], 3)
    print(f"Fitted q-profile coefficients (highest degree first): {coeffs_q}")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    valid_indices = ~np.isnan(R_for_q_profile)
    ax2.plot(R_for_q_profile[valid_indices], q_profile[valid_indices], 'b-', label='G-file q Profile')
    ax2.plot(R_for_q_profile[valid_indices], np.polyval(coeffs_q, R_for_q_profile[valid_indices]), 'r--', label='3rd Order Poly Fit')
    ax2.axvline(r_lcfs, color='k', linestyle='--', label='LCFS')
    ax2.set_title(f"Safety Factor (q) Profile: {gfile_basename}")
    ax2.set_xlabel('Major Radius R (m)')
    ax2.set_ylabel('Safety Factor q')
    ax2.legend()
    ax2.grid(True, linestyle='--')
    plt.tight_layout()

    if save_plots:
        output_filename = os.path.join(output_dir, f"q_profile_{gfile_basename}{figure_format}")
        plt.savefig(output_filename)
        print(f"Saved q-profile plot to {output_filename}")
    plt.show()

def extract_limiter_segments(r_vessel, z_vessel):
    """
    Extracts specific geometric segments from a list of vessel wall coordinates.

    This function identifies points belonging to pre-defined regions (e.g., a
    divertor target) and returns them as a list of continuous line segments.

    Args:
        r_vessel (np.array): 1D array of the R coordinates for the vessel wall.
        z_vessel (np.array): 1D array of the Z coordinates for the vessel wall.

    Returns:
        list: A list of tuples, where each tuple contains the (R, Z) coordinates
              for a single continuous segment. E.g., [(r_seg1, z_seg1), (r_seg2, z_seg2), ...].
    """
    # --- Define the geometric regions for the pink segments based on the image ---
    # These values can be adjusted to match your specific machine geometry.
    
    # Condition for the top divertor structure
    is_top_segment = lambda r, z: (z > 0.72) and (r > 2.4) and (r < 2.85)
    
    # Condition for the upper outboard limiter structure
    is_outboard_segment = lambda r, z: (r > 2.85) and (z > 0.4) and (z < 0.65)
    
    # --- Algorithm to find continuous segments ---
    
    # 1. Create a boolean mask: True if a point is in any of the desired regions
    is_pink_mask = np.zeros(len(r_vessel), dtype=bool)
    for i in range(len(r_vessel)):
        r_point, z_point = r_vessel[i], z_vessel[i]
        if is_top_segment(r_point, z_point) or is_outboard_segment(r_point, z_point):
            is_pink_mask[i] = True
            
    # 2. Find the start and end points of continuous blocks of 'True'
    # A 'diff' will be 1 at the start of a block and -1 at the end.
    diff = np.diff(is_pink_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1 # The end index is exclusive in slicing

    # Handle edge cases where a segment is at the start or end of the array
    if is_pink_mask[0]:
        starts = np.insert(starts, 0, 0)
    if is_pink_mask[-1]:
        ends = np.append(ends, len(is_pink_mask))
        
    # 3. Build the list of segments
    segments = []
    print(f"\nFound {len(starts)} pink segment(s).")
    for start, end in zip(starts, ends):
        # Add 1 to the end index to include the last point of the segment
        segment_r = r_vessel[start:end+1]
        segment_z = z_vessel[start:end+1]
        segments.append((segment_r, segment_z))
        
    return segments

# ============================ MAIN EXECUTION BLOCK ============================
if __name__ == "__main__":
    # --- Setup the Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Analyzes and plots a GEQDSK (g-file) magnetic equilibrium.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    
    # --- Define Arguments ---
    parser.add_argument(
        "gfile_path",
        type=str,
        help="Path to the GEQDSK file (g-file) to analyze."
    )
    
    parser.add_argument(
        "--no-miller",
        action="store_false",
        dest="plot_miller",
        help="Disable the Miller geometry overlay on the equilibrium plot."
    )

    parser.add_argument(
        "--shafranov",
        type=float,
        default=0.5,
        help="Shafranov shift parameter ('a_shift') for Miller geometry.\nDefault: 0.25"
    )

    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save_plots",
        help="Do not save the output plots to files."
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./",
        help="Directory to save the output plots.\nDefault: current directory"
    )

    parser.add_argument(
        "--format",
        type=str,
        default=".png",
        help="Format for the saved figures (e.g., .png, .pdf, .svg).\nDefault: .png"
    )

    parser.add_argument(
        "--x-in",
        type=float,
        default=0.10,
        help="Distance (in meters) to offset the inner Miller boundary from LCFS.\nDefault: 0.10 m"
    )

    parser.add_argument(
        "--x-out",
        type=float,
        default=0.05,
        help="Distance (in meters) to offset the outer Miller boundary from LCFS.\nDefault: 0.05 m"
    )

    # --- Parse Arguments and Run ---
    args = parser.parse_args()
    
    # Call the main analysis function with the parsed arguments
    analyze_and_plot(
        gfile_path=args.gfile_path,
        plot_miller=args.plot_miller,
        shafranov_param=args.shafranov,
        save_plots=args.save_plots,
        output_dir=args.outdir,
        figure_format=args.format,
        x_in=args.x_in,
        x_out=args.x_out
    )

    
    
    print("\nAnalysis complete.")