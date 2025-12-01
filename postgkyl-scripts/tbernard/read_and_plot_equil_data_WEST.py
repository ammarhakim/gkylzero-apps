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

def analyze_and_plot(gfile_path, plot_miller, shafranov_param, save_plots, output_dir, figure_format, limiter_segments):
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
    #levels = np.linspace(psi_RZ.min(), psi_RZ.max(), 20)
    vmin=0.32
    vmax=0.38
    levels = np.linspace(vmin, vmax, 20)
    contour = ax1.contour(RR, ZZ, psi_RZ, levels=levels)##psi_RZ.min(), vmax=psi_RZ.max()*.75)
    fig1.colorbar(contour, ax=ax1, label=r'$\psi$ (Wb/rad)')
    
    ax1.plot(gfile_data["rlim"], gfile_data["zlim"], 'k-', linewidth=2.0, label='Vessel Wall')
    ax1.plot(gfile_data["rbdry"], gfile_data["zbdry"], 'w-', linewidth=2.0, label='Experimental LCFS')
    ax1.plot(gfile_data["rmagx"], gfile_data["zmagx"], 'wx', markersize=10, mew=2.5, label='Magnetic Axis')

    if limiter_segments:
        print("\nExtracting limiter segments...")
        segments = extract_limiter_segments(gfile_data["rlim"], gfile_data["zlim"])
        # print segments for implmentation into C input file
        print("\nLimiter Segments (R, Z coordinates):")
        print("segments = [")
        for r_seg, z_seg in segments:
            for r, z in zip(r_seg, z_seg):
                print(f"    ({r:.6f}, {z:.6f}),")
            print("    # New Segment")
        print("]")
        # Plot the segments
        for i, (r_seg, z_seg) in enumerate(segments):
            ax1.plot(r_seg, z_seg, color='magenta', linewidth=3.0, label='Limiter Segment' if i == 0 else "")

    if plot_miller:
        print("\n--- Miller Geometry Analysis ---")
        miller_params = calculate_miller_parameters(gfile_data)
        print(f"  a={miller_params['a']:.4f} m, κ={miller_params['kappa']:.4f}, δ={miller_params['delta']:.4f}")
        # Main Miller LCFS
        R_miller, Z_miller = generate_miller_lcfs(miller_params, shafranov_param)
        ax1.plot(R_miller, Z_miller, 'r--', linewidth=2.5, label=f'Miller LCFS')
        # 10cm inside (a - 0.10)
        R_miller_inner, Z_miller_inner = generate_miller_lcfs(miller_params, shafranov_param, a_override=miller_params['a'] - 0.10)
        ax1.plot(R_miller_inner, Z_miller_inner, color='C1', linestyle=':', linewidth=3, label='inner/outer boundary')
        # 5cm outside (a + 0.05)
        R_miller_outer, Z_miller_outer = generate_miller_lcfs(miller_params, shafranov_param, a_override=miller_params['a'] + 0.05)
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

    R_of_psi_interpolator = CubicSpline(psi_outboard_midplane, R_outboard_midplane, extrapolate=False)
    psi_normalized = np.linspace(0, 1, len(q_profile))
    psi_values = gfile_data["simagx"] + psi_normalized * (gfile_data["sibdry"] - gfile_data["simagx"])
    R_for_q_profile = R_of_psi_interpolator(psi_values)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    valid_indices = ~np.isnan(R_for_q_profile)
    ax2.plot(R_for_q_profile[valid_indices], q_profile[valid_indices], 'b-')
    ax2.set_title(f"Safety Factor (q) Profile: {gfile_basename}")
    ax2.set_xlabel('Major Radius R (m)')
    ax2.set_ylabel('Safety Factor q')
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
    is_top_segment = lambda r, z: (z > 0.72) and (r > 2.45) and (r < 2.85)
    
    # Condition for the upper outboard limiter structure
    is_outboard_segment = lambda r, z: (r > 2.85) and (z > 0.4) and (z < 0.5)
    
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
        "--limiter-segments",
        action="store_true",
        help="Extract and print the limiter segments from the vessel wall."
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
        limiter_segments=args.limiter_segments,
        figure_format=args.format
    )
    
    print("\nAnalysis complete.")