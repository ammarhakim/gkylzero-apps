import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import h5py
import argparse
from freeqdsk import geqdsk

def zero_psi_outside_vessel(gfile_data):
    """
    Sets the psi values in a gfile data structure to zero for all points
    that lie outside the vacuum vessel wall.
    """
    print("Creating mask for points outside the vessel wall...")
    psi_RZ = gfile_data["psi"]
    r_vessel, z_vessel = gfile_data["rlim"], gfile_data["zlim"]
    nx, ny = gfile_data["nx"], gfile_data["ny"]
    R_grid = np.linspace(gfile_data["rleft"], gfile_data["rleft"] + gfile_data["rdim"], nx)
    Z_grid = np.linspace(gfile_data["zmid"] - gfile_data["zdim"]/2, gfile_data["zmid"] + gfile_data["zdim"]/2, ny)
    RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing='ij')
    grid_points = np.vstack([RR.ravel(), ZZ.ravel()]).T
    vessel_path = Path(np.vstack([r_vessel, z_vessel]).T)
    is_inside_mask_2d = vessel_path.contains_points(grid_points).reshape((nx, ny))
    psi_RZ_new = np.copy(psi_RZ)
    psi_RZ_new[~is_inside_mask_2d] = 0.0
    print(f"Successfully created mask. Zeroed out {np.sum(~is_inside_mask_2d)} grid points.")
    return psi_RZ_new

def plot_verification(gfile_data_original, psi_data_new):
    """
    Generates a plot to visually compare the old and new psi fields.
    Takes the original gfile object and the new psi numpy array.
    """
    print("Generating verification plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)
    
    nx, ny = gfile_data_original["nx"], gfile_data_original["ny"]
    R_grid = np.linspace(gfile_data_original["rleft"], gfile_data_original["rleft"] + gfile_data_original["rdim"], nx)
    Z_grid = np.linspace(gfile_data_original["zmid"] - gfile_data_original["zdim"]/2, gfile_data_original["zdim"]/2, ny)
    RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing='ij')
    
    # Plot original psi from the original object
    ax = axes[0]
    ax.contourf(RR, ZZ, gfile_data_original['psi'], levels=50)
    ax.plot(gfile_data_original['rlim'], gfile_data_original['zlim'], 'k-', lw=2)
    ax.set_title("Original Psi Field")
    ax.set_xlabel("R (m)"), ax.set_ylabel("Z (m)")
    ax.set_aspect('equal')

    # Plot the new psi data array
    ax = axes[1]
    ax.contourf(RR, ZZ, psi_data_new, levels=50)
    ax.plot(gfile_data_original['rlim'], gfile_data_original['zlim'], 'k-', lw=2)
    ax.set_title("Modified Psi Field (Zeroed Outside)")
    ax.set_xlabel("R (m)")
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("psi_zeroing_verification.png")
    print("Saved verification plot to 'psi_zeroing_verification.png'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Zeros out the psi field in a GEQDSK file for all points outside the vessel wall.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the original GEQDSK file.")
    parser.add_argument("output_file", help="Path where the modified GEQDSK file will be saved.")
    parser.add_argument("--plot", action="store_true", help="Generate a plot to visually verify the result.")
    args = parser.parse_args()

    try:
        print(f"Reading input gfile: {args.input_file}")
        with open(args.input_file, "r") as f:
            gfile_obj = geqdsk.read(f)
            
        # --- FIX IS HERE: REVISED ORDER OF OPERATIONS ---
        
        # 1. Calculate the new psi array from the original object.
        psi_RZ_new = zero_psi_outside_vessel(gfile_obj)
        
        # 2. (Optional) Plot the comparison BEFORE modifying the object.
        if args.plot:
            # Pass the original object and the new numpy array to the plot function.
            plot_verification(gfile_obj, psi_RZ_new)
            
        # 3. NOW, modify the object in-place.
        gfile_obj['psi'] = psi_RZ_new
        
        # 4. Write the now-modified object to the output file.
        print(f"Writing modified gfile to: {args.output_file}")
        with open(args.output_file, "w") as f:
            geqdsk.write(gfile_obj, f)
        # --- END FIX ---
            
        print("\nOperation completed successfully.")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()