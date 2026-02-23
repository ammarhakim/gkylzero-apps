import numpy as np
import matplotlib.path as mpl_path
import sys
import os
import re
import matplotlib.pyplot as plt

# --- Your provided RZ_func ---
def RZ_func(s):
    """
    Python version of divertor_plate_func_out
    s : float or numpy array in [0, 1]
    returns (R, Z)
    """
    s = np.asarray(s)
    RZ_lo = np.array([2.78, 0.82])
    RZ_up = np.array([3.155, 0.55])
    R = (1-s)*RZ_lo[0] + s*RZ_up[0]
    Z = (1-s)*RZ_lo[1] + s*RZ_up[1]
    return R, Z

class RobustGEQDSK:
    def __init__(self, filename):
        self.filename = filename
        self.data = {}
        self.scalars_list = []
        self.read()

    def read(self):
        """
        Reads and parses a standard 'geqdsk' (eqdsk) file.
        This version is robust to common formatting variations, including
        fused numbers (e.g., '1.23-4.56') and 'D' for scientific notation.
        """
        with open(self.filename, 'r') as f:
            raw_text = f.read()

        # 1. FIX FUSED NUMBERS (e.g. "1.23-4.56" -> "1.23 -4.56")
        # And replace 'D' with 'E' for easier float parsing
        cleaned_text = re.sub(r'(\d)-(\d)', r'\1 -\2', raw_text.replace('D', 'E'))
        
        lines = cleaned_text.splitlines()
        
        # 2. Extract description (first 48 characters of the first line)
        self.data['description'] = lines[0][:48]
        
        # 3. Parse header integers (idum, nw, nh)
        # These typically follow the description on the first line, or are on the second.
        # Let's tokenize the relevant part of the line.
        
        # Find the integers part of the first line
        header_int_str_candidates = [lines[0][48:].strip()]
        if len(lines) > 1:
            header_int_str_candidates.append(lines[1].strip())

        header_ints_found = False
        for s in header_int_str_candidates:
            parts = s.split()
            if len(parts) >= 3:
                try:
                    self.data['idum'] = int(parts[0])
                    self.data['nw'] = int(parts[1])
                    self.data['nh'] = int(parts[2])
                    header_ints_found = True
                    break
                except ValueError:
                    continue # Not valid integers, try next candidate
        
        if not header_ints_found:
            raise ValueError("Could not find idum, nw, nh in the header lines.")

        nw, nh = self.data['nw'], self.data['nh']

        # Determine where the actual scalar values (the 20 floats) begin in the cleaned_text
        # If idum, nw, nh were on lines[0], scalars start from lines[1] content.
        # If idum, nw, nh were on lines[1], scalars start from lines[2] content.
        
        # We need to re-tokenise the rest of the body from the appropriate line onwards
        body_start_idx = 1 # Default assumption (idum, nw, nh on line 0)
        if not lines[0][48:].strip().split() or len(lines[0][48:].strip().split()) < 3:
            body_start_idx = 2 # If idum, nw, nh were on line 1, scalars start on line 2

        body_text_for_scalars_and_arrays = "\n".join(lines[body_start_idx:])
        iter_tokens = iter(body_text_for_scalars_and_arrays.split())

        # 4. READ THE 20 SCALARS (PRESERVE ALL OF THEM)
        self.scalars_list = [float(next(iter_tokens)) for _ in range(20)]
        
        # Map important scalars to named variables
        self.data['rdim']   = self.scalars_list[0]
        self.data['zdim']   = self.scalars_list[1]
        self.data['rcentr'] = self.scalars_list[2] # R at current center for BC_Z
        self.data['rleft']  = self.scalars_list[3]
        self.data['zmid']   = self.scalars_list[4]
        self.data['simag']  = self.scalars_list[7] # Flux at axis
        self.data['sibdry'] = self.scalars_list[8] # Flux at boundary (separatrix)
        self.data['rci']    = self.scalars_list[9] # R of magnetic axis
        self.data['zci']    = self.scalars_list[10] # Z of magnetic axis

        # 5. READ ARRAYS
        self.data['fpol']   = np.array([float(next(iter_tokens)) for _ in range(nw)])
        self.data['pres']   = np.array([float(next(iter_tokens)) for _ in range(nw)])
        self.data['ffprim'] = np.array([float(next(iter_tokens)) for _ in range(nw)])
        self.data['pprim']  = np.array([float(next(iter_tokens)) for _ in range(nw)])

        # PSI Grid (Row-major: NH rows, NW cols)
        self.data['psi'] = np.array([float(next(iter_tokens)) for _ in range(nw * nh)]).reshape((nh, nw))
        self.data['qpsi'] = np.array([float(next(iter_tokens)) for _ in range(nw)])

        # 6. READ BOUNDARY/LIMITER COUNTS
        self.data['nbdry'] = int(next(iter_tokens))
        self.data['nlim']  = int(next(iter_tokens))

        # 7. READ COORDINATES
        self.data['rbdry'] = np.zeros(self.data['nbdry'])
        self.data['zbdry'] = np.zeros(self.data['nbdry'])
        for i in range(self.data['nbdry']):
            self.data['rbdry'][i] = float(next(iter_tokens))
            self.data['zbdry'][i] = float(next(iter_tokens))

        self.data['rlim'] = np.zeros(self.data['nlim'])
        self.data['zlim'] = np.zeros(self.data['nlim'])
        for i in range(self.data['nlim']):
            self.data['rlim'][i] = float(next(iter_tokens))
            self.data['zlim'][i] = float(next(iter_tokens))
            
        print(f"Loaded {self.filename} (Grid: {nw}x{nh})")

    def write(self, output_filename):
        def write_arr(f, arr):
            flat = arr.flatten()
            for i, val in enumerate(flat):
                f.write(f"{val:16.9E}") 
                if (i + 1) % 5 == 0: 
                    f.write("\n")
            if len(flat) % 5 != 0: 
                f.write("\n")

        with open(output_filename, 'w') as f:
            f.write(f"{self.data['description']:48}{self.data['idum']:4d}{self.data['nw']:4d}{self.data['nh']:4d}\n")
            
            for i, val in enumerate(self.scalars_list):
                f.write(f"{val:16.9E}")
                if (i+1)%5==0: f.write("\n")
            if len(self.scalars_list)%5!=0: f.write("\n")

            write_arr(f, self.data['fpol'])
            write_arr(f, self.data['pres'])
            write_arr(f, self.data['ffprim'])
            write_arr(f, self.data['pprim'])
            write_arr(f, self.data['psi'])
            write_arr(f, self.data['qpsi'])
            
            f.write(f"{self.data['nbdry']:5d}{self.data['nlim']:5d}\n")
            
            bdry = np.column_stack((self.data['rbdry'], self.data['zbdry']))
            write_arr(f, bdry)
            lim = np.column_stack((self.data['rlim'], self.data['zlim']))
            write_arr(f, lim)
            
        print(f"Saved to {output_filename}")

    def get_grids(self):
        """Helper to get R and Z grids based on GEQDSK conventions."""
        nw, nh = self.data['nw'], self.data['nh']
        r_grid = np.linspace(self.data['rleft'], self.data['rleft'] + self.data['rdim'], nw)
        z_grid = np.linspace(self.data['zmid'] - self.data['zdim']/2, self.data['zmid'] + self.data['zdim']/2, nh)
        return r_grid, z_grid

    def apply_rz_func_mask(self, rz_func_boundary_points, set_to_zero=False):
        """
        Sets psi values to NaN (or 0) for points *inside* the polygon defined by rz_func_boundary_points.
        This function assumes the input polygon defines the region to be masked.

        Args:
            rz_func_boundary_points (np.ndarray): A 2D array of (R, Z) coordinates
                                                  defining the boundary polygon of the region to MASK.
            set_to_zero (bool): If True, set masked regions to 0.0 instead of NaN.
        """
        psi_2d_array = self.data['psi']
        r_grid, z_grid = self.get_grids()
        
        # Ensure the polygon is closed for mpl_path.Path
        polygon_points_closed = np.array(rz_func_boundary_points)
        if not np.allclose(polygon_points_closed[0], polygon_points_closed[-1]):
            polygon_points_closed = np.vstack((polygon_points_closed, polygon_points_closed[0]))

        polygon_path = mpl_path.Path(polygon_points_closed)

        R_mesh, Z_mesh = np.meshgrid(r_grid, z_grid)
        grid_points = np.vstack((R_mesh.flatten(), Z_mesh.flatten())).T

        # Here, is_inside_mask_region means "is_inside the region to be masked"
        is_inside_mask_region = polygon_path.contains_points(grid_points).reshape(psi_2d_array.shape)
        
        if set_to_zero:
            self.data['psi'][is_inside_mask_region] = 0.0
            print("Regions defined by RZ_func boundary set to 0.")
        else:
            self.data['psi'][is_inside_mask_region] = np.nan
            print("Regions defined by RZ_func boundary set to NaN.")

# --- Main Script Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process GEQDSK files to mask regions based on an RZ_func boundary.")
    parser.add_argument("input", help="Input GEQDSK file path")
    parser.add_argument("output", help="Output GEQDSK filename")
    parser.add_argument("--set-to-zero", action='store_true',
                        help="Set masked regions to 0.0 instead of NaN.")
    
    args = parser.parse_args()

    try:
        g = RobustGEQDSK(args.input)

        original_psi = np.copy(g.data['psi']) # Store original for plotting comparison

        # 1. Generate points for your RZ_func line segment
        s_values = np.linspace(0, 1, 100)
        boundary_R_line, boundary_Z_line = RZ_func(s_values)

        # 2. Construct the polygon for the *region you want to MASK*.
        #    We want to mask points (R,Z) where R is to the right of the RZ_func line,
        #    and Z is within the Z-range covered by the RZ_func line.
        
        # Get grid R and Z limits
        r_grid_min = g.get_grids()[0].min()
        r_grid_max = g.get_grids()[0].max()
        z_grid_min_full = g.get_grids()[1].min()
        z_grid_max_full = g.get_grids()[1].max()
        
        # Z-range of the RZ_func line
        min_rz_z = np.min(boundary_Z_line)
        max_rz_z = np.max(boundary_Z_line)

        # Polygon for the region to MASK:
        # This will create a rectangle that is bounded by the RZ_func line on the left,
        # extends to the r_grid_max on the right, and is bounded by the Z-range of RZ_func.
        
        polygon_points_to_mask = np.array([
            [boundary_R_line[0], boundary_Z_line[0]],   # RZ_lo (bottom-left of the RZ_func line)
            [boundary_R_line[-1], boundary_Z_line[-1]], # RZ_up (top-right of the RZ_func line)
            [r_grid_max,          boundary_Z_line[-1]], # Top-right corner of the *masking* rectangle
            [r_grid_max,          boundary_Z_line[0]],   # Bottom-right corner of the *masking* rectangle
        ])
        # The closing point for mpl_path.Path is handled within apply_rz_func_mask.

        # Apply the mask.
        g.apply_rz_func_mask(polygon_points_to_mask, set_to_zero=args.set_to_zero)

        # Write the modified GEQDSK data to a new file
        g.write(args.output)

        # --- Visualization ---
        r_grid, z_grid = g.get_grids()
        R_mag_axis = g.data['rci']
        Z_mag_axis = g.data['zci']

        plt.figure(figsize=(12, 6))

        # Original Psi Plot
        plt.subplot(1, 2, 1)
        levels = np.linspace(np.nanmin(original_psi), np.nanmax(original_psi), 50)
        plt.contourf(r_grid, z_grid, original_psi, levels=levels, cmap='viridis', extend='both')
        plt.colorbar(label='Original Psi')
        plt.contour(r_grid, z_grid, original_psi, levels=[g.data['sibdry']], colors='w', linestyles='--', linewidths=2, label='Separatrix')
        plt.scatter(R_mag_axis, Z_mag_axis, color='red', marker='x', s=100, label='Magnetic Axis')
        plt.plot(g.data['rlim'], g.data['zlim'], 'k-', linewidth=1, label='Limiter')
        plt.plot(g.data['rbdry'], g.data['zbdry'], 'm--', linewidth=1, label='Plasma Boundary')
        plt.plot(boundary_R_line, boundary_Z_line, 'r--', linewidth=2, label='RZ_func Boundary')
        plt.title('Original Psi Profile (from eqdsk)')
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.legend()
        plt.grid(True)


        # Modified Psi Plot
        plt.subplot(1, 2, 2)
        modified_psi_non_nan = g.data['psi'][~np.isnan(g.data['psi'])]
        if len(modified_psi_non_nan) > 0:
            levels_mod = np.linspace(np.nanmin(modified_psi_non_nan), np.nanmax(modified_psi_non_nan), 50)
        else: # If everything is NaN, just use original psi levels for consistency
            levels_mod = levels
        
        plt.contourf(r_grid, z_grid, g.data['psi'], levels=levels_mod, cmap='viridis', extend='both')
        plt.colorbar(label='Modified Psi (Outside Boundary Masked)')
        
        # Re-plot separatrix and other features from original data for context
        plt.contour(r_grid, z_grid, original_psi, levels=[g.data['sibdry']], colors='w', linestyles='--', linewidths=2)
        plt.scatter(R_mag_axis, Z_mag_axis, color='red', marker='x', s=100)
        plt.plot(g.data['rlim'], g.data['zlim'], 'k-', linewidth=1)
        plt.plot(g.data['rbdry'], g.data['zbdry'], 'm--', linewidth=1)
        plt.plot(boundary_R_line, boundary_Z_line, 'r--', linewidth=2)
        # Plot the *masking* polygon to show what area was identified for removal
        plt.plot(polygon_points_to_mask[:, 0], polygon_points_to_mask[:, 1], 'k:', alpha=0.5, label='Actual Masking Polygon')


        plt.title('Modified Psi Profile (RZ_func masked)')
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # 1D Z-slice visualization
        plt.figure(figsize=(10, 4))
        # Choose a Z-slice that is within the RZ_func boundary's Z-range
        z_slice_val = (min_rz_z + max_rz_z) / 2
        z_slice_idx = np.argmin(np.abs(z_grid - z_slice_val))

        plt.plot(r_grid, original_psi[z_slice_idx, :], label='Original Psi at Z_mid')
        plt.plot(r_grid, g.data['psi'][z_slice_idx, :], 'o--', markersize=3, label='Modified Psi at Z_mid')
        
        # Calculate R on the RZ_func line for this Z-slice
        s_for_slice_Z = (z_grid[z_slice_idx] - RZ_func(0)[1]) / (RZ_func(1)[1] - RZ_func(0)[1])
        R_at_slice_Z = RZ_func(s_for_slice_Z)[0]
        
        plt.axvline(R_at_slice_Z, color='red', linestyle='--', label='RZ_func Boundary R at Z_mid')
        plt.title(f'Psi Profile at Z = {z_grid[z_slice_idx]:.3f} m')
        plt.xlabel('R')
        plt.ylabel('Psi')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{args.input}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")