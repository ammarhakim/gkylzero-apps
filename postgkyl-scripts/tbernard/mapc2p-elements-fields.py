import postgkyl as pg
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import utils

# Function to convert to bcart to bcyl
def cartesian_to_cylindrical_vector(vx, vy, vz, x, y):
    R = np.sqrt(x**2 + y**2)
    if R == 0:
        raise ValueError("Cannot convert at R=0 (singularity in phi direction).")

    # Basis vectors
    e_R = np.array([x, y, 0.0]) / R
    e_phi = np.array([-y, x, 0.0]) / R
    e_Z = np.array([0.0, 0.0, 1.0])

    v_cart = np.array([vx, vy, vz])
    v_R = np.dot(v_cart, e_R)
    v_phi = np.dot(v_cart, e_phi)
    v_Z = vz  # same as dot(v_cart, e_Z)

    return v_R, v_phi, v_Z

def calc_b_cyl(prefix, arr_slice):
    # Load bcart. 
    bcart_data = pg.data.GData(prefix+"-bcart.gkyl", mapc2p_name=prefix+"-mapc2p.gkyl")
    _, b_x = utils.interpolate_field(bcart_data, 0)
    _, b_y = utils.interpolate_field(bcart_data, 1)
    _, b_z = utils.interpolate_field(bcart_data, 2)
    bxFlat = b_x[arr_slice,0].flatten()
    byFlat = b_y[arr_slice,0].flatten()
    bzFlat = b_z[arr_slice,0].flatten()

    b_R = np.zeros(np.shape(bxFlat))
    b_phi = np.zeros(np.shape(bxFlat))
    b_Z = np.zeros(np.shape(bxFlat))
    for i in range(len(bxFlat)):
        b_R[i], b_phi[i], b_Z[i] = cartesian_to_cylindrical_vector(bxFlat[i], byFlat[i], bzFlat[i], xCart[i], yCart[i])
    return b_R, b_phi, b_Z

# Some physical constants
me = utils.mass_elc
mi = utils.mass_proton*2.014
eV = utils.elem_charge

if len(sys.argv) < 3:
    print("Usage: python plot-mapc2p-grid.py <species_name> <frame_no>")
    sys.exit("Error: No species name / frame num provided.")

# Parse command-line arguments.
sname = sys.argv[1]
frame = sys.argv[2] # frame is always used as string here.

if sname == "elc": 
    mass = me
elif sname == "ion":
    mass = mi
else:
    sys.exit("Error: Species name must be ion/elc.")

# Find the file prefix using the helper function from utils.
prefix = utils.find_prefix('-mapc2p.gkyl', '.')

# Create degas-inp dir if it doesn't exist yet.
os.makedirs("degas-inp", exist_ok=True)

# Load data with mapc2p information.
fld_data = pg.data.GData(prefix+"-"+sname+"_BiMaxwellianMoments_"+frame+".gkyl", mapc2p_name=prefix+"-mapc2p.gkyl")
dg = pg.GInterpModal(fld_data)
dg.interpolate(overwrite=True)

arr_slice = (slice(None, None, 2), 0, slice(None, None, 2))
arr_slice_fld = (slice(None, None, 2), 0, slice(None, None, 2), 0)

# Step 0: Get grid and (x,y) coordinates
grid = fld_data.get_grid()
print(np.shape(grid[0]))
X = grid[0][arr_slice]
Y = grid[1][arr_slice]
Z = grid[2][arr_slice]
xCart = X.flatten()
yCart = Y.flatten()
zCart = Z.flatten()
R = np.sqrt(np.square(X) + np.square(Y))
fig = plt.figure(figsize=(5,6))
plt.plot(R,Z,marker=".", color="k", linestyle="none")
plt.scatter(R,Z, marker=".")
segs1 = np.stack((R,Z), axis=2)
segs2 = segs1.transpose(1,0,2)
plt.gca().add_collection(LineCollection(segs1))
plt.gca().add_collection(LineCollection(segs2))
plt.gca().set_aspect('equal')
plt.grid()
plt.savefig("plot-"+prefix+"-grid.png", dpi=300)
plt.show()

# Step 1: Build list of nodes
nodes = np.column_stack((R.flatten(), Z.flatten()))  # (N, 2)

# Step 2: Build list of quadrilateral elements (each referencing 4 node indices)
Nx, Nz = R.shape
elements = []
for i in range(Nx - 1):
    for j in range(Nz - 1):
        n0 = i * Nz + j
        n1 = i * Nz + (j + 1)
        n2 = (i + 1) * Nz + (j + 1)
        n3 = (i + 1) * Nz + j
        elements.append((n0, n1, n2, n3))

# Optional: print counts
print(f"Total nodes: {len(nodes)}")
print(f"Total elements: {len(elements)}")

# Define mapping between gkyl grid and RZ grid and interpolate the BiMaxwellian moms
moms_data_gkyl = pg.GData(prefix+"-"+sname+"_BiMaxwellianMoments_"+frame+".gkyl")
grid_gkyl, m0 = utils.interpolate_field(moms_data_gkyl, 0)
Th_gkyl, X_gkyl = np.meshgrid(grid_gkyl[2][::2], grid_gkyl[0][::2],)
print(np.shape(X_gkyl))
print(np.shape(R))

# Interp, slice and store BiMax Moments
_, upar = utils.interpolate_field(moms_data_gkyl, 1)
_, Tpar = utils.interpolate_field(moms_data_gkyl, 2)
_, Tperp = utils.interpolate_field(moms_data_gkyl, 3)
Temp = 2/3*Tperp*mass/eV + 1/3*Tpar*mass/eV
biMaxMomsStr = ["M0", "Upar", "Temp"]
biMaxMomsUnits = [r"m$^{-3}", "m/s", "eV"]
biMaxMoms = []
biMaxMoms.append(m0[arr_slice_fld])
biMaxMoms.append(upar[arr_slice_fld])
biMaxMoms.append(Temp[arr_slice_fld])

# Write nodes to ASCII file
with open("degas-inp/nodes-"+prefix+".txt", "w") as f:
    f.write("# NodeID    R           Z\n")
    for i, (R_val, Z_val) in enumerate(nodes):
        f.write(f"{i} {R_val:.10e} {Z_val:.10e}\n")

# Write elements to ASCII file
with open("degas-inp/elments-"+prefix+".txt", "w") as f:
    f.write("# ElementID   Node1   Node2   Node3   Node4\n")
    for i, (n0, n1, n2, n3) in enumerate(elements):
        f.write(f"{i} {n0} {n1} {n2} {n3}\n")

# Define mapping between RZ and gkyl coordinates.
# Assume R, Z, X, Y are 2D arrays of the same shape from meshgrid or GData.
R_flat = R.flatten()
Z_flat = Z.flatten()
X_flat = X_gkyl.flatten()
Th_flat = Th_gkyl.flatten()
with open("degas-inp/map-nodes-RZ-to-gkyl-"+prefix + ".txt", "w") as f:
    f.write("# NodeID    R              Z              X              Theta\n")
    for i, (r_val, z_val, x_val, y_val) in enumerate(zip(R_flat, Z_flat, X_flat, Th_flat)):
        f.write(f"{i:<8d} {r_val:.10e} {z_val:.10e} {x_val:.10e} {y_val:.10e}\n")

# For plotting fields
values = fld_data.get_values()
vals2d = values[arr_slice_fld]
fig = plt.figure(figsize=(5,6))
ax = plt.pcolormesh(R, Z, vals2d)
ax.axes.set_aspect('equal')
plt.title(sname+" m0")
plt.colorbar()
plt.savefig("mapc2p-"+prefix+"-"+sname+"m0.png")
plt.show()

# Write out the field values, element-wise
element_values = []
for i in range(Nx - 1):
    for j in range(Nz - 1):
        element_values.append(vals2d[i, j])

with open("degas-inp/element_"+field_name+"_values.txt", "w") as f:
    f.write("# ElementID   FieldValue\n")
    for k, val in enumerate(element_values):
        f.write(f"{k:<10d} {val:.10e}\n")