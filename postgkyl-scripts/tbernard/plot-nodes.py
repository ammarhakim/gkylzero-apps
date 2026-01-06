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

def get_U_cyl_comp(prefix, xCart, yCart, U_vals):
    # Load bcart. 
    bcart_data = pg.GData(prefix+"-bcart.gkyl")
    dg = pg.data.GInterpModal(bcart_data, 1, 'ms')
    cdim = 3
    dg_coef = np.sqrt(2)**cdim
    b_x = dg._getRawModal(0)[:,:,0]/dg_coef
    b_y = dg._getRawModal(1)[:,:,0]/dg_coef
    b_z = dg._getRawModal(2)[:,:,0]/dg_coef
    bxFlat = b_x.flatten()
    byFlat = b_y.flatten()
    bzFlat = b_z.flatten()
    orig_shape = U_vals.shape
    U_flat = U_vals.flatten()

    b_R = np.zeros(np.shape(bxFlat))
    b_phi = np.zeros(np.shape(bxFlat))
    b_Z = np.zeros(np.shape(bxFlat))
    for i in range(len(bxFlat)):
        b_R[i], b_phi[i], b_Z[i] = cartesian_to_cylindrical_vector(bxFlat[i], byFlat[i], bzFlat[i], xCart[i], yCart[i])
    
    return (b_R*U_flat).reshape(orig_shape), (b_phi*U_flat).reshape(orig_shape), (b_Z*U_flat).reshape(orig_shape)

# Some physical constants
me = utils.mass_elc
mi = utils.mass_proton*2.014
eV = utils.elem_charge

if len(sys.argv) < 2:
    print("Usage: python mapc2p-elements-fields.py <frame_no>")
    sys.exit("Error: No frame num provided.")

# Parse command-line arguments.
frame = sys.argv[1] # frame is always used as string here.

# Find the file prefix using the helper function from utils.
prefix = utils.find_prefix('-mapc2p.gkyl', '.')

# Create degas-inp dir if it doesn't exist yet.
os.makedirs("degas-inp", exist_ok=True)

# Load nodes data.
data = pg.GData(prefix+"-nodes.gkyl")
node_vals = data.get_values()

print(node_vals)

is1d = True
if len(node_vals.shape) == 2:
    is1d = True

if is1d:
    R = node_vals[:,0]
    Z = node_vals[:,1]
    phi = node_vals[:,2]
else:
    R = node_vals[:,:,0]
    Z = node_vals[:,:,1]
    phi = node_vals[:,:,2]
 
fig = plt.figure(figsize=(5,6))
plt.plot(R,Z,marker=".", color="k", linestyle="-")
plt.scatter(R,Z, marker=".")
if not is1d:
    segs1 = np.stack((R,Z), axis=2)
    segs2 = segs1.transpose(1,0,2)
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))
plt.gca().set_aspect('equal')
plt.grid()
plt.savefig("plot-"+prefix+"-grid.png", dpi=300)
plt.show()

exit()

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

# Get elc and ion BiMaxwellian moms data
moms_data_gkyl = pg.GData(prefix+"-elc_BiMaxwellianMoments_"+frame+".gkyl")
dg = pg.data.GInterpModal(moms_data_gkyl, 1, 'ms')
cdim = 3
dg_coef = np.sqrt(2)**cdim
elcM0 = dg._getRawModal(0)[:,:,0]/dg_coef
elcUpar = dg._getRawModal(1)[:,:,0]/dg_coef
elcTpar = dg._getRawModal(2)[:,:,0]/dg_coef
elcTperp = dg._getRawModal(3)[:,:,0]/dg_coef
elcTemp = 2/3*elcTperp*me/eV + 1/3*elcTpar*me/eV

# Ions
moms_data_gkyl = pg.GData(prefix+"-ion_BiMaxwellianMoments_"+frame+".gkyl")
dg = pg.data.GInterpModal(moms_data_gkyl, 1, 'ms')
ionM0 = dg._getRawModal(0)[:,:,0]/dg_coef
ionUpar = dg._getRawModal(1)[:,:,0]/dg_coef
ionTpar = dg._getRawModal(2)[:,:,0]/dg_coef
ionTperp = dg._getRawModal(3)[:,:,0]/dg_coef
ionTemp = 2/3*ionTperp*mi/eV + 1/3*ionTpar*mi/eV

# Get cylindrical components of Upar
elcU_R, elcU_phi, elcU_Z = get_U_cyl_comp(prefix, xCart, yCart, elcUpar)
ionU_R, ionU_phi, ionU_Z = get_U_cyl_comp(prefix, xCart, yCart, ionUpar)

biMaxMomsStr = ["elcM0", "elcTemp", "elcU_R", "elcU_phi", "elcU_Z", "ionM0", "ionTemp", "ionU_R", "nionU_phi", "ionU_Z"]
biMaxMomsUnits = [r"m$^{-3}$", "eV",  "m/s", "m/s", "m/s",  r"m$^{-3}$", "eV",  "m/s", "m/s", "m/s",]
biMaxMoms = [elcM0, elcTemp, elcU_R, elcU_phi, elcU_Z, ionM0, ionTemp, ionU_R, ionU_phi, ionU_Z]

biMaxMomsStr = ["elcM0", "elcTemp", "elcUpar", "ionM0", "ionTemp", "ionUpar"]
biMaxMomsUnits = [r"m$^{-3}$", "eV",  "m/s", r"m$^{-3}$", "eV", "m/s"]
biMaxMoms = [elcM0, elcTemp, elcUpar, ionM0, ionTemp, ionUpar]

# Plot 3 subplots
fig, axes = plt.subplots(2, len(biMaxMoms)//2, figsize=(15, 8), constrained_layout=True)
axes = axes.flatten()
print(len(axes))

for i, ax in enumerate(axes):
    print(i)
    pcm = ax.pcolormesh(R, Z, biMaxMoms[i], shading='auto')
    ax.set_aspect('auto')
    ax.set_title(f"{biMaxMomsStr[i]}")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")

    # Add individual, appropriately sized colorbars
    cbar = fig.colorbar(pcm, ax=ax, aspect=20)
    cbar.set_label(biMaxMomsUnits[i])

plt.savefig(f"mapc2p-{prefix}-biMaxMoms.png")
plt.show()

#-------------Write output files for DEGAS2 coupling------------#

# Gkyl grid data for mapping.
grid_gkyl, _ = utils.interpolate_field(moms_data_gkyl, 0)
Th_gkyl, X_gkyl = np.meshgrid(grid_gkyl[1][::2], grid_gkyl[0][::2],)
print(np.shape(X_gkyl))
print(np.shape(R))

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

# Write out the field values, element-wise
biMaxMomsStr = ["elcM0", "elcTemp", "elcU_R", "elcU_phi", "elcU_Z", "ionM0", "ionTemp", "ionU_R", "nionU_phi", "ionU_Z"]
biMaxMomsUnits = [r"m$^{-3}$", "eV",  "m/s", "m/s", "m/s",  r"m$^{-3}$", "eV",  "m/s", "m/s", "m/s",]
biMaxMoms = [elcM0, elcTemp, elcU_R, elcU_phi, elcU_Z, ionM0, ionTemp, ionU_R, ionU_phi, ionU_Z]

# Open single file for writing
with open("degas-inp/species_moms_values_vs_element.txt", "w") as f:
    # Write header
    header = "ElementID " + " ".join(f"{name} [{unit}]" for name, unit in zip(biMaxMomsStr, biMaxMomsUnits)) + "\n"
    f.write("# " + header)

    # Loop through elements (assumes uniform grid for all fields)
    for i in range(Nx - 1):
        for j in range(Nz - 1):
            elem_id = i * (Nz - 1) + j
            values = [field[i, j] for field in biMaxMoms]
            line = f"{elem_id:<9d} " + " ".join(f"{val:15.8e}" for val in values) + "\n"
            f.write(line)

print("Wrote degas-inp/species_moms_values_vs_element.txt")