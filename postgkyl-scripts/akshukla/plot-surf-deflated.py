# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import postgkyl as pg
import os
import math
import sys

def fix_gridvals(grid):
    """Output file grids have cell-edge coordinates by default, but the values are at cell centers.
    This function will return the cell-center coordinate grid.
    Usage: cell_center_grid = fix_gridvals(cell_edge_grid)
    """
    grid = np.array(grid).squeeze()
    grid = (grid[0:-1] + grid[1:])/2
    return grid

def get_interp_grid(grid):
    num = (len(grid)-1)*2
    diff = np.diff(grid)[0]/2.0
    grid2 = np.zeros(num)
    coord = grid[0]-diff/2.0
    for i in range(num):
        coord+=diff
        grid2[i] = coord
    return grid2


def basis(x):
    return 1/np.sqrt(2), np.sqrt(3)*x/np.sqrt(2)

bnum = int(sys.argv[2])
sim_name = "simdata/step8/step8_b%d"%bnum
quant=sys.argv[1]
data = pg.GData(sim_name+'-%s_dir1.gkyl'%(quant))
grid = data.get_grid()
coeffs = data.get_values()
component = 0
num_basis = 2
coeffs = coeffs[:,:,component*num_basis:(component+1)*num_basis]

xval = -1/2
q1 = np.sum(coeffs*basis(xval),axis=-1)
xval = 1/2
q2 = np.sum(coeffs*basis(xval),axis=-1)

q = np.zeros((q1.shape[0]*2, q1.shape[1]))
for ix in range(q.shape[0]):
    q[ix] = q1[ix//2] if ix % 2 == 0 else q2[ix//2]

x = get_interp_grid(grid[0])
z = grid[1][:-1]

plt.figure()
plt.pcolor(x,z,q.T, cmap="inferno")
plt.colorbar()
plt.title("b_2 at z surfaces")
plt.xlabel("x")
plt.ylabel("z")

data = pg.GData(sim_name+'-%s_dir0.gkyl'%(quant))
grid = data.get_grid()
coeffs = data.get_values()
num_basis = 2
coeffs = coeffs[:,:,component*num_basis:(component+1)*num_basis]

zval = -1/2
q1 = np.sum(coeffs*basis(zval),axis=-1)
zval = 1/2
q2 = np.sum(coeffs*basis(zval),axis=-1)

q = np.zeros((q1.shape[0], q1.shape[1]*2))
for ix in range(q.shape[1]):
    q[:,ix] = q1[:,ix//2] if ix % 2 == 0 else q2[:,ix//2]

x = grid[0][:-1]
z = get_interp_grid(grid[1])

plt.figure()
plt.pcolor(x,z,q.T, cmap="inferno")
plt.colorbar()
plt.title("b_2 at x surfaces")
plt.xlabel("x")
plt.ylabel("z")








