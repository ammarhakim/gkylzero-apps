# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import postgkyl as pg
import os
import math
import sys

import scipy.interpolate 
from scipy.interpolate import RegularGridInterpolator, interp1d
import scipy.integrate as sci


def fix_gridvals(grid):
    """Output file grids have cell-edge coordinates by default, but the values are at cell centers.
    This function will return the cell-center coordinate grid.
    Usage: cell_center_grid = fix_gridvals(cell_edge_grid)
    """
    grid = np.array(grid).squeeze()
    grid = (grid[0:-1] + grid[1:])/2
    return grid



# Universal params
mp = 1.67262192e-27
me = 9.1093837e-31
eV = 1.602e-19
mu_0 = 12.56637061435917295385057353311801153679e-7
eps0=8.854187817620389850536563031710750260608e-12

masses = {}
masses["elc"] = me
#masses["ion"] = 2.014*mp
masses["ion"] = mp


charges = {}
charges["elc"] = -eV 
charges["ion"] = eV


# Set up names for data loading
sim_dir = "./"
base_name = sim_dir+'h11'
bmin, bmax = 0, 12
sim_names = ['%s_b%d'%(base_name,i) for i in range(bmin,bmax)]

frame = int(np.genfromtxt("gkeyll_text_output/new_data_flag"))

Rlist = []
Zlist = []
Rilist = []
Zilist = []
mom_data_list = []
raw_mom_data_list = []
source_mom_data_list = []
for isim, sim_name in enumerate(sim_names):
    mom_data = {}
    #Load geometry
    bdata = pg.GData('%s-bmag.gkyl'%sim_name)
    grid,val = pg.data.GInterpModal(bdata,poly_order=1,basis_type='ms').interpolate(0)
    mom_data["B"] = val.squeeze()
    
    ci = 0
    for c1 in ["x","y","z"]:
        for c2 in ["x","y","z"]:
            if( c1=="y" and c2 =="x"):
                continue
            if( c1=="z" and c2 !="z"):
                continue
            gdata = pg.GData('%s-g_ij.gkyl'%sim_name)
            grid,val = pg.data.GInterpModal(gdata,poly_order=1,basis_type='ms').interpolate(ci)
            mom_data["g_%s%s"%(c1,c2)] = val.squeeze()
            ci+=1
    ci = 0
    for c1 in ["x","y","z"]:
        for c2 in ["x","y","z"]:
            if( c1=="y" and c2 =="x"):
                continue
            if( c1=="z" and c2 !="z"):
                continue
            gdata = pg.GData('%s-gij.gkyl'%sim_name)
            grid,val = pg.data.GInterpModal(gdata,poly_order=1,basis_type='ms').interpolate(ci)
            mom_data["g%s%s"%(c1,c2)] = val.squeeze()
            ci+=1
    
    jdata = pg.GData('%s-jacobgeo.gkyl'%sim_name)
    grid,val = pg.data.GInterpModal(jdata,poly_order=1,basis_type='ms').interpolate(0)
    mom_data["J"] = val.squeeze()
    geo_fac = 1/mom_data["J"]/mom_data["B"]

    #Load grid data
    node_data = pg.GData(sim_name+"-nodes.gkyl")
    vals = node_data.get_values()
    R = vals[:,:,0]
    Z = vals[:,:,1]
    PHI = vals[:,:,2]
    mom_data["R"] = R
    mom_data["Z"] = Z

    #Get interpolated physical coords
    temp_nodal_grid = node_data.get_grid()
    nodal_grid = []
    for d in range(0,len(temp_nodal_grid)):
        nodal_grid.append( np.linspace(temp_nodal_grid[d][0], temp_nodal_grid[d][-1], len(temp_nodal_grid[d])-1) )

    Rinterpolator = RegularGridInterpolator((nodal_grid[0], nodal_grid[1]), R)
    Zinterpolator = RegularGridInterpolator((nodal_grid[0], nodal_grid[1]), Z)
    g0, g1 = np.meshgrid(grid[0], grid[1])
    R = Rinterpolator((g0,g1))
    Z = Zinterpolator((g0, g1))

    g0i, g1i = np.meshgrid(fix_gridvals(grid[0]), fix_gridvals(grid[1]))
    Ri = Rinterpolator((g0i,g1i))
    Zi = Zinterpolator((g0i, g1i))

    Zavg = (Z.T[:,1:] + Z.T[:,:-1])/2
    Ravg = (R.T[1:,:] + R.T[:-1,:])/2
    mom_data["Zavg"] = Zavg
    mom_data["Ravg"] = Ravg

    Zperpavg = (Z.T[1:,:] + Z.T[:-1,:])/2
    mom_data["Zperpavg"] = Zperpavg

    Rlist.append(R.T)
    Zlist.append(Z.T)

    Rilist.append(Ri.T)
    Zilist.append(Zi.T)

    #Get Plate angle information
    if isim == 3 : 
        plate_data = np.genfromtxt("./plate_data/cornerplate/osol.txt", delimiter = ",")
    if isim == 4 : 
        plate_data = np.genfromtxt("./plate_data/cornerplate/opf.txt", delimiter = ",")
    if isim == 8 : 
        plate_data = np.genfromtxt("./plate_data/cornerplate/isol.txt", delimiter = ",")
    if isim == 9 : 
        plate_data = np.genfromtxt("./plate_data/cornerplate/ipf.txt", delimiter = ",")

    #Load moment data
    for species in ["elc", "ion"]:
        for mom in ["M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp"]:
            mdata = pg.GData('%s-%s_%s_%d.gkyl'%(sim_name, species,mom,frame))
            grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
            x = fix_gridvals(grid[0])
            z = fix_gridvals(grid[1])
            val = val.squeeze()
            mom_data[species+mom] = val
        # Set interpolated moment data
        mom_data[species+"Temp"] =  (masses[species]/3) * (mom_data[species+"M2"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
        mom_data[species+"Tpar"] =  (masses[species]) * (mom_data[species+"M2par"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
        mom_data[species+"Tperp"] =  (masses[species]/2) * (mom_data[species+"M2perp"])/mom_data[species+"M0"] / eV
        mom_data[species+"Q"] =  masses[species]/2 * (mom_data[species+"M3par"] + mom_data[species+"M3perp"])
        mom_data[species+"Upar"] =  mom_data[species+"M1"]/mom_data[species+"M0"]
    
    mom_data["Qtot"] =  mom_data["elcQ"]+mom_data["ionQ"]
    
    # Calculate sound speed for ion species
    for species in ["ion" ]:
        mom_data[species+"cs"] = np.sqrt( (mom_data["elcTemp"]*eV + mom_data[species+"Temp"]*eV) / masses[species])
    
    for species in ["ion", "elc" ]:
        mom_data[species+"normUpar"] = mom_data[species+"Upar"]/mom_data["ioncs"]


    # Load the potential
    mdata = pg.GData('%s-field_%d.gkyl'%(sim_name, frame))
    grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
    x = fix_gridvals(grid[0])
    z = fix_gridvals(grid[1])
    val = val.squeeze()
    mom_data["phi"] = val

    # Interpolate B ratio at plates
    if isim in [3,4,8,9]:
        Binterpolator = interp1d(plate_data[:,0], plate_data[:,1])
        Bratio = Binterpolator(x)
        mom_data["Bratio" ] = Bratio

    #Save grids
    mom_data["x"] = x
    mom_data["z"] = z

    #Apply a floor to n and T
    for species in ["elc", "ion"]:
        mom_data[species+"M0"][mom_data[species+"M0"] < 0] = 1e12
    for species in ["elc", "ion"]:
        mom_data[species+"Temp"][mom_data[species+"Temp"] < 0] = 10

    # Append data
    mom_data_list.append(mom_data)

#Copy some upper plate data to lower plates
mom_data_list[1]["Bratio"] = mom_data_list[3]["Bratio"]
mom_data_list[6]["Bratio"] = mom_data_list[8]["Bratio"]
mom_data_list[0]["Bratio"] = mom_data_list[4]["Bratio"]
mom_data_list[5]["Bratio"] = mom_data_list[9]["Bratio"]


xidx = 0
xidx_core=-1
zidx = [mom_data_list[i]["Zavg"].shape[1]//2 for i in range(12)]
zidx_xpt = [0,0,0]
Rmid_out = mom_data_list[2]["Ravg"][:,zidx[2]]
Rmid_in= mom_data_list[7]["Ravg"][:,zidx[7]]

for bi in range(12):
    mom_data_list[bi]["wallflux"] = np.zeros(mom_data_list[bi]["elcM0"].shape)
    mom_data_list[bi]["plateflux"] = np.zeros(mom_data_list[bi]["elcM0"].shape)

# Now look at Particle Flux to Side Wall. Use surface method
D = 0.6
edge_inds = [0, -1,-1,-1, 0,0, -1,-1,-1, 0]
for bi in [0,1,2,3,4,5,6,7,8,9]:
    diff_density_surf = {}
    for species in ["ion"]:
        diff_density_surf[species] = 0.0
        edge_ind = edge_inds[bi]
        dM0dx = np.gradient(mom_data_list[bi]["ionM0"], mom_data_list[bi]["x"], axis=0, edge_order=2)
        sign = -1
        mom_data_list[bi]["wallflux"][edge_ind] = sign*D*dM0dx[edge_ind]*np.sqrt(mom_data_list[bi]["gxx"][edge_ind])

#Calculate Particle Flux to SE
total_pflux = 0
zedge = [0, 0, -1, -1, 0, 0, -1, -1]
for bi, bidx in enumerate([0, 1, 3,4, 5, 6, 8,9]):
    sign = -1
    mom_data_list[bidx]["plateflux"][:, zedge[bi]] = sign*mom_data_list[bidx]["ionM1"][:, zedge[bi]]*np.sin(mom_data_list[bidx]["Bratio"])


Rall = np.array([])
Zall = np.array([])
niall = np.array([])
neall = np.array([])
Tiall = np.array([])
Teall = np.array([])
upariall = np.array([])
phiall = np.array([])
GammaRadall = np.array([])
GammaParall = np.array([])
for i in range(bmin,bmax):
    Rall = np.append(Rall, Rilist[i].flatten())
    Zall = np.append(Zall, Zilist[i].flatten())
    niall = np.append(niall, mom_data_list[i]["ionM0"].flatten())
    neall = np.append(neall, mom_data_list[i]["elcM0"].flatten())
    Tiall = np.append(Tiall, mom_data_list[i]["ionTemp"].flatten())
    Teall = np.append(Teall, mom_data_list[i]["elcTemp"].flatten())
    upariall = np.append(upariall, mom_data_list[i]["ionUpar"].flatten())
    phiall = np.append(phiall, mom_data_list[i]["phi"].flatten())
    GammaRadall = np.append(GammaRadall, mom_data_list[i]["wallflux"].flatten())
    GammaParall = np.append(GammaParall, mom_data_list[i]["plateflux"].flatten())

alldata = np.column_stack((Rall, Zall, niall, neall, Tiall, Teall, upariall, phiall, GammaRadall, GammaParall))
np.savetxt("./gkeyll_text_output/ehl2data.txt", alldata,  header='R Z ni ne Ti Te upari phi Gamma_R Gamma_Z', comments='')

celldata = np.zeros((12,3), dtype="int")
for i in range(bmin,bmax):
    nR = mom_data_list[i]["ionM0"].shape[0]
    nZ = mom_data_list[i]["ionM0"].shape[1]
    celldata[i] = np.r_[i, nR, nZ]
np.savetxt("./gkeyll_text_output/cells_ehl2data.txt", celldata,  header='blockid nR nZ', comments='', fmt="%d")

print("Processed Gkeyll output for frame %d"%frame)
    
    
   
