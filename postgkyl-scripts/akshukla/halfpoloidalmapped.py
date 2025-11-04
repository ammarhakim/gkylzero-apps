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


import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def fix_gridvals(grid):
    """Output file grids have cell-edge coordinates by default, but the values are at cell centers.
    This function will return the cell-center coordinate grid.
    Usage: cell_center_grid = fix_gridvals(cell_edge_grid)
    """
    grid = np.array(grid).squeeze()
    grid = (grid[0:-1] + grid[1:])/2
    return grid

def myinterpolate(raw_x,raw_z,x,z,data):
    myinterpolator = RegularGridInterpolator((raw_x,raw_z), data, bounds_error=False, fill_value=None)
    x0,z0 = np.meshgrid(x,z)
    return myinterpolator((x0,z0)).T




# Universal params
mp = 1.67262192e-27
me = 9.1093837e-31
eV = 1.602e-19
mu_0 = 12.56637061435917295385057353311801153679e-7
eps0=8.854187817620389850536563031710750260608e-12

masses = {}
masses["elc"] = me
masses["ion"] = 2.014*mp
#masses["ion"] = mp


charges = {}
charges["elc"] = -eV 
charges["ion"] = eV


# Set up names for data loading
sim_dir = "./simdata/step9/"
#sim_dir = "./"
stripped_name = 'step9'
base_name = sim_dir+stripped_name
bmin, bmax = 0, 8
sim_names = ['%s_b%d'%(base_name,i) for i in range(bmin,bmax)]

sim_labels= ['b%d'%i for i in range(bmin,bmax)]


#frame = 1
start= int(sys.argv[1])
neut_frame = start
end_frame = start+1
#end_frame=95

for frame in range(start,end_frame):
    Rlist = []
    Zlist = []
    half_mom_data_list = []
    raw_half_mom_data_list = []
    source_half_mom_data_list = []
    for isim, sim_name in enumerate(sim_names):
    
        mom_data = {}
        raw_mom_data = {}
    
        #Load geometry
    
        bdata = pg.GData('%s-bmag.gkyl'%sim_name)
        grid,val = pg.data.GInterpModal(bdata,poly_order=1,basis_type='ms').interpolate(0)
        mom_data["B"] = val.squeeze()
        raw_mom_data["B"] = bdata.get_values()[:,:,0]/2
        
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
                raw_mom_data["g_%s%s"%(c1,c2)] = gdata.get_values()[:,:,0]/2
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
                raw_mom_data["g%s%s"%(c1,c2)] = gdata.get_values()[:,:,0]/2
                ci+=1
        
        jdata = pg.GData('%s-jacobgeo.gkyl'%sim_name)
        grid,val = pg.data.GInterpModal(jdata,poly_order=1,basis_type='ms').interpolate(0)
        mom_data["J"] = val.squeeze()
        raw_mom_data["J"] = jdata.get_values()[:,:,0]/2
        raw_grid = jdata.get_grid()
        geo_fac = 1/mom_data["J"]/mom_data["B"]
    
        #Load grid data
        node_data = pg.GData(sim_name+"-nodes.gkyl")
        vals = node_data.get_values()
        R = vals[:,:,0]
        Z = vals[:,:,1]
        PHI = vals[:,:,2]
        mom_data["R"] = R
        mom_data["Z"] = Z

        # Get cell avg physical coords
        raw_Zavg = (Z[:,1:] + Z[:,:-1])/2
        raw_Ravg = (R[1:,:] + R[:-1,:])/2
        raw_mom_data["Zavg"] = raw_Zavg
        raw_mom_data["Ravg"] = raw_Ravg

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

        mom_data["Ri"] = Ri.T
        mom_data["Zi"] = Zi.T
    
        Zavg = (Z.T[:,1:] + Z.T[:,:-1])/2
        Ravg = (R.T[1:,:] + R.T[:-1,:])/2
        mom_data["Zavg"] = Zavg
        mom_data["Ravg"] = Ravg

        Zperpavg = (Z.T[1:,:] + Z.T[:-1,:])/2
        mom_data["Zperpavg"] = Zperpavg
    
        Rlist.append(R.T)
        Zlist.append(Z.T)

        raw_g0i, raw_g1i = np.meshgrid(fix_gridvals(raw_grid[0]), fix_gridvals(raw_grid[1]))
        raw_Ri = Rinterpolator((raw_g0i,raw_g1i))
        raw_Zi = Zinterpolator((raw_g0i, raw_g1i))
        raw_mom_data["Ri"] = raw_Ri.T
        raw_mom_data["Zi"] = raw_Zi.T


        #Get Plate angle information
        if isim == 1 : 
            plate_data = np.genfromtxt("./stepplate_data/highres/osol.txt", delimiter = ",")
        if isim == 0 : 
            plate_data = np.genfromtxt("./stepplate_data/highres/opf.txt", delimiter = ",")
        if isim == 4 : 
            plate_data = np.genfromtxt("./stepplate_data/highres/isol.txt", delimiter = ",")
        if isim == 5 : 
            plate_data = np.genfromtxt("./stepplate_data/highres/ipf.txt", delimiter = ",")

    
    
        #Load moment data
        for species in ["elc", "ion"]:
            for mom in ["M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp"]:
                mdata = pg.GData('%s-%s_%s_%d.gkyl'%(sim_name, species,mom,frame))
                raw_mom_data[species+mom] = mdata.get_values()[:,:,0]/2
                raw_grid = mdata.get_grid()
                raw_x = fix_gridvals(raw_grid[0])
                raw_z = fix_gridvals(raw_grid[1])
                grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
                x = fix_gridvals(grid[0])
                z = fix_gridvals(grid[1])
                val = val.squeeze()
                mom_data[species+mom] = val
 
            #Set cell avg moment data
            raw_mom_data[species+"Temp"] =  (masses[species]/3) * (raw_mom_data[species+"M2"] - raw_mom_data[species+"M1"]**2 / raw_mom_data[species+"M0"])/raw_mom_data[species+"M0"] / eV
            raw_mom_data[species+"Tpar"] =  (masses[species]) * (raw_mom_data[species+"M2par"] - raw_mom_data[species+"M1"]**2 / raw_mom_data[species+"M0"])/raw_mom_data[species+"M0"] / eV
            raw_mom_data[species+"Tperp"] =  (masses[species]/2) * (raw_mom_data[species+"M2perp"])/raw_mom_data[species+"M0"] / eV
            raw_mom_data[species+"Q"] =  masses[species]/2 * (raw_mom_data[species+"M3par"] + raw_mom_data[species+"M3perp"])
            raw_mom_data[species+"Upar"] =  raw_mom_data[species+"M1"]/raw_mom_data[species+"M0"]

            #Try interpolated M1 and Q
            #M1interpolator = RegularGridInterpolator((raw_x,raw_z), raw_mom_data[species+"M1"], bounds_error=False, fill_value=None)
            #x0,z0 = np.meshgrid(x,z)
            #mom_data[species+"M1"] = M1interpolator((x0,z0)).T
            #Qinterpolator = RegularGridInterpolator((raw_x,raw_z), raw_mom_data[species+"Q"], bounds_error=False, fill_value=None)
            #x0,z0 = np.meshgrid(x,z)
            #mom_data[species+"Q"] = Qinterpolator((x0,z0)).T
            #mom_data[species+"Q"] =  masses[species]/2 * (mom_data[species+"M3par"] + mom_data[species+"M3perp"])


            #Try interpolated M0
            #M0interpolator = RegularGridInterpolator((raw_x,raw_z), raw_mom_data[species+"M0"], bounds_error=False, fill_value=None)
            #x0,z0 = np.meshgrid(x,z)
            #mom_data[species+"M0"] = M0interpolator((x0,z0)).T

            #use interpolating function
            mom_data[species+"M1"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"M1"])
            mom_data[species+"M0"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"M0"])
            mom_data[species+"Q"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Q"])
            mom_data[species+"Temp"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Temp"])
            mom_data[species+"Tpar"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Tpar"])
            mom_data[species+"Tperp"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Tperp"])
            mom_data[species+"Upar"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Upar"])

            # Set interpolated moment data
            #mom_data[species+"Temp"] =  (masses[species]/3) * (mom_data[species+"M2"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
            #mom_data[species+"Tpar"] =  (masses[species]) * (mom_data[species+"M2par"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
            #mom_data[species+"Tperp"] =  (masses[species]/2) * (mom_data[species+"M2perp"])/mom_data[species+"M0"] / eV
            ##mom_data[species+"Q"] =  masses[species]/2 * (mom_data[species+"M3par"] + mom_data[species+"M3perp"])
            #mom_data[species+"Upar"] =  mom_data[species+"M1"]/mom_data[species+"M0"]
        
        mom_data["Qtot"] =  mom_data["elcQ"]+mom_data["ionQ"]
        raw_mom_data["Qtot"] =  raw_mom_data["elcQ"]+raw_mom_data["ionQ"]


        # Calculate sound speed for ion species
        for species in ["ion" ]:
            mom_data[species+"cs"] = np.sqrt( (mom_data["elcTemp"]*eV + mom_data[species+"Temp"]*eV) / masses[species])
            raw_mom_data[species+"cs"] = np.sqrt( (raw_mom_data["elcTemp"]*eV + raw_mom_data[species+"Temp"]*eV) / masses[species])
        
        for species in ["ion", "elc" ]:
            mom_data[species+"normUpar"] = mom_data[species+"Upar"]/mom_data["ioncs"]
            raw_mom_data[species+"normUpar"] = raw_mom_data[species+"Upar"]/raw_mom_data["ioncs"]


        # Load the potential
        mdata = pg.GData('%s-field_%d.gkyl'%(sim_name, frame))
        grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
        raw_mom_data["phi"] = mdata.get_values()[:,:,0]/2
        x = fix_gridvals(grid[0])
        z = fix_gridvals(grid[1])
        val = val.squeeze()
        mom_data["phi"] = val
        mom_data["time"] = float(mdata.info()[9:21])

        for species in ["elc", "ion"]:
            mom_data[species+"Qwall"] = mom_data[species+"Q"] + charges[species]*mom_data[species+"M1"]*mom_data["phi"]
            raw_mom_data[species+"Qwall"] = raw_mom_data[species+"Q"] + charges[species]*raw_mom_data[species+"M1"]*raw_mom_data["phi"]



        # Interpolate B ratio at plates
        if isim in [1,0,4,5]:
            Binterpolator = interp1d(plate_data[:,0], plate_data[:,1])
            Bratio = Binterpolator(x)
            mom_data["Bratio" ] = Bratio


        #Save grids
        mom_data["x"] = x
        mom_data["z"] = z

        raw_mom_data["x"] = raw_x
        raw_mom_data["z"] = raw_z

        raw_mom_data["rawx"] = raw_grid[0]
        raw_mom_data["rawz"] = raw_grid[1]

        #Load source moment data
        source_mom_data = {}
        if (isim == 6 or isim==7):
            for species in ["elc", "ion"]:
                for mom in ["M0", "M1", "M2", "M2par", "M2perp"]:
                    mdata = pg.GData('%s-%s_source_%s_%d.gkyl'%(sim_name, species,mom,0))
                    grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
                    x = fix_gridvals(grid[0])
                    z = fix_gridvals(grid[1])
                    val = val.squeeze()
                    source_mom_data[species+mom] = val
                # Set interpolated moment data
                source_mom_data[species+"Temp"] =  (masses[species]/3) * (source_mom_data[species+"M2"] - source_mom_data[species+"M1"]**2 / source_mom_data[species+"M0"])/source_mom_data[species+"M0"] / eV
                source_mom_data[species+"Tpar"] =  (masses[species]) * (source_mom_data[species+"M2par"] - source_mom_data[species+"M1"]**2 / source_mom_data[species+"M0"])/source_mom_data[species+"M0"] / eV
                source_mom_data[species+"Tperp"] =  (masses[species]/2) * (source_mom_data[species+"M2perp"])/source_mom_data[species+"M0"] / eV
                source_mom_data[species+"Upar"] =  source_mom_data[species+"M1"]/source_mom_data[species+"M0"]
    

            ion_source_integrand = masses["ion"]/2 *(source_mom_data["ionM2"]) * mom_data["J"]
            elc_source_integrand = masses["elc"]/2 *(source_mom_data["elcM2"]) * mom_data["J"]

            ion_sp = np.sum(ion_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
            elc_sp = np.sum(elc_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
            source_mom_data["ionsourcepower"] = ion_sp
            source_mom_data["elcsourcepower"] = elc_sp

            ion_source_integrand = (source_mom_data["ionM0"]) * mom_data["J"]
            elc_source_integrand = (source_mom_data["elcM0"]) * mom_data["J"]

            ion_sn = np.sum(ion_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
            elc_sn = np.sum(elc_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
            source_mom_data["ionsourcedensity"] = ion_sn
            source_mom_data["elcsourcedensity"] = elc_sn
    
    
    
        # Append data
        half_mom_data_list.append(mom_data)
        raw_half_mom_data_list.append(raw_mom_data)
        source_half_mom_data_list.append(source_mom_data)

    total_sp = (source_half_mom_data_list[6]["ionsourcepower"] + source_half_mom_data_list[6]["elcsourcepower"] + source_half_mom_data_list[7]["ionsourcepower"] + source_half_mom_data_list[7]["elcsourcepower"])*2
    total_sn = (source_half_mom_data_list[6]["ionsourcedensity"] + source_half_mom_data_list[6]["elcsourcedensity"] + source_half_mom_data_list[7]["ionsourcedensity"] + source_half_mom_data_list[7]["elcsourcedensity"])*2
    print(" Total (doubled) source power = MW", total_sp/1e6)
    print(" Total (doubled) source density = ", total_sn)

    # Construct the 12 block data
    mom_data_list = [None]*12
    raw_mom_data_list = [None]*12
    even_keys = ["elcM0", "elcTemp", "ionM0", "ionTemp", "elcM2", "ionM2", "phi", "gxx", "gzz", "Ri", "J", "elcTperp", "elcTpar", "ionTperp", "ionTpar", "B", "J"]
    odd_keys = ["elcM1", "elcUpar", "ionM1", "ionUpar", "elcQ", "elcQwall", "ionQ", "ionQwall", "Zi", "elcnormUpar", "ionnormUpar", "Qtot"]
    
    #Unmodified but renumbered blocks
    mom_data_list[0] = half_mom_data_list[0]
    mom_data_list[1] = half_mom_data_list[1]
    mom_data_list[8] = half_mom_data_list[4]
    mom_data_list[9] = half_mom_data_list[5]

    raw_mom_data_list[0] = raw_half_mom_data_list[0]
    raw_mom_data_list[1] = raw_half_mom_data_list[1]
    raw_mom_data_list[8] = raw_half_mom_data_list[4]
    raw_mom_data_list[9] = raw_half_mom_data_list[5]
    
    
    # Reflected blocks
    mom_data_list[3] = {}
    mom_data_list[6] = {}
    mom_data_list[4] = {}
    mom_data_list[5] = {}
    raw_mom_data_list[3] = {}
    raw_mom_data_list[6] = {}
    raw_mom_data_list[4] = {}
    raw_mom_data_list[5] = {}
    #Do Bratio myself manually
    mom_data_list[3]["Bratio"] = mom_data_list[1]["Bratio"]
    mom_data_list[6]["Bratio"] = mom_data_list[8]["Bratio"]
    mom_data_list[4]["Bratio"] = mom_data_list[0]["Bratio"]
    mom_data_list[5]["Bratio"] = mom_data_list[9]["Bratio"]
    raw_mom_data_list[3]["Bratio"] = mom_data_list[1]["Bratio"]
    raw_mom_data_list[6]["Bratio"] = mom_data_list[8]["Bratio"]
    raw_mom_data_list[4]["Bratio"] = mom_data_list[0]["Bratio"]
    raw_mom_data_list[5]["Bratio"] = mom_data_list[9]["Bratio"]
    
    
    
    # Doubled blocks
    mom_data_list[2] = {}
    mom_data_list[7] = {}
    mom_data_list[10] = {}
    mom_data_list[11] = {}
    raw_mom_data_list[2] = {}
    raw_mom_data_list[7] = {}
    raw_mom_data_list[10] = {}
    raw_mom_data_list[11] = {}
    for key in even_keys:
        mom_data_list[2][key] = np.zeros((half_mom_data_list[2]["ionM0"].shape[0], 2*half_mom_data_list[2]["ionM0"].shape[1]))
        mom_data_list[7][key] = np.zeros((half_mom_data_list[3]["ionM0"].shape[0], 2*half_mom_data_list[3]["ionM0"].shape[1]))
        mom_data_list[10][key] = np.zeros((half_mom_data_list[6]["ionM0"].shape[0], 2*half_mom_data_list[6]["ionM0"].shape[1]))
        mom_data_list[11][key] = np.zeros((half_mom_data_list[7]["ionM0"].shape[0], 2*half_mom_data_list[7]["ionM0"].shape[1]))

        raw_mom_data_list[2][key] = np.zeros((raw_half_mom_data_list[2]["ionM0"].shape[0], 2*raw_half_mom_data_list[2]["ionM0"].shape[1]))
        raw_mom_data_list[7][key] = np.zeros((raw_half_mom_data_list[3]["ionM0"].shape[0], 2*raw_half_mom_data_list[3]["ionM0"].shape[1]))
        raw_mom_data_list[10][key] = np.zeros((raw_half_mom_data_list[6]["ionM0"].shape[0], 2*raw_half_mom_data_list[6]["ionM0"].shape[1]))
        raw_mom_data_list[11][key] = np.zeros((raw_half_mom_data_list[7]["ionM0"].shape[0], 2*raw_half_mom_data_list[7]["ionM0"].shape[1]))
    
    for key in odd_keys:
        mom_data_list[2][key] = np.zeros((half_mom_data_list[2]["ionM0"].shape[0], 2*half_mom_data_list[2]["ionM0"].shape[1]))
        mom_data_list[7][key] = np.zeros((half_mom_data_list[3]["ionM0"].shape[0], 2*half_mom_data_list[3]["ionM0"].shape[1]))
        mom_data_list[10][key] = np.zeros((half_mom_data_list[6]["ionM0"].shape[0], 2*half_mom_data_list[6]["ionM0"].shape[1]))
        mom_data_list[11][key] = np.zeros((half_mom_data_list[7]["ionM0"].shape[0], 2*half_mom_data_list[7]["ionM0"].shape[1]))

        raw_mom_data_list[2][key] = np.zeros((raw_half_mom_data_list[2]["ionM0"].shape[0], 2*raw_half_mom_data_list[2]["ionM0"].shape[1]))
        raw_mom_data_list[7][key] = np.zeros((raw_half_mom_data_list[3]["ionM0"].shape[0], 2*raw_half_mom_data_list[3]["ionM0"].shape[1]))
        raw_mom_data_list[10][key] = np.zeros((raw_half_mom_data_list[6]["ionM0"].shape[0], 2*raw_half_mom_data_list[6]["ionM0"].shape[1]))
        raw_mom_data_list[11][key] = np.zeros((raw_half_mom_data_list[7]["ionM0"].shape[0], 2*raw_half_mom_data_list[7]["ionM0"].shape[1]))
    
    
    #Do x myself manually
    mom_data_list[3]["x"] = mom_data_list[1]["x"]
    mom_data_list[6]["x"] = mom_data_list[8]["x"]
    mom_data_list[4]["x"] = mom_data_list[0]["x"]
    mom_data_list[5]["x"] = mom_data_list[9]["x"]
    raw_mom_data_list[3]["x"] = raw_mom_data_list[1]["x"]
    raw_mom_data_list[6]["x"] = raw_mom_data_list[8]["x"]
    raw_mom_data_list[4]["x"] = raw_mom_data_list[0]["x"]
    raw_mom_data_list[5]["x"] = raw_mom_data_list[9]["x"]
    
    
    mom_data_list[2]["x"] = mom_data_list[1]["x"]
    mom_data_list[7]["x"] = mom_data_list[8]["x"]
    mom_data_list[10]["x"] = half_mom_data_list[6]["x"]
    mom_data_list[11]["x"] = half_mom_data_list[7]["x"]
    raw_mom_data_list[2]["x"] = raw_mom_data_list[1]["x"]
    raw_mom_data_list[7]["x"] = raw_mom_data_list[8]["x"]
    raw_mom_data_list[10]["x"] = raw_half_mom_data_list[6]["x"]
    raw_mom_data_list[11]["x"] = raw_half_mom_data_list[7]["x"]

    #Do z myself manually
    mom_data_list[3]["z"] = -np.flip(mom_data_list[1]["z"])
    mom_data_list[6]["z"] = -np.flip(mom_data_list[8]["z"])
    mom_data_list[4]["z"] = np.flip(mom_data_list[0]["z"])
    mom_data_list[5]["z"] = np.flip(mom_data_list[9]["z"])

    raw_mom_data_list[3]["z"] = -np.flip(raw_mom_data_list[1]["z"])
    raw_mom_data_list[6]["z"] = -np.flip(raw_mom_data_list[8]["z"])
    raw_mom_data_list[4]["z"] = np.flip(raw_mom_data_list[0]["z"])
    raw_mom_data_list[5]["z"] = np.flip(raw_mom_data_list[9]["z"])
    
    mom_data_list[2]["z"] = np.r_[half_mom_data_list[2]["z"], -np.flip(half_mom_data_list[2]["z"])]
    mom_data_list[7]["z"] = np.r_[-np.flip(half_mom_data_list[2]["z"]), half_mom_data_list[2]["z"]]
    mom_data_list[10]["z"] = np.array([half_mom_data_list[6]["z"][0] + np.diff(half_mom_data_list[6]["z"])[0]*i for i in range(2*len(half_mom_data_list[6]["z"]))])
    mom_data_list[11]["z"] = np.array([half_mom_data_list[7]["z"][-1] - np.diff(half_mom_data_list[7]["z"])[0]*i for i in range(2*len(half_mom_data_list[7]["z"]))])

    raw_mom_data_list[2]["z"] = np.r_[raw_half_mom_data_list[2]["z"], -np.flip(raw_half_mom_data_list[2]["z"])]
    raw_mom_data_list[7]["z"] = np.r_[-np.flip(raw_half_mom_data_list[2]["z"]), raw_half_mom_data_list[2]["z"]]
    raw_mom_data_list[10]["z"] = np.array([raw_half_mom_data_list[6]["z"][0] + np.diff(raw_half_mom_data_list[6]["z"])[0]*i for i in range(2*len(raw_half_mom_data_list[6]["z"]))])
    raw_mom_data_list[11]["z"] = np.array([raw_half_mom_data_list[7]["z"][-1] - np.diff(raw_half_mom_data_list[7]["z"])[0]*i for i in range(2*len(raw_half_mom_data_list[7]["z"]))])
    
    
    
    # Flip some and double/reflect some
    for key in even_keys:
        #Upper SOL
        mom_data_list[3][key] = np.flip(mom_data_list[1][key], axis=-1)
        mom_data_list[6][key] = np.flip(mom_data_list[8][key], axis=-1)
        raw_mom_data_list[3][key] = np.flip(raw_mom_data_list[1][key], axis=-1)
        raw_mom_data_list[6][key] = np.flip(raw_mom_data_list[8][key], axis=-1)
        #Upper PF
        mom_data_list[4][key] = np.flip(mom_data_list[0][key], axis=-1)
        mom_data_list[5][key] = np.flip(mom_data_list[9][key], axis=-1)
        raw_mom_data_list[4][key] = np.flip(raw_mom_data_list[0][key], axis=-1)
        raw_mom_data_list[5][key] = np.flip(raw_mom_data_list[9][key], axis=-1)
    
        #Outer Middle
        mom_data_list[2][key][:, 0:mom_data_list[2][key].shape[1]//2] = half_mom_data_list[2][key] 
        mom_data_list[2][key][:, mom_data_list[2][key].shape[1]//2:] = np.flip(half_mom_data_list[2][key], axis=-1)
        raw_mom_data_list[2][key][:, 0:raw_mom_data_list[2][key].shape[1]//2] = raw_half_mom_data_list[2][key] 
        raw_mom_data_list[2][key][:, raw_mom_data_list[2][key].shape[1]//2:] = np.flip(raw_half_mom_data_list[2][key], axis=-1)
    
        mom_data_list[10][key][:, 0:mom_data_list[10][key].shape[1]//2] = half_mom_data_list[6][key] 
        mom_data_list[10][key][:, mom_data_list[10][key].shape[1]//2:] = np.flip(half_mom_data_list[6][key], axis=-1)
        raw_mom_data_list[10][key][:, 0:raw_mom_data_list[10][key].shape[1]//2] = raw_half_mom_data_list[6][key] 
        raw_mom_data_list[10][key][:, raw_mom_data_list[10][key].shape[1]//2:] = np.flip(raw_half_mom_data_list[6][key], axis=-1)
    
        #Inner Middle
        mom_data_list[7][key][:, 0:mom_data_list[7][key].shape[1]//2] = np.flip(half_mom_data_list[3][key], axis=-1) 
        mom_data_list[7][key][:, mom_data_list[7][key].shape[1]//2:] = half_mom_data_list[3][key]
        raw_mom_data_list[7][key][:, 0:raw_mom_data_list[7][key].shape[1]//2] = np.flip(raw_half_mom_data_list[3][key], axis=-1) 
        raw_mom_data_list[7][key][:, raw_mom_data_list[7][key].shape[1]//2:] = raw_half_mom_data_list[3][key]
    
        mom_data_list[11][key][:, 0:mom_data_list[11][key].shape[1]//2] = np.flip(half_mom_data_list[7][key], axis=-1) 
        mom_data_list[11][key][:, mom_data_list[11][key].shape[1]//2:] = half_mom_data_list[7][key]
        raw_mom_data_list[11][key][:, 0:raw_mom_data_list[11][key].shape[1]//2] = np.flip(raw_half_mom_data_list[7][key], axis=-1) 
        raw_mom_data_list[11][key][:, raw_mom_data_list[11][key].shape[1]//2:] = raw_half_mom_data_list[7][key]
    
    for key in odd_keys:
        #Upper SOL
        mom_data_list[3][key] = -np.flip(mom_data_list[1][key], axis=-1)
        mom_data_list[6][key] = -np.flip(mom_data_list[8][key], axis=-1)
        raw_mom_data_list[3][key] = -np.flip(raw_mom_data_list[1][key], axis=-1)
        raw_mom_data_list[6][key] = -np.flip(raw_mom_data_list[8][key], axis=-1)
        #Upper PF
        mom_data_list[4][key] = -np.flip(mom_data_list[0][key], axis=-1)
        mom_data_list[5][key] = -np.flip(mom_data_list[9][key], axis=-1)
        raw_mom_data_list[4][key] = -np.flip(raw_mom_data_list[0][key], axis=-1)
        raw_mom_data_list[5][key] = -np.flip(raw_mom_data_list[9][key], axis=-1)
    
        #Outer Middle
        mom_data_list[2][key][:, 0:mom_data_list[2][key].shape[1]//2] = half_mom_data_list[2][key] 
        mom_data_list[2][key][:, mom_data_list[2][key].shape[1]//2:] = -np.flip(half_mom_data_list[2][key], axis=-1)
        raw_mom_data_list[2][key][:, 0:raw_mom_data_list[2][key].shape[1]//2] = raw_half_mom_data_list[2][key] 
        raw_mom_data_list[2][key][:, raw_mom_data_list[2][key].shape[1]//2:] = -np.flip(raw_half_mom_data_list[2][key], axis=-1)
    
        mom_data_list[10][key][:, 0:mom_data_list[10][key].shape[1]//2] = half_mom_data_list[6][key] 
        mom_data_list[10][key][:, mom_data_list[10][key].shape[1]//2:] = -np.flip(half_mom_data_list[6][key], axis=-1)
        raw_mom_data_list[10][key][:, 0:raw_mom_data_list[10][key].shape[1]//2] = raw_half_mom_data_list[6][key] 
        raw_mom_data_list[10][key][:, raw_mom_data_list[10][key].shape[1]//2:] = -np.flip(raw_half_mom_data_list[6][key], axis=-1)
    
        #Inner Middle
        mom_data_list[7][key][:, 0:mom_data_list[7][key].shape[1]//2] = -np.flip(half_mom_data_list[3][key], axis=-1) 
        mom_data_list[7][key][:, mom_data_list[7][key].shape[1]//2:] = half_mom_data_list[3][key]
        raw_mom_data_list[7][key][:, 0:raw_mom_data_list[7][key].shape[1]//2] = -np.flip(raw_half_mom_data_list[3][key], axis=-1) 
        raw_mom_data_list[7][key][:, raw_mom_data_list[7][key].shape[1]//2:] = raw_half_mom_data_list[3][key]
    
        mom_data_list[11][key][:, 0:mom_data_list[11][key].shape[1]//2] = -np.flip(half_mom_data_list[7][key], axis=-1) 
        mom_data_list[11][key][:, mom_data_list[11][key].shape[1]//2:] = half_mom_data_list[7][key]
        raw_mom_data_list[11][key][:, 0:raw_mom_data_list[11][key].shape[1]//2] = -np.flip(raw_half_mom_data_list[7][key], axis=-1) 
        raw_mom_data_list[11][key][:, raw_mom_data_list[11][key].shape[1]//2:] = raw_half_mom_data_list[7][key]








    
    #Print input power and density stats:
    #print("ion source power = %e, elc source power = %e"%(source_mom_data_list[10]["ionsourcepower"] + source_mom_data_list[11]["ionsourcepower"], source_mom_data_list[10]["elcsourcepower"]+ source_mom_data_list[11]["elcsourcepower"]))
    #print("ion source density = %e, elc source density = %e"%(source_mom_data_list[10]["ionsourcedensity"] + source_mom_data_list[11]["ionsourcedensity"], source_mom_data_list[10]["elcsourcedensity"]+ source_mom_data_list[11]["elcsourcedensity"]))

    #xidx = Ri.shape[0]//2
    xidx = -1
    xidx_core=0
    zidx = [mom_data_list[i]["Zi"].shape[1]//2 for i in range(12)]
    zidx_xpt = [0,0,0]
    Rmid_out = mom_data_list[2]["Ri"][:,zidx[2]]
    Rmid_in= mom_data_list[7]["Ri"][:,zidx[7]]

    #raw_zidx = [raw_mom_data_list[i]["Ri"].shape[1]//2 for i in range(12)]
    #raw_Rmid_out = raw_mom_data_list[2]["Ri"][:,raw_zidx[2]]
    #raw_Rmid_in= raw_mom_data_list[7]["Ri"][:,raw_zidx[7]]

    # Now look at heat to side wall from diffusion
    # Use surface method
    diff_power_surf = {}
    for species in ["elc", "ion"]:
        diff_power_surf[species] = 0.0
        for edge_ind in [-1]:
            dM2dx = np.gradient(mom_data_list[2][species+"M2"], mom_data_list[2]["x"], axis=0, edge_order=2)
            
            D=0.6
            inner = mom_data_list[2]["gxx"]*mom_data_list[2]["J"]*D* masses[species]/2 * dM2dx
            diffintegrand = inner
            diff_power_surf[species]  += np.abs(np.sum(diffintegrand[edge_ind,:]*np.diff(mom_data_list[2]["z"])[0]) * 2 * np.pi/1e6)
    print("outboard diff_power_surf [MW] = ", diff_power_surf)

    diff_power_surf = {}
    for species in ["elc", "ion"]:
        diff_power_surf[species] = 0.0
        for edge_ind in [-1]:
            dM2dx = np.gradient(mom_data_list[7][species+"M2"], mom_data_list[7]["x"], axis=0, edge_order=2)
            
            D=0.6
            inner = mom_data_list[7]["gxx"]*mom_data_list[7]["J"]*D* masses[species]/2 * dM2dx
            diffintegrand = inner
            diff_power_surf[species]  += np.abs(np.sum(diffintegrand[edge_ind,:]*np.diff(mom_data_list[7]["z"])[0]) * 2 * np.pi/1e6)
    print("inboard diff_power_surf [MW] = ", diff_power_surf)

    ## Now look at Particle Flux to Side Wall
    ## Use surface method
    total_wall_pflux = 0.0
    edge_inds = [0, -1,-1,-1, 0,0, -1,-1,-1, 0]
    signs = [1, -1,-1,-1, 1,1, -1,-1,-1, 1]
    for bi in [0,1,2,3,4,5,6,7,8,9]:
        for species in ["elc", "ion"]:
            edge_ind = edge_inds[bi]
            sign = signs[bi]
            dM0dx = np.gradient(mom_data_list[bi][species+"M0"], mom_data_list[bi]["x"], axis=0, edge_order=2)
            
            D=0.6
            inner = mom_data_list[bi]["gxx"]*mom_data_list[bi]["J"]*D * dM0dx
            diffintegrand = inner
            total_wall_pflux  += sign*np.sum(diffintegrand[edge_ind,:]*np.diff(mom_data_list[bi]["z"])[0]) * 2 * np.pi

    ## Now look at Heat Flux to Side Wall
    ## Use surface method
    total_wall_hflux = 0.0
    #edge_inds = [0, -1,-1,-1, 0,0, -1,-1,-1, 0]
    #signs = [1, -1,-1,-1, 1,1, -1,-1,-1, 1]
    edge_inds = [-1, 0,0,0, -1,-1, 0,0,0, -1]
    signs = [-1, 1,1,1, -1,-1, 1,1,1, -1]
    for bi in [0,1,2,3,4,5,6,7,8,9]:
        for species in ["elc", "ion"]:
            edge_ind = edge_inds[bi]
            sign = signs[bi]
            dM2dx = np.gradient(mom_data_list[bi][species+"M2"], mom_data_list[bi]["x"], axis=0, edge_order=2)
            
            D=0.6
            inner = mom_data_list[bi]["gxx"]*mom_data_list[bi]["J"]*D* masses[species]/2 * dM2dx
            diffintegrand = inner
            total_wall_hflux  += sign*np.sum(diffintegrand[edge_ind,:]*np.diff(mom_data_list[bi]["z"])[0]) * 2 * np.pi



    # Calculate Gradient scale lengths at midplane
    for species in ["elc", "ion" ]:
        mom_data_list[2][species + "M0_L"] = np.abs(mom_data_list[2][species+"M0"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"M0"][:, zidx[2] ], mom_data_list[2]["Ri"][:, zidx[2]], axis =0, edge_order=2))
        mom_data_list[2][species + "Temp_L"] = np.abs(mom_data_list[2][species+"Temp"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"Temp"][:, zidx[2] ], mom_data_list[2]["Ri"][:, zidx[2]], axis =0, edge_order=2))
        mom_data_list[2][species + "Tpar_L"] = np.abs(mom_data_list[2][species+"Tpar"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"Tpar"][:, zidx[2] ], mom_data_list[2]["Ri"][:, zidx[2]], axis =0, edge_order=2) )
        mom_data_list[2][species + "Tperp_L"] = np.abs(mom_data_list[2][species+"Tperp"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"Tperp"][:, zidx[2] ], mom_data_list[2]["Ri"][:, zidx[2]], axis =0, edge_order=2))



    #Calculate Particle Flux to SE
    total_pflux = 0
    zedge = [0, 0, -1, -1, 0, 0, -1, -1]
    for bi, bidx in enumerate([0, 1, 3,4, 5, 6, 8,9]):
        TotalM1 = np.zeros(mom_data_list[bidx]["elcM1"].shape)
        for species in ["elc", "ion"]:
            TotalM1 += mom_data_list[bidx][species+"M1"]
        integrand = TotalM1/mom_data_list[bidx]["B"]

        pflux = np.sum(integrand[:,zedge[bi]]*np.diff(mom_data_list[bidx]["x"])[0]) * 2 * np.pi
        if(bi==1):
            print("block 1 pflux = %e\n"%pflux)
        if zedge[bi] == 0 :
            total_pflux -= pflux
        else:
            total_pflux+=pflux

    #Calculate Heat to SE
    total_heat = 0
    total_outboard_heat = 0
    total_inboard_heat = 0
    zedge = [0, 0, -1, -1, 0, 0, -1, -1]
    for bi, bidx in enumerate([0, 1, 3, 4, 5, 6, 8, 9]):
        TotalQ = np.zeros(mom_data_list[bidx]["elcQ"].shape)
        for species in ["elc", "ion"]:
            TotalQ += mom_data_list[bidx][species+"Q"]
        integrand = TotalQ/mom_data_list[bidx]["B"]

        heat = np.sum(integrand[:,zedge[bi]]*np.diff(mom_data_list[bidx]["x"])[0]) * 2 * np.pi
        if zedge[bi] == 0 :
            total_heat-= heat
            if bidx in [5,6,8,9]:
                total_inboard_heat -= heat
            else:
                total_outboard_heat -= heat
        else:
            total_heat+= heat
            if bidx in [5,6,8,9]:
                total_inboard_heat += heat
            else:
                total_outboard_heat += heat

    #Calculate ES heat to se
    total_esheat = 0
    total_outboard_esheat = 0
    total_inboard_esheat = 0
    zedge = [0, 0, -1, -1, 0, 0, -1, -1]
    for bi, bidx in enumerate([0, 1, 3,4, 5, 6, 8,9]):
        TotalQes = np.zeros(mom_data_list[bidx]["elcQ"].shape)
        for species in ["elc", "ion"]:
            TotalQes +=  charges[species]*mom_data_list[bidx][species+"M1"]
        integrand = mom_data_list[bidx]["phi"]/mom_data_list[bidx]["B"]*TotalQes
        es_heat = np.sum(integrand[:,zedge[bi]]*np.diff(mom_data_list[bidx]["x"])[0]) * 2 * np.pi
        if zedge[bi] == 0 :
            total_esheat-= es_heat
            if bidx in [5,6,8,9]:
                total_inboard_esheat-=es_heat
            else:
                total_outboard_esheat-=es_heat
        else:
            total_esheat+= es_heat
            if bidx in [5,6,8,9]:
                total_inboard_esheat+=es_heat
            else:
                total_outboard_esheat+=es_heat



    ##Calculate ES to plate
    #zedge = [0,-1,0,-1]
    #for bi, bidx in enumerate([1,3,6,8]):
    #    TotalES = np.zeros(mom_data_list[bidx]["elcM1"].shape)
    #    for species in ["elc", "ion"]:
    #        TotalES += charges[species]*mom_data_list[bidx][species+"M1"]*mom_data_list[bidx]["phi"]
    #    integrand = TotalES/mom_data_list[bidx]["B"]

    #    esheat = np.sum(integrand[:,zedge[bi]]*np.diff(mom_data_list[bidx]["x"])[0]) * 2 * np.pi
    #    print("bidx = %d, esheat = %e\n"%( bidx, esheat));

    print("Total Heat to SE = %g MW"%(total_heat/1e6))
    print("Total ES Heat to Wall = %g MW"%(total_esheat/1e6))
    print("Total Heat to Wall = %g Mw"%((total_heat + total_esheat)/1e6))
    print("Total pflux to SE = %g "%(total_pflux))

    print("Total Outboard Heat to SE = %g MW"%(total_outboard_heat/1e6))
    print("Total Inboard Heat to SE = %g MW"%(total_inboard_heat/1e6))

    print("Total Outboard ES Heat to SE = %g MW"%(total_outboard_esheat/1e6))
    print("Total Inboard ES Heat to SE = %g MW"%(total_inboard_esheat/1e6))

    print("Total Heat flux to side walls = %g MW"%(total_wall_hflux/1e6))
    print("Total Particle flux to side walls = %g"%total_wall_pflux)

    
    
    
    
    
    #Now let us set up data for plotting
    raw_outboard_data = {}
    outboard_data = {}
    raw_keys = ["ionM0"]
    keys = ["z"]
    interp_keys = ["ionM0"]
    for raw_key in raw_keys : 
        raw_outboard_data[raw_key] = np.column_stack((raw_mom_data_list[1][raw_key], raw_mom_data_list[2][raw_key], raw_mom_data_list[3][raw_key]))
    for key in keys : 
        outboard_data[key] = np.concatenate((mom_data_list[1][key], mom_data_list[2][key], mom_data_list[3][key]))
        raw_outboard_data[key] = np.concatenate((raw_mom_data_list[1][key], raw_mom_data_list[2][key], raw_mom_data_list[3][key]))
    for ikey in interp_keys:
        outboard_data[ikey] = myinterpolate(raw_mom_data_list[1]["x"], raw_outboard_data["z"], mom_data_list[1]["x"], outboard_data["z"], raw_outboard_data[ikey])

#Make plots
quantities  = ["elcM0", "ionM0", "elcTemp", "elcTpar", "elcTperp", "ionTemp", "ionTpar", "ionTperp", "J"]
quant_labels= [r'$n_e\, [m^{-3}]$', r'$n_i\, [m^{-3}]$', r'$T_e\, [eV]$', r'$T_{e,\parallel} \, [eV]$',  r'$T_{e,\perp} \, [eV]$', r'$T_i\, [eV]$', r'$T_{i,\parallel} \, [eV]$',  r'$T_{i,\perp} \, [eV]$', r'$J$']
quant_savenames = ["ne", "ni", "Te", "Tepar", "Teperp", "Ti", "Tipar", "Tiperp", "J"]
quant_min = [1e16, 1e16, 20,   20,   20,   20,   20,   20  ,0]
quant_max = [2e20, 2e20, 1500, 1500, 1500, 1500, 1500, 1500, 20]
bmin  = 0
bmax = 12
for qidx, qname in enumerate(quantities):
    fig, ax = plt.subplots(figsize = (4,9))
    zfig, zax = plt.subplots(figsize = (4,4))
    qmin = 1e30
    qmax = -1e30
    for i in range(bmin, bmax):
        q = mom_data_list[i][qname]
        qmax = max(min(np.max(q), quant_max[qidx]), qmax)
        qmin = min(max(np.min(q), quant_min[qidx]), qmin)
    for i in range(bmin, bmax):
        q = mom_data_list[i][qname]
        if qidx in [0,1]:
            im = ax.pcolor(mom_data_list[i]["Ri"], mom_data_list[i]["Zi"], q, cmap='inferno', norm=mpl.colors.LogNorm(vmin=qmin, vmax=qmax))
            zim = zax.pcolor(mom_data_list[i]["Ri"], mom_data_list[i]["Zi"], q, cmap='inferno', norm=mpl.colors.LogNorm(vmin=qmin, vmax=qmax))
        else:
            im = ax.pcolor(mom_data_list[i]["Ri"], mom_data_list[i]["Zi"], q, cmap='inferno', vmin=qmin, vmax=qmax)
            zim = zax.pcolor(mom_data_list[i]["Ri"], mom_data_list[i]["Zi"], q, cmap='inferno', vmin=qmin, vmax=qmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label(quant_labels[qidx])

    cbar.set_label(quant_labels[qidx], rotation=270, labelpad=25, fontsize=16)
    ax.axis("tight")
    ax.axis("image")
    ax.set_xlabel("R [m]", fontsize=16)
    ax.set_ylabel("Z [m]", fontsize=16)
    ax.set_xlim(1.45,6.2)
    ax.set_ylim(-8.7,8.7)
    fig.tight_layout()
    fig.savefig('/global/homes/a/akshukla/geopaper/figures/%s/poloidalplots/%s.png'%(stripped_name, quant_savenames[qidx]), dpi=300)

 
    divider = make_axes_locatable(zax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(zim, cax=cax, orientation='vertical')
    cbar.set_label(quant_labels[qidx])

    cbar.set_label(quant_labels[qidx], rotation=270, labelpad=25, fontsize=16)
    zax.axis("tight")
    zax.axis("image")
    zax.set_xlabel("R [m]", fontsize=16)
    zax.set_ylabel("Z [m]", fontsize=16)
    zax.set_xlim(2.12,3.0)
    zax.set_ylim(5.36,6.7)
    zfig.tight_layout()
    zfig.savefig('/global/homes/a/akshukla/geopaper/figures/%s/zoomedpoloidalplots/%s.png'%(stripped_name, quant_savenames[qidx]), dpi=300)

 
