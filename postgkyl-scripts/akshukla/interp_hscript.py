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
sim_dir = "./simdata/h11/"
#sim_dir = "./"
base_name = sim_dir+'h11'
bmin, bmax = 0, 12
sim_names = ['%s_b%d'%(base_name,i) for i in range(bmin,bmax)]

sim_labels= ['b%d'%i for i in range(bmin,bmax)]


#frame = 1
start= int(sys.argv[1])
neut_frame = start
end_frame = start+1
#end_frame=500

for frame in range(start,end_frame):
    Rlist = []
    Zlist = []
    mom_data_list = []
    raw_mom_data_list = []
    source_mom_data_list = []
    for isim, sim_name in enumerate(sim_names):
    
        mom_data = {}
        raw_mom_data = {}
    
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
    
        Zavg = (Z.T[:,1:] + Z.T[:,:-1])/2
        Ravg = (R.T[1:,:] + R.T[:-1,:])/2
        mom_data["Zavg"] = Zavg
        mom_data["Ravg"] = Ravg

        Zperpavg = (Z.T[1:,:] + Z.T[:-1,:])/2
        mom_data["Zperpavg"] = Zperpavg
    
        Rlist.append(R.T)
        Zlist.append(Z.T)

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
            M1interpolator = RegularGridInterpolator((raw_x,raw_z), raw_mom_data[species+"M1"], bounds_error=False, fill_value=None)
            x0,z0 = np.meshgrid(x,z)
            mom_data[species+"M1"] = M1interpolator((x0,z0)).T
            Qinterpolator = RegularGridInterpolator((raw_x,raw_z), raw_mom_data[species+"Q"], bounds_error=False, fill_value=None)
            x0,z0 = np.meshgrid(x,z)
            mom_data[species+"Q"] = Qinterpolator((x0,z0)).T

            # Set interpolated moment data
            mom_data[species+"Temp"] =  (masses[species]/3) * (mom_data[species+"M2"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
            mom_data[species+"Tpar"] =  (masses[species]) * (mom_data[species+"M2par"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
            mom_data[species+"Tperp"] =  (masses[species]/2) * (mom_data[species+"M2perp"])/mom_data[species+"M0"] / eV
            #mom_data[species+"Q"] =  masses[species]/2 * (mom_data[species+"M3par"] + mom_data[species+"M3perp"])
            mom_data[species+"Upar"] =  mom_data[species+"M1"]/mom_data[species+"M0"]
        
        mom_data["Qtot"] =  mom_data["elcQ"]+mom_data["ionQ"]


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



        # Interpolate B ratio at plates
        if isim in [3,4,8,9]:
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
        if (isim == 10 or isim==11):
            for species in ["elc", "ion"]:
                for mom in ["M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp"]:
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
                source_mom_data[species+"Q"] =  masses[species]/2 * (source_mom_data[species+"M3par"] + source_mom_data[species+"M3perp"])
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
        mom_data_list.append(mom_data)
        raw_mom_data_list.append(raw_mom_data)
        source_mom_data_list.append(source_mom_data)

    
    #Copy some upper plate data to lower plates
    mom_data_list[1]["Bratio"] = mom_data_list[3]["Bratio"]
    mom_data_list[6]["Bratio"] = mom_data_list[8]["Bratio"]
    mom_data_list[0]["Bratio"] = mom_data_list[4]["Bratio"]
    mom_data_list[5]["Bratio"] = mom_data_list[3]["Bratio"]

    #Print input power and density stats:
    print("ion source power = %e, elc source power = %e"%(source_mom_data_list[10]["ionsourcepower"] + source_mom_data_list[11]["ionsourcepower"], source_mom_data_list[10]["elcsourcepower"]+ source_mom_data_list[11]["elcsourcepower"]))
    print("ion source density = %e, elc source density = %e"%(source_mom_data_list[10]["ionsourcedensity"] + source_mom_data_list[11]["ionsourcedensity"], source_mom_data_list[10]["elcsourcedensity"]+ source_mom_data_list[11]["elcsourcedensity"]))

    #xidx = Ravg.shape[0]//2
    xidx = 0
    xidx_core=-1
    zidx = [mom_data_list[i]["Zavg"].shape[1]//2 for i in range(12)]
    zidx_xpt = [0,0,0]
    Rmid_out = mom_data_list[2]["Ravg"][:,zidx[2]]
    Rmid_in= mom_data_list[7]["Ravg"][:,zidx[7]]

    raw_zidx = [raw_mom_data_list[i]["Zavg"].shape[1]//2 for i in range(12)]
    raw_Rmid_out = raw_mom_data_list[2]["Ravg"][:,raw_zidx[2]]
    raw_Rmid_in= raw_mom_data_list[7]["Ravg"][:,raw_zidx[7]]

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
    edge_inds = [0, -1,-1,-1, 0,0, -1,-1,-1, 0]
    signs = [1, -1,-1,-1, 1,1, -1,-1,-1, 1]
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
        mom_data_list[2][species + "M0_L"] = np.abs(mom_data_list[2][species+"M0"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"M0"][:, zidx[2] ], mom_data_list[2]["Ravg"][:, zidx[2]], axis =0, edge_order=2))
        mom_data_list[2][species + "Temp_L"] = np.abs(mom_data_list[2][species+"Temp"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"Temp"][:, zidx[2] ], mom_data_list[2]["Ravg"][:, zidx[2]], axis =0, edge_order=2))
        mom_data_list[2][species + "Tpar_L"] = np.abs(mom_data_list[2][species+"Tpar"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"Tpar"][:, zidx[2] ], mom_data_list[2]["Ravg"][:, zidx[2]], axis =0, edge_order=2) )
        mom_data_list[2][species + "Tperp_L"] = np.abs(mom_data_list[2][species+"Tperp"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"Tperp"][:, zidx[2] ], mom_data_list[2]["Ravg"][:, zidx[2]], axis =0, edge_order=2))



    #Calculate Particle Flux to SE
    total_pflux = 0
    zedge = [0, 0, -1, -1, 0, 0, -1, -1]
    for bi, bidx in enumerate([0, 1, 3,4, 5, 6, 8,9]):
        TotalM1 = np.zeros(mom_data_list[bidx]["elcM1"].shape)
        for species in ["elc", "ion"]:
            TotalM1 += mom_data_list[bidx][species+"M1"]
        integrand = TotalM1/mom_data_list[bidx]["B"]

        pflux = np.sum(integrand[:,zedge[bi]]*np.diff(mom_data_list[bidx]["x"])[0]) * 2 * np.pi
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

    
    
    
    
    
    fig,ax = plt.subplots(nrows=2, ncols=4, figsize=(16,9))
    pffig,pfax = plt.subplots(nrows=2, ncols=4, figsize=(16,9))
    
    linestyles = ['-' , '-' , '-' , '--']
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    
    mb_handle = mlines.Line2D([],[], color = 'k', linestyle = '-', label = "MB")
    sb_handle = mlines.Line2D([],[], color = 'k', linestyle = '--', label = "SB")
    handles = [mb_handle, sb_handle]
    
    elc_handle = mlines.Line2D([],[], color = 'tab:blue', label = "elc")
    ion_handle = mlines.Line2D([],[], color = 'tab:orange', label = "ion")
    out_handle = mlines.Line2D([],[], color = 'k', label = "outboard")
    in_handle = mlines.Line2D([],[], color = 'k', label = "inboard", linestyle="dashed")
    handles_species = [elc_handle, ion_handle]


    t_handle = mlines.Line2D([],[], color = 'tab:blue', label = r'$T$')
    tpar_handle = mlines.Line2D([],[], color = 'tab:orange', label = r'$T_\parallel$')
    tperp_handle = mlines.Line2D([],[], color = 'tab:green', label = r'$T_\perp$')
    temp_handles = [t_handle, tpar_handle, tperp_handle]
    temp_colors = ["tab:blue", "tab:orange", "tab:green"]
    elctemp_handle = mlines.Line2D([],[], color = 'k', label = "elc")
    iontemp_handle = mlines.Line2D([],[], color = 'k', label = "ion", linestyle="dashed")
    
    handles_all = [mb_handle, sb_handle, elc_handle, ion_handle]
    
    # Parallel
    # DENSITY
    for i in range(1,4):
        for s, species in enumerate(["elc", "ion"]):
        #for s, species in enumerate(["elc"]):
            if species == "H0":
                ax[0,0].plot(mom_data_list[i]["Zavg"][xidx], mom_data_list[i][species+"M0"][xidx,:], color = colors[s])
            else:
                ax[0,0].plot(mom_data_list[i]["Zavg"][xidx], mom_data_list[i][species+"M0"][xidx,:], color = colors[s])

    for i in range(6,9):
        for s, species in enumerate(["elc", "ion"]):
        #for s, species in enumerate(["elc"]):
            if species == "H0":
                ax[0,0].plot(mom_data_list[i]["Zavg"][xidx], mom_data_list[i][species+"M0"][xidx,:], color = colors[s], linestyle="dashed")
            else:
                ax[0,0].plot(mom_data_list[i]["Zavg"][xidx], mom_data_list[i][species+"M0"][xidx,:], color = colors[s], linestyle="dashed")
    
    ax[0,0].set_xlabel('Z [m]')
    ax[0,0].set_ylabel(r'$n\, [m^{-3}]$')
    ax[0,0].legend(handles = handles_species + [out_handle, in_handle] )


    # DENSITY PF
    for i in range(4,6):
        for s, species in enumerate(["elc", "ion"]):
            pfax[0,0].plot(mom_data_list[i]["z"], mom_data_list[i][species+"M0"][xidx_core,:], color = colors[s])
    pfax[0,0].set_xlabel(r'$\theta$')
    pfax[0,0].set_ylabel(r'$n\, [m^{-3}]$')
    pfax[0,0].legend(handles = handles_species)

    # DENSITY PF Lower
    for i in [0, 9]:
        for s, species in enumerate(["elc", "ion"]):
            pfax[1,2].plot(mom_data_list[i]["z"], mom_data_list[i][species+"M0"][xidx_core,:], color = colors[s])
    pfax[1,2].set_xlabel(r'$\theta$')
    pfax[1,2].set_ylabel(r'$n\, [m^{-3}]$')
    pfax[1,2].legend(handles = handles_species)
    

    
    #PHI 
    for i in range(1,4):
        ax[0,2].plot(mom_data_list[i]["Zavg"][xidx], mom_data_list[i]["phi"][xidx,:], color = 'tab:blue')
        #ax[0,2].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i]["phi"][xidx,:], color = 'tab:blue')

    for i in range(6,9):
        ax[0,2].plot(mom_data_list[i]["Zavg"][xidx], mom_data_list[i]["phi"][xidx,:], linestyle="dashed", color = 'tab:blue')
        #ax[0,2].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i]["phi"][xidx,:], linestyle="dashed", color = 'tab:blue')
    
    ax[0,2].set_xlabel('Z [m]')
    ax[0,2].set_ylabel(r'$\phi\, [V]$')
    ax[0,2].legend(handles = [out_handle, in_handle])


    #PHI PF
    for i in range(4,6):
        pfax[0,1].plot(mom_data_list[i]["z"], mom_data_list[i]["phi"][xidx_core,:], color = 'tab:blue')
    pfax[0,1].set_xlabel(r'$\theta$')
    pfax[0,1].set_ylabel(r'$\phi\, [V]$')

    
    
    # Outboard Temp 
    for i in range(1,4):
        for s, species in enumerate(["elc"]):
            ax[1,1].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Temp"][xidx,:], color = "tab:blue")
            ax[1,1].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Tpar"][xidx,:], color = "tab:orange")
            ax[1,1].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Tperp"][xidx,:], color = "tab:green")
    for i in range(1,4):
        for s, species in enumerate(["ion"]):
            ax[1,1].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Temp"][xidx,:], color = "tab:blue", linestyle="dashed")
            ax[1,1].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Tpar"][xidx,:], color = "tab:orange", linestyle="dashed")
            ax[1,1].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Tperp"][xidx,:], color = "tab:green", linestyle="dashed")

    
    
    ax[1,1].set_xlabel('Z [m]')
    ax[1,1].set_ylabel(r'$T\, [eV]$')
    ax[1,1].legend(handles=temp_handles + [elctemp_handle, iontemp_handle])
    ax[1,1].set_title("Outboard")
    
    # Inboard Temp 
    for i in range(6,9):
        for s, species in enumerate(["elc"]):
            ax[1,0].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Temp"][xidx,:], color = "tab:blue")
            ax[1,0].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Tpar"][xidx,:], color = "tab:orange")
            ax[1,0].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Tperp"][xidx,:], color = "tab:green")
    for i in range(6,9):
        for s, species in enumerate(["ion"]):
            ax[1,0].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Temp"][xidx,:], color = "tab:blue", linestyle="dashed")
            ax[1,0].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Tpar"][xidx,:], color = "tab:orange", linestyle="dashed")
            ax[1,0].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"Tperp"][xidx,:], color = "tab:green", linestyle="dashed")

    
    ax[1,0].set_xlabel('Z [m]')
    ax[1,0].set_ylabel(r'$T\, [eV]$')
    ax[1,0].legend(handles=temp_handles + [elctemp_handle, iontemp_handle])
    ax[1,0].set_title("Inboard")
    

    

    
    # Upar 
    for i in range(1,4):
        for s, species in enumerate(["elc", "ion"]):
            ax[1,2].plot(mom_data_list[i]["Zavg"][xidx], mom_data_list[i][species+"normUpar"][xidx,:], color = colors[s])
            #ax[1,2].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"normUpar"][xidx,:], color = colors[s])
    for i in range(6,9):
        for s, species in enumerate(["elc", "ion"]):
            ax[1,2].plot(mom_data_list[i]["Zavg"][xidx], mom_data_list[i][species+"normUpar"][xidx,:], color = colors[s], linestyle="dashed")
            #ax[1,2].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"normUpar"][xidx,:], color = colors[s], linestyle="dashed")
    
    ax[1,2].set_xlabel('Z [m]')
    ax[1,2].set_ylabel(r'$u_\parallel/c_s$')
    ax[1,2].legend(handles = handles_species + [out_handle , in_handle])

    # Upar PF
    for i in range(4,6):
        for s, species in enumerate(["elc", "ion"]):
            pfax[0,3].plot(mom_data_list[i]["z"], mom_data_list[i][species+"normUpar"][xidx_core,:], color = colors[s])
            #pfax[0,3].plot(raw_mom_data_list[i]["z"], raw_mom_data_list[i][species+"normUpar"][xidx_core,:], color = colors[s])
    
    pfax[0,3].set_xlabel(r'$\theta$')
    pfax[0,3].set_ylabel(r'$u_\parallel/c_s$')
    pfax[0,3].legend(handles = handles_species )

    # M1
    for i in range(1,4):
        #for s, species in enumerate(["elc"]):
        for s, species in enumerate(["elc", "ion"]):
            ax[1,3].plot(mom_data_list[i]["Zavg"][xidx], mom_data_list[i][species+"M1"][xidx,:], color = colors[s])
            #ax[1,3].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"M1"][xidx,:], color = colors[s])
    for i in range(6,9):
        for s, species in enumerate(["elc", "ion"]):
            ax[1,3].plot(mom_data_list[i]["Zavg"][xidx], mom_data_list[i][species+"M1"][xidx,:], color = colors[s], linestyle="dashed")
            #ax[1,3].plot(raw_mom_data_list[i]["Zavg"][xidx], raw_mom_data_list[i][species+"M1"][xidx,:], color = colors[s], linestyle="dashed")
    
    ax[1,3].set_xlabel('Z [m]')
    ax[1,3].set_ylabel(r'$\Gamma\, [m^{-2}s^{-1}]$')
    ax[1,3].legend(handles = handles_species + [out_handle , in_handle])
    
    
    
    #Parallel Core
    # DENSITY
    for i in range(10,12):
        for s, species in enumerate(["elc", "ion"]):
            #ax[0,1].plot(mom_data_list[i]["z"], mom_data_list[i][species+"M0"][xidx_core,:], color = colors[s])
            ax[0,1].plot(raw_mom_data_list[i]["z"], raw_mom_data_list[i][species+"M0"][xidx_core,:], color = colors[s])
    
    ax[0,1].set_xlabel(r'$\theta$')
    ax[0,1].set_ylabel(r'$n\, [m^{-3}]$')
    ax[0,1].set_title("Core")
    ax[0,1].legend(handles = handles_species)
    

    
    #PHI 
    for i in range(10,12):
        ax[0,3].plot(mom_data_list[i]["z"], mom_data_list[i]["phi"][xidx_core,:], color = 'tab:blue')
        #ax[0,3].plot(raw_mom_data_list[i]["z"], raw_mom_data_list[i]["phi"][xidx_core,:], color = 'tab:blue')
    
    ax[0,3].set_xlabel(r'$\theta$')
    ax[0,3].set_ylabel(r'$\phi\, [V]$')
    ax[0,3].set_title("Core")



    pffig.suptitle(sim_name + ", frame = %d"%frame + ", t = %1.6f ms"%(mom_data_list[0]["time"]/1e-3))
    pffig.tight_layout()

    fig.suptitle(sim_name + ", frame = %d"%frame + ", t = %1.6f ms"%(mom_data_list[0]["time"]/1e-3))
    fig.tight_layout()
    fig.savefig('movies/parplot_%03d'%frame);
    plt.show()
    
    
    # Radial
    psifig,psiax = plt.subplots(nrows=2, ncols=4, figsize=(16,9))
    
    # Density at midplane
    for i in [7, 11]:
        for s, species in enumerate(["elc", "ion"]):
            psiax[0,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"M0"][:,zidx[i]], color = colors[s])
    
    psiax[0,0].set_xlabel('R [m]')
    psiax[0,0].set_ylabel(r'$n\, [m^{-3}]$')
    psiax[0,0].set_title("At inboard midplane")
    psiax[0,0].legend(handles = handles_species)
    for i in [2, 10]:
        for s, species in enumerate(["elc", "ion"]):
            psiax[0,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"M0"][:,zidx[i]], color = colors[s])
    
    psiax[0,1].set_xlabel('R [m]')
    psiax[0,1].set_ylabel(r'$n\, [m^{-3}]$')
    psiax[0,1].set_title("At outboard midplane")
    psiax[0,1].legend(handles = handles_species)

    # Potential at midplane
    #for i in [7, 11]:
    #    psiax[1,2].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i]["phi"][:,zidx[i]], color = colors[s])
    #
    #psiax[1,2].set_xlabel('R [m]')
    #psiax[1,2].set_ylabel(r'$\phi\, [V]$')
    #psiax[1,2].set_title("At inboard midplane")
    #psiax[1,2].legend(handles = handles_species)
    #for i in [2, 10]:
    #    psiax[1,3].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i]["phi"][:,zidx[i]], color = colors[s])
    #
    #psiax[1,3].set_xlabel('R [m]')
    #psiax[1,3].set_ylabel(r'$\phi\, [V]$')
    #psiax[1,3].set_title("At outboard midplane")
    #psiax[1,3].legend(handles = handles_species)

    # Potential at plate vs flux
    #for i in [8, 9]:
    #    psiax[0,2].plot(mom_data_list[i]["x"], mom_data_list[i]["phi"][:,-1],color = colors[s])
    #
    #psiax[0,2].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,2].set_ylabel(r'$\phi\, [V]$')
    #psiax[0,2].set_title("At inboard plate")
    #psiax[0,2].legend(handles = handles_species)
    #for i in [3, 4]:
    #    psiax[0,3].plot(mom_data_list[i]["x"], mom_data_list[i]["phi"][:,-1],color = colors[s])
    #
    #psiax[0,3].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,3].set_ylabel(r'$\phi\, [V]$')
    #psiax[0,3].set_title("At outboard plate")
    #psiax[0,3].legend(handles = handles_species)



    
    # Density at plate
    #for i in [0]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,1].plot(mom_data_list[i]["Ravg"][:,0], mom_data_list[i][species+"M0"][:,0], linestyles[i], color = colors[s])
    #
    #psiax[0,1].set_xlabel('R [m]')
    #psiax[0,1].set_ylabel(r'$n\, [m^{-3}]$')
    #psiax[0,1].set_title("At plate")
    #psiax[0,1].legend(handles = handles_species)
    
    # Density at Xpt 
    #for i in [1]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,2].plot(mom_data_list[i]["Ravg"][:,zidx_xpt[i]], mom_data_list[i][species+"M0"][:,zidx_xpt[i]], linestyles[i], color = colors[s])
    #
    #psiax[0,2].set_xlabel('R [m]')
    #psiax[0,2].set_ylabel(r'$n\, [m^{-3}]$')
    #psiax[0,2].set_title("At xpt")
    #psiax[0,2].legend(handles = handles_species)
    
    # Potential at outboard Plate mapped upstream
    #for i in [3]:
    #        psiax[1,3].plot(Rmid_out, mom_data_list[i]["phi"][:,-1])
    #
    #psiax[1,3].set_xlabel('R [m] mapped upstream')
    #psiax[1,3].set_ylabel(r'$\phi\, [V]$')
    #psiax[1,3].set_title("At plate")
    #psiax[1,3].legend(handles = handles)

    # Potential at inboard Plate mapped upstream
    #for i in [8]:
    #        psiax[0,3].plot(Rmid_in, mom_data_list[i]["phi"][:,-1])
    #
    #psiax[0,3].set_xlabel('R [m] mapped upstream')
    #psiax[0,3].set_ylabel(r'$\phi\, [V]$')
    #psiax[0,3].set_title("At plate")
    #psiax[0,3].legend(handles = handles)

    ## Potential at Xpt 
    #for i in [1]:
    #        psiax[1,3].plot(mom_data_list[i]["Ravg"][:,zidx_xpt[i]], mom_data_list[i]["phi"][:,zidx_xpt[i]], linestyles[i])
    #
    #psiax[1,3].set_xlabel('R [m]')
    #psiax[1,3].set_ylabel(r'$\phi\, [V]$')
    #psiax[1,3].set_title("At xpt")
    #psiax[1,3].legend(handles = handles)
    
    
    # Temp at outboard midplane
    for i in [2,10]:
        for s, species in enumerate(["elc"]):
            psiax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0])
            psiax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1])
            psiax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2])
    for i in [2,10]:
        for s, species in enumerate(["ion"]):
            psiax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0], linestyle="dashed")
            psiax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1], linestyle="dashed")
            psiax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2], linestyle="dashed")

    
    psiax[1,1].set_xlabel('R [m]')
    psiax[1,1].set_ylabel(r'$T \, [eV]$')
    psiax[1,1].set_title("At outboard midplane")
    psiax[1,1].legend(handles = temp_handles + [elctemp_handle, iontemp_handle])

    # Temp at inboard midplane
    for i in [7,11]:
        for s, species in enumerate(["elc"]):
            psiax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0])
            psiax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1])
            psiax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2])
    for i in [7,11]:
        for s, species in enumerate(["ion"]):
            psiax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0], linestyle="dashed")
            psiax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1], linestyle="dashed")
            psiax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2], linestyle="dashed")
    
    psiax[1,0].set_xlabel('R [m]')
    psiax[1,0].set_ylabel(r'$T \, [eV]$')
    psiax[1,0].set_title("At inboard midplane")
    psiax[1,0].legend(handles = temp_handles + [elctemp_handle, iontemp_handle])

    # Source Temp at outboard midplane
    for i in [10]:
        for s, species in enumerate(["elc"]):
            pfax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0])
            pfax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1])
            pfax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2])
    for i in [10]:
        for s, species in enumerate(["ion"]):
            pfax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0], linestyle="dashed")
            pfax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1], linestyle="dashed")
            pfax[1,1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2], linestyle="dashed")
    
    pfax[1,1].set_xlabel('R [m]')
    pfax[1,1].set_ylabel(r'$T \, [eV]$')
    pfax[1,1].set_title("At outboard midplane")
    pfax[1,1].legend(handles = temp_handles + [elctemp_handle, iontemp_handle])

    # Source Temp at inboard midplane
    for i in [11]:
        for s, species in enumerate(["elc"]):
            pfax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0])
            pfax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1])
            pfax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2])
    for i in [11]:
        for s, species in enumerate(["ion"]):
            pfax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0], linestyle="dashed")
            pfax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1], linestyle="dashed")
            pfax[1,0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], source_mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2], linestyle="dashed")
    
    pfax[1,0].set_xlabel('R [m]')
    pfax[1,0].set_ylabel(r'$T \, [eV]$')
    pfax[1,0].set_title("At inboard midplane")
    pfax[1,0].legend(handles = temp_handles + [elctemp_handle, iontemp_handle])
    
    # Elc Temp at plate
    # mapped upstream
    #for i in [2]:
    #    for s, species in enumerate(["elc"]):
    #        psiax[0,2].plot(mom_data_list[1]["Ravg"][:,zidx[1]], mom_data_list[i][species+"Temp"][:,-1], label = r'$T$')
    #        psiax[0,2].plot(mom_data_list[1]["Ravg"][:,zidx[1]], mom_data_list[i][species+"Tpar"][:,-1], label = r'$T_\parallel$')
    #        psiax[0,2].plot(mom_data_list[1]["Ravg"][:,zidx[1]], mom_data_list[i][species+"Tperp"][:,-1], label = r'$T_\perp$')
    #
    #psiax[0,2].set_xlabel('R [m] mapped upstream')
    #psiax[0,2].set_ylabel(r'$T_e\, [eV]$')
    #psiax[0,2].set_title("At plate")
    #psiax[0,2].legend()

    # M1 at plate vs flux
    #for i in [8, 9]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,2].plot(mom_data_list[i]["x"], mom_data_list[i][species+"M1"][:,-1],color = colors[s])
    #        #psiax[0,2].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"M1"][:,-1],color = colors[s])
    #
    #psiax[0,2].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,2].set_ylabel(r'$\Gamma \,\, [1/m^2s]$')
    #psiax[0,2].set_title("At inboard plate")
    #psiax[0,2].legend(handles = handles_species)
    #for i in [3, 4]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,3].plot(mom_data_list[i]["x"], mom_data_list[i][species+"M1"][:,-1],color = colors[s])
    #        #psiax[0,3].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"M1"][:,-1],color = colors[s])
    #
    #psiax[0,3].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,3].set_ylabel(r'$\Gamma \,\, [1/m^2s]$')
    #psiax[0,3].set_title("At outboard plate")
    #psiax[0,3].legend(handles = handles_species)

    # Poloidal M1 at plate vs flux
    #for i in [8, 9]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,2].plot(mom_data_list[i]["x"], mom_data_list[i][species+"M1"][:,-1]*np.sin(mom_data_list[i]["Bratio"]), color = colors[s])
    #        #psiax[0,2].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"M1"][:,-1],color = colors[s])
    #
    #psiax[0,2].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,2].set_ylabel(r'$\Gamma \,\, [1/m^2s]$')
    #psiax[0,2].set_title("At inboard plate")
    #psiax[0,2].legend(handles = handles_species)
    #for i in [3, 4]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,3].plot(mom_data_list[i]["x"], mom_data_list[i][species+"M1"][:,-1]*np.sin(mom_data_list[i]["Bratio"]), color = colors[s])
    #        #psiax[0,3].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"M1"][:,-1],color = colors[s])
    #
    #psiax[0,3].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,3].set_ylabel(r'$\Gamma \,\, [1/m^2s]$')
    #psiax[0,3].set_title("At outboard plate")
    #psiax[0,3].legend(handles = handles_species)

    # Density at plate vs flux
    #for i in [8, 9]:
    #    for s, species in enumerate(["elc", "ion", "H0"]):
    #        psiax[0,2].plot(mom_data_list[i]["x"], mom_data_list[i][species+"M0"][:,-1],color = colors[s])
    #        #psiax[0,2].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    #
    #psiax[0,2].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,2].set_ylabel(r'$n\, [1/m^3]$')
    #psiax[0,2].set_title("At inboard plate")
    #psiax[0,2].legend(handles = handles_species)
    #for i in [1, 0]:
    #    for s, species in enumerate(["elc", "ion", "H0"]):
    #        psiax[0,3].plot(mom_data_list[i]["x"], mom_data_list[i][species+"M0"][:,-1],color = colors[s])
    #        #psiax[0,3].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    #
    #psiax[0,3].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,3].set_ylabel(r'$n\, [1/m^3]$')
    #psiax[0,3].set_title("At outboard plate")
    #psiax[0,3].legend(handles = handles_species)

    # Phi at plate vs flux
    #for i in [8, 9]:
    #    psiax[0,2].plot(mom_data_list[i]["x"], mom_data_list[i]["phi"][:,-1],color = colors[s])
    #
    #psiax[0,2].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,2].set_ylabel(r'$\phi\, [eV]$')
    #psiax[0,2].set_title("At inboard plate")
    #for i in [1, 0]:
    #    psiax[0,3].plot(mom_data_list[i]["x"], mom_data_list[i]["phi"][:,-1],color = colors[s])
    #
    #psiax[0,3].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,3].set_ylabel(r'$\phi\, [eV]$')
    #psiax[0,3].set_title("At outboard plate")

    # Temp at plate vs flux
    #for i in [8, 9]:
    #    for s, species in enumerate(["elc", "ion", "H0"]):
    #        psiax[0,2].plot(mom_data_list[i]["x"], mom_data_list[i][species+"Temp"][:,-1],color = colors[s])
    #        #psiax[0,2].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    #
    #psiax[0,2].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,2].set_ylabel(r'$T\, [eV]$')
    #psiax[0,2].set_title("At inboard plate")
    #psiax[0,2].legend(handles = handles_species)
    #psiax[0,2].set_ylim(bottom = 0.0)
    #for i in [3, 4]:
    #    for s, species in enumerate(["elc", "ion", "H0"]):
    #        psiax[0,3].plot(mom_data_list[i]["x"], mom_data_list[i][species+"Temp"][:,-1],color = colors[s])
    #        #psiax[0,3].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    #
    #psiax[0,3].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,3].set_ylabel(r'$T\, [eV]$')
    #psiax[0,3].set_title("At outboard plate")
    #psiax[0,3].legend(handles = handles_species)
    #psiax[0,3].set_ylim(bottom = 0.0)

    # Heat Flux at plate vs flux
    #for i in [8, 9]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,2].plot(mom_data_list[i]["x"], mom_data_list[i][species+"Q"][:,-1]/1e6,color = colors[s])
    #        #psiax[0,2].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    #
    #psiax[0,2].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,2].set_ylabel(r'$Q\, [MW/m^2]$')
    #psiax[0,2].set_title("At inboard plate")
    #psiax[0,2].legend(handles = handles_species)
    #for i in [3, 4]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,3].plot(mom_data_list[i]["x"], mom_data_list[i][species+"Q"][:,-1]/1e6,color = colors[s])
    #        #psiax[0,3].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    #
    #psiax[0,3].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    #psiax[0,3].set_ylabel(r'$Q\, [MW/m^2]$')
    #psiax[0,3].set_title("At outboard plate")
    #psiax[0,3].legend(handles = handles_species)


    #Poloidal Heat Flux at plate vs flux
    for i in [8, 9]:
        for s, species in enumerate(["elc", "ion"]):
            psiax[0,2].plot(mom_data_list[i]["x"], mom_data_list[i][species+"Qwall"][:,-1]/1e6*np.sin(mom_data_list[i]["Bratio"]),color = colors[s])
            #psiax[0,2].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    
    psiax[0,2].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    psiax[0,2].set_ylabel(r'$Q_{pol}\, [MW/m^2]$')
    psiax[0,2].set_title("At inboard plate")
    psiax[0,2].legend(handles = handles_species)
    for i in [3, 4]:
        for s, species in enumerate(["elc", "ion"]):
            psiax[0,3].plot(mom_data_list[i]["x"], mom_data_list[i][species+"Qwall"][:,-1]/1e6*np.sin(mom_data_list[i]["Bratio"]),color = colors[s])
            #psiax[0,3].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    
    psiax[0,3].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    psiax[0,3].set_ylabel(r'$Q_{pol}\, [MW/m^2]$')
    psiax[0,3].set_title("At outboard plate")
    psiax[0,3].legend(handles = handles_species)

    #Poloidal Heat Flux at plate vs physical coordinate
    #for i in [8, 9]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,2].plot(mom_data_list[i]["Zperpavg"][:,-1], mom_data_list[i][species+"Q"][:,-1]/1e6*np.sin(mom_data_list[i]["Bratio"]),color = colors[s])
    #        #psiax[0,2].plot(raw_mom_data_list[i]["x"], raw_mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    #
    #psiax[0,2].set_xlabel(r'$Z\,\, [m]$')
    #psiax[0,2].set_ylabel(r'$Q_{pol}\, [MW/m^2]$')
    #psiax[0,2].set_title("At inboard plate")
    #psiax[0,2].legend(handles = handles_species)
    #for i in [3, 4]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        if i == 4:
    #            psiax[0,3].plot(mom_data_list[i]["Ravg"][:,-1], mom_data_list[i][species+"Q"][:,-1]/1e6*np.sin(mom_data_list[i]["Bratio"]),color = colors[s])
    #        if i == 3:
    #            psiax[0,3].plot(mom_data_list[i]["Zperpavg"][:,-1], mom_data_list[i][species+"Q"][:,-1]/1e6*np.sin(mom_data_list[i]["Bratio"]),color = colors[s])
    #
    #psiax[0,3].set_xlabel(r'$R\,\, [m]$')
    #psiax[0,3].set_ylabel(r'$Q_{pol}\, [MW/m^2]$')
    #psiax[0,3].set_title("At outboard plate")
    #psiax[0,3].legend(handles = handles_species)


    # M1 at plate
    #for i in [8]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,2].plot(Rmid_in, mom_data_list[i][species+"M1"][:,-1],color = colors[s])
    #
    #psiax[0,2].set_xlabel('R [m] (mapped upstream)')
    #psiax[0,2].set_ylabel(r'$\Gamma \,\, [1/m^2s]$')
    #psiax[0,2].set_title("At inboard plate")
    #psiax[0,2].legend(handles = handles_species)
    #for i in [3]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,3].plot(Rmid_out, mom_data_list[i][species+"M1"][:,-1],color = colors[s])
    #
    #psiax[0,3].set_xlabel('R [m] (mapped upstream)')
    #psiax[0,3].set_ylabel(r'$\Gamma \,\, [1/m^2s]$')
    #psiax[0,3].set_title("At outboard plate")
    #psiax[0,3].legend(handles = handles_species)

    # Heat Flux at plate
    #for i in [8]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,2].plot(Rmid_in, mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    #
    #psiax[0,2].set_xlabel('R [m] (mapped upstream)')
    #psiax[0,2].set_ylabel(r'$Q\, [MW/m^2]$')
    #psiax[0,2].set_title("At inboard plate")
    #psiax[0,2].legend(handles = handles_species)
    #for i in [3]:
    #    for s, species in enumerate(["elc", "ion"]):
    #        psiax[0,3].plot(Rmid_out, mom_data_list[i][species+"Q"][:,-1],color = colors[s])
    #
    #psiax[0,3].set_xlabel('R [m] (mapped upstream)')
    #psiax[0,3].set_ylabel(r'$Q\, [MW/m^2]$')
    #psiax[0,3].set_title("At outboard plate")
    #psiax[0,3].legend(handles = handles_species)

    #Normalized electron density, temp  at midplane
    #for i in [2,10]:
    #    for s, species in enumerate(["elc"]):
    #        psiax[1,2].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"M0"][:,zidx[i]]/mom_data_list[10][species+"M0"][:,zidx[10]].max(), color = 'tab:blue', label = r'$n_e/n_{e,max}$') 
    #        psiax[1,2].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]]/mom_data_list[10][species+"Temp"][:,zidx[10]].max(), color = 'tab:orange', label = r'$T_e/T_{e,max}$') 
    #
    #psiax[1,2].set_xlabel('R [m]')
    #psiax[1,2].set_ylabel('Electron Normalized Quantities')
    #psiax[1,2].set_title("At midplane")
    #psiax[1,2].legend()

    #Normalized electron density, temp, and heat flux at outboard plate mapped upstream
    for i in [3]:
        for s, species in enumerate(["elc"]):
            psiax[1,3].plot(Rmid_out, mom_data_list[i][species+"M0"][:,-1]/mom_data_list[i][species+"M0"][:,-1].max(), color = 'tab:blue', label = r'$n_e/n_{e,max}$') 
            psiax[1,3].plot(Rmid_out, mom_data_list[i][species+"Temp"][:,-1]/mom_data_list[i][species+"Temp"][:,-1].max(), color = 'tab:orange', label = r'$T_e/T_{e,max}$') 
            psiax[1,3].plot(Rmid_out, mom_data_list[i][species+"M1"][:,-1]/mom_data_list[i][species+"M1"][:,-1].max(), color = 'tab:green', label = r'$\Gamma_e/\Gamma_{e,max}$') 
            psiax[1,3].plot(Rmid_out, mom_data_list[i][species+"Q"][:,-1]/mom_data_list[i][species+"Q"][:,-1].max(), color = 'k', label = r'$Q_e/Q_{e,max}$') 
            psiax[1,3].plot(Rmid_out, mom_data_list[i]["Qtot"][:,-1]/mom_data_list[i]["Qtot"][:,-1].max(), color = 'k', linestyle="dashed", label = r'$Q/Q_{max}$') 
            psiax[1,3].plot(Rmid_out, mom_data_list[i]["phi"][:,-1]/mom_data_list[i]["phi"][:,-1].max(), color = 'tab:purple', label = r'$\phi/\phi_{max}$') 
            psiax[1,3].plot(Rmid_out, mom_data_list[i]["ion"+"M1"][:,-1]/mom_data_list[i]["ion"+"M1"][:,-1].max(), color = 'tab:brown', label = r'$\Gamma_i/\Gamma_{i,max}$') 
            psiax[1,3].plot(Rmid_out, mom_data_list[i]["ionQ"][:,-1]/mom_data_list[i]["ionQ"][:,-1].max(), color = 'tab:cyan', label = r'$Q_i/Q_{i,max}$') 
    
    psiax[1,3].set_xlabel('R [m] mapped upstream')
    psiax[1,3].set_ylabel('Electron Normalized Quantities')
    psiax[1,3].set_title("At outboard plate")
    psiax[1,3].legend()

    ##Normalized electron density, temp, and heat flux at inboard plate mapped upstream
    for i in [8]:
        for s, species in enumerate(["elc"]):
            psiax[1,2].plot(Rmid_in, mom_data_list[i][species+"M0"][:,-1]/mom_data_list[i][species+"M0"][:,-1].max(), color = 'tab:blue', label = r'$n_e/n_{e,max}$') 
            psiax[1,2].plot(Rmid_in, mom_data_list[i][species+"Temp"][:,-1]/mom_data_list[i][species+"Temp"][:,-1].max(), color = 'tab:orange', label = r'$T_e/T_{e,max}$') 
            psiax[1,2].plot(Rmid_in, mom_data_list[i][species+"M1"][:,-1]/mom_data_list[i][species+"M1"][:,-1].max(), color = 'tab:green', label = r'$\Gamma_e/\Gamma_{e,max}$') 
            psiax[1,2].plot(Rmid_in, mom_data_list[i][species+"Q"][:,-1]/mom_data_list[i][species+"Q"][:,-1].max(), color = 'k', label = r'$Q_e/Q_{e,max}$') 
            psiax[1,2].plot(Rmid_in, mom_data_list[i]["Qtot"][:,-1]/mom_data_list[i]["Qtot"][:,-1].max(), color = 'k', linestyle="dashed", label = r'$Q/Q_{max}$') 
            psiax[1,2].plot(Rmid_in, mom_data_list[i]["phi"][:,-1]/mom_data_list[i]["phi"][:,-1].max(), color = 'tab:purple', label = r'$\phi/\phi_{max}$') 
            psiax[1,2].plot(Rmid_in, mom_data_list[i]["ion"+"M1"][:,-1]/mom_data_list[i]["ion"+"M1"][:,-1].max(), color = 'tab:brown', label = r'$\Gamma_i/\Gamma_{i,max}$') 
            psiax[1,2].plot(Rmid_in, mom_data_list[i]["ionQ"][:,-1]/mom_data_list[i]["ionQ"][:,-1].max(), color = 'tab:cyan', label = r'$Q_i/Q_{i,max}$') 
    
    psiax[1,2].set_xlabel('R [m] mapped upstream')
    psiax[1,2].set_ylabel('Electron Normalized Quantities')
    psiax[1,2].set_title("At inboard plate")
    psiax[1,2].legend()

    
    psifig.suptitle(sim_name + ", frame = %d"%frame + ", t = %1.6f ms"%(mom_data_list[0]["time"]/1e-3))
    psifig.tight_layout()
    psifig.savefig('movies/radplot_%03d'%frame);
    plt.show()

    #plt.close('all')

ni= mom_data_list[2]["ionM0"][0,zidx[2]]
ne= mom_data_list[2]["elcM0"][0,zidx[2]]
Ti= mom_data_list[2]["ionTemp"][0,zidx[2]]
Te= mom_data_list[2]["elcTemp"][0,zidx[2]]
clogi=30 - np.log(np.sqrt(ni)/Ti**1.5)
cloge=31.3 - np.log(np.sqrt(ne)/Te**1.5)
R=1
a=0.5
invA=a/R
q=10
nustari = 4.9e-18*q*R*ni*clogi/(Ti**2*invA**1.5)
nustare = 6.921e-18*q*R*ne*cloge/(Te**2*invA**1.5)
print("nustari = ", nustari, "nustare=", nustare)


#Calculate ion mfp and lt
#Ions
# mfp
nuFrac = 0.25
n0 = 2.0e19
Ti0 = 1000 *  eV
logLambdaIon = 6.6 - 0.5*np.log(n0/1e20) + 1.5*np.log(Ti0/eV)
nuIon = nuFrac*logLambdaIon*(eV**4)*n0/(12*np.pi**(3/2)*(eps0**2)*np.sqrt(2.014*mp)*(Ti0)**(3/2))
nuIon = nuIon/nuFrac # remove the nuFrac so we can do the target coll freq
vti0 = np.sqrt(Ti0/masses["ion"])
normNu = nuIon/n0 * (2*vti0**2)**1.5
vti = np.sqrt(mom_data_list[2]['ionTemp']*eV/masses['ion'])
nuIon = normNu * mom_data_list[2]['ionM0']/(2*vti**2)**1.5
lambda_mfp_i = vti/nuIon
##Lt
#geo_fac = 1/mom_data_list[1]["J"]/mom_data_list[1]["B"]
#L_ti = (mom_data_list[1]['ionTemp'][xidx,:]*eV/(geo_fac[xidx,:]*np.gradient(mom_data_list[1]['ionTemp'][xidx,:]*eV, mom_data_list[1]["z"], edge_order=2))).clip(-20,20) #Not sure about Lc for hl2a
#mfp_norm_i = np.abs(lambda_mfp_i[xidx,:]/L_ti)
#figmfp,axmfp = plt.subplots(1)
#axmfp.plot(mom_data_list[1]["z"],mfp_norm_i)
#axmfp.set_xlabel(r'Normalized Arc Length')
#axmfp.set_ylabel(r'$\lambda_{mfp,i}/L_{T,i} (\psi = %1.3f)$'%x[xidx])
#axmfp.set_title("Normalized Deuterium Mean Free Path Along Field Line")
#plt.show()

#Lc is 26m at psi = -0.10

#figmfp,axmfp = plt.subplots(1)
#axmfp.plot(mom_data_list[2]["z"],lambda_mfp_i[xidx]/vti[xidx])
#axmfp.set_xlabel(r'Normalized Arc Length')
#axmfp.set_ylabel(r'$\tau_i (\psi = %1.3f)$'%mom_data_list[2]["x"][xidx])
#axmfp.set_title("Ion  collision time Along Field Line")
#plt.show()

#Calculate heat flux width
qmid_out = mom_data_list[3]["Qtot"][:, -1]
qmid_out_cum = sci.cumulative_trapezoid(qmid_out,x=Rmid_out, initial=0)
xcut = np.argwhere(qmid_out_cum/qmid_out_cum.max() > 0.63)[0]
width = (Rmid_out[xcut] - Rmid_out[0])
print("Rcut = ", Rmid_out[xcut])
print("Outboard heat flux width = %g mm"%(width[0]/1e-3))

qmid_in = mom_data_list[8]["Qtot"][:, -1]
qmid_in_cum = -sci.cumulative_trapezoid(qmid_in,x=Rmid_in, initial=0)
xcut = np.argwhere(qmid_in_cum/qmid_in_cum.max() > 0.63)[0]
width = np.abs(Rmid_in[xcut] - Rmid_in[0])
print("Rcut = ", Rmid_in[xcut])
print("Inboard heat flux width = %g mm"%(width[0]/1e-3))


#Calculate actual width
Rmid = mom_data_list[2]["Ravg"][:, zidx[2]]
nimid = mom_data_list[2]["ionM0"][:, zidx[2]]
nimid_cum = sci.cumulative_trapezoid(nimid,x=Rmid, initial=0)
#print(nimid_cum/nimid_cum.max())
xcut = np.argwhere(nimid_cum/nimid_cum.max() > 0.63)[0]
width = (Rmid[xcut] - Rmid[0])

#Caclulate expected width
Lc = 26 # 26m from midplane to plate
t_transit = Lc/vti[xidx,zidx[2]]
t_coll = (lambda_mfp_i/vti)[xidx,zidx[2]]
l_transit = (D*t_transit)**0.5
l_coll = (D*t_coll)**0.5

#Calculate ratio of B and then trapped fraction
Bmax = mom_data_list[2]["B"][xidx,-1]
Bmin = mom_data_list[2]["B"][xidx,zidx[2]]
frac_trap = np.sqrt(1 - 1/(Bmax/Bmin))
frac_pass = 1 - frac_trap

#Use a  weighted sum to get an expected width
Rtest = np.linspace(0,21e-3,100)
y_trap = frac_trap*np.exp(-Rtest/l_coll)
y_pass = frac_pass*np.exp(-Rtest/l_transit)
y_tot = y_trap+y_pass
ycum = sci.cumulative_trapezoid(y_tot, x=Rtest, initial = 0)
exp_xcut = np.argwhere(ycum/ycum.max() > 0.63)[0]
exp_width = (Rtest[exp_xcut] - Rtest[0])

print("width  = %g mm, expected width = %g mm"%(width[0]/1e-3, exp_width[0]/1e-3))

#nfig,nax = plt.subplots(1)
#n0i = mom_data_list[2]["ionM0"][xidx,zidx[2]]
#exp_ni = n0i*frac_pass*np.exp(-(Rmid-Rmid[0])/l_transit) + n0i*frac_trap*np.exp(-(Rmid-Rmid[0])/l_coll)
#nax.plot(Rmid, mom_data_list[2]["ionM0"][:, zidx[2]], label = "actual")
#nax.plot(Rmid, exp_ni, label = "two decay lengths")
#nax.legend()
#nax.set_xlabel("R [m]")
#nax.set_ylabel(r'$n_i\, [m^{-3}]$')

#Plot scale lengths

# Electron scale lengths at midplane
#Lfig, Lax = plt.subplots(1,3, figsize=(16,9))
#for i in [2]:
#    for s, species in enumerate(["elc"]):
#        Lax[0].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"M0_L"], label = r'$L_n = |n/(dn/dR)|$')
#        Lax[1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Temp_L"], label = r'$L_T = |T/(dT/dR)|$')
#        Lax[2].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Tpar_L"], label = r'$L_{T\parallel} = |T_\parallel/(dT_\parallel/dR)|$')
#        Lax[1].plot(mom_data_list[i]["Ravg"][:,zidx[i]], mom_data_list[i][species+"Tperp_L"], label = r'$L_{T\perp} = |T_\perp/(dT_\perp/dR)|$')
#
#Lax[0].set_xlabel('R [m]')
#Lax[1].set_xlabel('R [m]')
#Lax[2].set_xlabel('R [m]')
#Lax[0].set_ylabel('Electron Density Scale Lengths L [m]')
#Lax[1].set_ylabel('Electron Temperature Scale Lengths L [m]')
#Lax[2].set_ylabel('Electron Temperature Scale Lengths L [m]')
#Lax[0].set_title("At midplane")
#Lax[1].set_title("At midplane")
#Lax[2].set_title("At midplane")
#Lax[0].legend()
#Lax[1].legend()
#Lax[2].legend()
#Lfig.tight_layout()
#Lfig.savefig('movies/scaleplot_%03d'%frame);
