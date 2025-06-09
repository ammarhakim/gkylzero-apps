import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import postgkyl as pg
import scipy.interpolate 
from scipy.interpolate import RegularGridInterpolator


fig, ax  = plt.subplots(1)

psid = pg.GData("hl2a_psi.gkyl")
interp = pg.GInterpModal(psid,2,"mt")
grid, psi = interp.interpolate()
#pg.output.plot(psid, contour=True)

for d in range(len(grid)):
    grid[d] = 0.5*(grid[d][:-1] + grid[d][1:])

psisep =  -0.1257861755585853
psi_min = psisep
psi_max = -0.10289308

plt.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep], colors="r")
#plt.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psi_min], colors="g")
plt.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psi_max], colors="b")


#colors = ["tab:orange","tab:blue","tab:green", "tab:brown", "tab:purple"]
#sim_dir = "./simdata/h7_try2/"
sim_dir = "./"
baseName = sim_dir+'h15'
bmin = 0
bmax = 12
simNames = ['%s_b%d'%(baseName,i) for i in range(bmin,bmax)]
# simNames = ['hl2a']
Rlist = []
Zlist = []
jlist = []
jmax = 0.0
jmin = np.inf
for i, simName in enumerate(simNames):
    data = pg.GData(simName+"-nodes.gkyl")
    vals = data.get_values()
    R = vals[:,:,0]
    Z = vals[:,:,1]
    phi = vals[:,:,2]

    temp_nodal_grid = data.get_grid()
    nodal_grid = []
    for d in range(0,len(temp_nodal_grid),2):
        nodal_grid.append( np.linspace(temp_nodal_grid[d][0], temp_nodal_grid[d][-1], len(temp_nodal_grid[d])-1) )



    
    
    #Plot all nodes and cell boundaries
    plt.plot(R,Z,marker=".", color="k", linestyle="none")
    plt.scatter(R,Z, marker=".")
    segs1 = np.stack((R,Z), axis=2)
    segs2 = segs1.transpose(1,0,2)
    #plt.gca().add_collection(LineCollection(segs1, color = colors[i]))
    #plt.gca().add_collection(LineCollection(segs2, color = colors[i]))
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))
    #plt.legend()

    #segs1 = np.stack((R,Z), axis=2)
    #segs2 = segs1.transpose(1,0,2)
    ##Plot side boundaries
    #plt.plot(segs1[0][:,0], segs1[0][:,1], color = 'tab:blue', label = "Simulation Domain")
    #plt.plot(segs1[-1][:,0], segs1[-1][:,1], color = 'tab:blue')
    ##plot plates
    #plt.plot(segs2[0][:,0], segs2[0][:,1], color='tab:orange', label = "Divertor Plate", linewidth=1)
    #plt.plot(segs2[-1][:,0], segs2[-1][:,1], color='tab:orange', linewidth=1)
    Rlist.append(R)
    Zlist.append(Z)





    


plt.grid()
#plt.axis("tight")
plt.xlabel("R [m]")
plt.ylabel("Z [m]")
#plt.axis("image")
ax.set_aspect("equal")
plt.show()
plt.savefig("ehl2_domain.png")
