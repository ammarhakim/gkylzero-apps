import numpy as np
import postgkyl as pg

eV=1.602e-19
# Load Jonathan's data on a scattered R,Z grid
hdata = np.genfromtxt("eirene_text_output/hl2a_data_for_akash_upar.txt", skip_header=1)
edata = np.genfromtxt("eirene_text_output/hl2a_data_for_akash_XYZ.txt", skip_header=1)

frame = int(np.genfromtxt("gkeyll_text_output/new_data_flag"))

# Step 1: load the Gkeyll grid information
baseName = 'h11'
bmin = 0
bmax = 12
simNames = ['%s_b%d'%(baseName,i) for i in range(bmin,bmax)]
Rlist = []
Zlist = []
nodal_grid_list = []
jlist = []
for i, simName in enumerate(simNames):
    data = pg.GData(simName+"-nodes.gkyl")
    vals = data.get_values()
    R = vals[:,:,0]
    Z = vals[:,:,1]
    phi = vals[:,:,2]
    temp_nodal_grid = data.get_grid()
    # This nodal grid is the true (psi,theta) coords
    nodal_grid = []
    for d in range(0,len(temp_nodal_grid)):
        nodal_grid.append( np.linspace(temp_nodal_grid[d][0], temp_nodal_grid[d][-1], len(temp_nodal_grid[d])-1) )
    
    Rlist.append(R)
    Zlist.append(Z)
    nodal_grid_list.append(nodal_grid)

# Step 2: Fill Nodal Gkeyll data by finding closest point from Eirene
nlist = []
Tlist = []
uxlist = []
uylist = []
uzlist = []
ulist = []
for i, simName in enumerate(simNames):
    nx, nz = Rlist[i].shape
    density = np.zeros((nx,nz))
    temp = np.zeros((nx,nz))
    ux = np.zeros((nx,nz))
    uy = np.zeros((nx,nz))
    uz = np.zeros((nx,nz))
    u = np.zeros((nx,nz,3))
    for ix in range(nx):
        for iz in range(nz):
            lindist = np.sqrt((Rlist[i][ix,iz] - hdata[:,0])**2 + (Zlist[i][ix,iz] - hdata[:,1])**2)
            linidx = np.argmin(lindist)
            density[ix,iz] = hdata[linidx,2]
            temp[ix,iz] = hdata[linidx,4]

            lindist = np.sqrt((Rlist[i][ix,iz] - edata[:,0])**2 + (Zlist[i][ix,iz] - edata[:,1])**2)
            linidx = np.argmin(lindist)
            ux[ix,iz] = edata[linidx,3]
            uy[ix,iz] = edata[linidx,4]
            uz[ix,iz] = edata[linidx,5]

            #Set u to zero for now
            ux[ix,iz] = 0.0
            uy[ix,iz] = 0.0
            uz[ix,iz] = 0.0

            u[ix,iz,:] = np.r_[ux[ix,iz], uy[ix,iz], uz[ix,iz]]

    # Apply a floor
    density[density < 1e8] = 1e8
    temp[temp<0.0] = 50.0*eV;
    nlist.append(density)
    Tlist.append(temp)
    uxlist.append(ux)
    uylist.append(uy)
    uzlist.append(uz)
    ulist.append(u)

# Step 2.1 Symmetrize Data based on belief that lower half is correct
for qlist in [nlist, Tlist]:
    qlist[3] = np.flip(qlist[1], axis = 1)
    qlist[4] = np.flip(qlist[0], axis = 1)

    qlist[6] = np.flip(qlist[8], axis = 1)
    qlist[5] = np.flip(qlist[9], axis = 1)

    qlist[2][:,qlist[2].shape[1]//2+1:] = np.flip(qlist[2][:,0:qlist[2].shape[1]//2], axis=1)
    qlist[10][:,qlist[10].shape[1]//2+1:] = np.flip(qlist[10][:,0:qlist[10].shape[1]//2], axis=1)

    qlist[7][:, 0:qlist[7].shape[1]//2] = np.flip(qlist[7][:, qlist[7].shape[1]//2+1:], axis=1)
    qlist[11][:, 0:qlist[11].shape[1]//2] = np.flip(qlist[11][:, qlist[11].shape[1]//2+1:], axis=1)

for qlist in [ulist]:
    qlist[3][:,:,0:2] = np.flip(qlist[1][:,:,0:2], axis = 1)
    qlist[4][:,:,0:2] = np.flip(qlist[0][:,:,0:2], axis = 1)
    qlist[3][:,:,2] = -np.flip(qlist[1][:,:,2], axis = 1)
    qlist[4][:,:,2] = -np.flip(qlist[0][:,:,2], axis = 1)

    qlist[6][:,:,0:2] = np.flip(qlist[8][:,:,0:2], axis = 1)
    qlist[5][:,:,0:2] = np.flip(qlist[9][:,:,0:2], axis = 1)
    qlist[6][:,:,2] = -np.flip(qlist[8][:,:,2], axis = 1)
    qlist[5][:,:,2] = -np.flip(qlist[9][:,:,2], axis = 1)

    qlist[2][:,qlist[2].shape[1]//2+1:, 0:2] = np.flip(qlist[2][:,0:qlist[2].shape[1]//2, 0:2], axis=1)
    qlist[10][:,qlist[10].shape[1]//2+1:, 0:2] = np.flip(qlist[10][:,0:qlist[10].shape[1]//2, 0:2], axis=1)
    qlist[2][:,qlist[2].shape[1]//2+1:, 2] = -np.flip(qlist[2][:,0:qlist[2].shape[1]//2, 2], axis=1)
    qlist[10][:,qlist[10].shape[1]//2+1:, 2] = -np.flip(qlist[10][:,0:qlist[10].shape[1]//2, 2], axis=1)

    qlist[7][:, 0:qlist[7].shape[1]//2, 0:2] = np.flip(qlist[7][:, qlist[7].shape[1]//2+1:, 0:2], axis=1)
    qlist[11][:, 0:qlist[11].shape[1]//2, 0:2] = np.flip(qlist[11][:, qlist[11].shape[1]//2+1:, 0:2], axis=1)
    qlist[7][:, 0:qlist[7].shape[1]//2, 2] = -np.flip(qlist[7][:, qlist[7].shape[1]//2+1:, 2], axis=1)
    qlist[11][:, 0:qlist[11].shape[1]//2, 2] = -np.flip(qlist[11][:, qlist[11].shape[1]//2+1:, 2], axis=1)


# Step 2.2: Read previous gkeyll data and average
#for i, simName in enumerate(simNames):
#    nlist[i] = (nlist[i] + np.genfromtxt('gkeyll_text_input/'+simName+"-H0_M0.txt").reshape((nlist[i].shape[0], nlist[i].shape[1])))/2.0
#    Tlist[i] = (Tlist[i] + np.genfromtxt('gkeyll_text_input/'+simName+"-H0_Temp.txt").reshape((Tlist[i].shape[0], Tlist[i].shape[1])))/2.0
#    uxlist[i] = (uxlist[i] + np.genfromtxt('gkeyll_text_input/'+simName+"-H0_ux.txt").reshape((uxlist[i].shape[0], uxlist[i].shape[1])))/2.0
#    uylist[i] = (uylist[i] + np.genfromtxt('gkeyll_text_input/'+simName+"-H0_uy.txt").reshape((uylist[i].shape[0], uylist[i].shape[1])))/2.0
#    uzlist[i] = (uzlist[i] + np.genfromtxt('gkeyll_text_input/'+simName+"-H0_uz.txt").reshape((uzlist[i].shape[0], uzlist[i].shape[1])))/2.0


# Limit the inboard density for now
#if frame < 100:
#for i in [6,7,8,5,9]:
#    nlist[i] = nlist[i]/5.0

# Step 3: Write nodal data to text file 
for i, simName in enumerate(simNames):
    np.savetxt('gkeyll_text_input/'+simName+"-H0_M0.txt", nlist[i].flatten())
    np.savetxt('gkeyll_text_input/'+simName+"-H0_Temp.txt", Tlist[i].flatten())
    np.savetxt('gkeyll_text_input/'+simName+"-H0_ux.txt", uxlist[i].flatten())
    np.savetxt('gkeyll_text_input/'+simName+"-H0_uy.txt", uylist[i].flatten())
    np.savetxt('gkeyll_text_input/'+simName+"-H0_uz.txt", uzlist[i].flatten())

            
print("Finished converting text to Gkeyll input")    


