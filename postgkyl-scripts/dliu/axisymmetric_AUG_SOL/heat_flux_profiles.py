import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.transforms import ScaledTranslation


# Load GK data
root = '/scratch/gpfs/dingyunl/nersc/g0-main-new/'
sub = ['output_IC2_bgk_constNu/',
       'output_IC2_bgk_Nz96_constNu/',
       'output_IC2_lbo_Nz96_constNu/']
sim = ['gk_bgk_im_asdex_IC2_2x2v_p1',
       'gk_bgk_im_asdex_IC2_Nz96_2x2v_p1',
       'gk_lbo_asdex_IC2_Nz96_2x2v_p1']
diag = ['-elc_M3par_', '-elc_M3perp_', '-ion_M3par_', '-ion_M3perp_']
frame = '20'

polyOrder = 1
basisType = 'ms'
testName = ['BGK, low', 'BGK, high', 'LBD, high']

n0 = 1.4e19
me = 9.10938215e-31
mp = 1.672621637e-27
mi = 2.014 * mp
eV = 1.602176487e-19
Te = 62.5 * eV
Ti = 94.5 * eV
vte = np.sqrt(Te/me)
vti = np.sqrt(Ti/mi)
norm_e = n0 * np.power(vte, 3.0)
norm_i = n0 * np.power(vti, 3.0)

z0 = '0.162' # radial center of the domain
psi_src = 0.1574
M3_e = []
M3_i = []
for i in range(len(testName)):
    #elcM3parData = pg.GData(elcM3parFile[i])
    elcM3parData = pg.GData(root+sub[i]+sim[i]+diag[0]+frame+'.gkyl')
    elcM3parDataInterp = pg.GInterpModal(elcM3parData, polyOrder, basisType)
    elcM3parDataInterp.interpolate(overwrite=True)
    grid, elcM3par = pg.data.select(elcM3parDataInterp.data, z0=z0)
    #elcM3perpData = pg.GData(elcM3perpFile[i])
    elcM3perpData = pg.GData(root+sub[i]+sim[i]+diag[1]+frame+'.gkyl')
    elcM3perpDataInterp = pg.GInterpModal(elcM3perpData, polyOrder, basisType)
    elcM3perpDataInterp.interpolate(overwrite=True)
    grid, elcM3perp = pg.data.select(elcM3perpDataInterp.data, z0=z0)
    M3_e.append(np.squeeze(elcM3par)+np.squeeze(elcM3perp))

    #ionM3parData = pg.GData(ionM3parFile[i])
    ionM3parData = pg.GData(root+sub[i]+sim[i]+diag[2]+frame+'.gkyl')
    ionM3parDataInterp = pg.GInterpModal(ionM3parData, polyOrder, basisType)
    ionM3parDataInterp.interpolate(overwrite=True)
    grid, ionM3par = pg.data.select(ionM3parDataInterp.data, z0=z0)
    #ionM3perpData = pg.GData(ionM3perpFile[i])
    ionM3perpData = pg.GData(root+sub[i]+sim[i]+diag[3]+frame+'.gkyl')
    ionM3perpDataInterp = pg.GInterpModal(ionM3perpData, polyOrder, basisType)
    ionM3perpDataInterp.interpolate(overwrite=True)
    grid, ionM3perp = pg.data.select(ionM3perpDataInterp.data, z0=z0)
    M3_i.append(np.squeeze(ionM3par)+np.squeeze(ionM3perp))

    if i==0:
        theta1 = (grid[1][0:-1]+grid[1][1:])/2.0/np.pi
        #print(np.shape(theta1), theta1)
    if i==1:
        theta2 = (grid[1][0:-1]+grid[1][1:])/2.0/np.pi
        #print(np.shape(theta2), theta2)

z1 = '3.14159265359'
M3_e_ID = []
M3_i_ID = []
for i in range(len(testName)):
    #elcM3parData = pg.GData(elcM3parFile[i])
    elcM3parData = pg.GData(root+sub[i]+sim[i]+diag[0]+frame+'.gkyl')
    elcM3parDataInterp = pg.GInterpModal(elcM3parData, polyOrder, basisType)
    elcM3parDataInterp.interpolate(overwrite=True)
    grid, elcM3par = pg.data.select(elcM3parDataInterp.data, z1=z1)
    #elcM3perpData = pg.GData(elcM3perpFile[i])
    elcM3perpData = pg.GData(root+sub[i]+sim[i]+diag[1]+frame+'.gkyl')
    elcM3perpDataInterp = pg.GInterpModal(elcM3perpData, polyOrder, basisType)
    elcM3perpDataInterp.interpolate(overwrite=True)
    grid, elcM3perp = pg.data.select(elcM3perpDataInterp.data, z1=z1)
    M3_e_ID.append(np.squeeze(elcM3par)+np.squeeze(elcM3perp))

    #ionM3parData = pg.GData(ionM3parFile[i])
    ionM3parData = pg.GData(root+sub[i]+sim[i]+diag[2]+frame+'.gkyl')
    ionM3parDataInterp = pg.GInterpModal(ionM3parData, polyOrder, basisType)
    ionM3parDataInterp.interpolate(overwrite=True)
    grid, ionM3par = pg.data.select(ionM3parDataInterp.data, z1=z1)
    #ionM3perpData = pg.GData(ionM3perpFile[i])
    ionM3perpData = pg.GData(root+sub[i]+sim[i]+diag[3]+frame+'.gkyl')
    ionM3perpDataInterp = pg.GInterpModal(ionM3perpData, polyOrder, basisType)
    ionM3perpDataInterp.interpolate(overwrite=True)
    grid, ionM3perp = pg.data.select(ionM3perpDataInterp.data, z1=z1)
    M3_i_ID.append(np.squeeze(ionM3par)+np.squeeze(ionM3perp))
psi_ID = (grid[0][0:-1]+grid[0][1:])/2

z1 = '-3.14159265359'
M3_e_OD = []
M3_i_OD = []
for i in range(len(testName)):
    #elcM3parData = pg.GData(elcM3parFile[i])
    elcM3parData = pg.GData(root+sub[i]+sim[i]+diag[0]+frame+'.gkyl')
    elcM3parDataInterp = pg.GInterpModal(elcM3parData, polyOrder, basisType)
    elcM3parDataInterp.interpolate(overwrite=True)
    grid, elcM3par = pg.data.select(elcM3parDataInterp.data, z1=z1)
    #elcM3perpData = pg.GData(elcM3perpFile[i])
    elcM3perpData = pg.GData(root+sub[i]+sim[i]+diag[1]+frame+'.gkyl')
    elcM3perpDataInterp = pg.GInterpModal(elcM3perpData, polyOrder, basisType)
    elcM3perpDataInterp.interpolate(overwrite=True)
    grid, elcM3perp = pg.data.select(elcM3perpDataInterp.data, z1=z1)
    M3_e_OD.append(np.squeeze(elcM3par)+np.squeeze(elcM3perp))

    #ionM3parData = pg.GData(ionM3parFile[i])
    ionM3parData = pg.GData(root+sub[i]+sim[i]+diag[2]+frame+'.gkyl')
    ionM3parDataInterp = pg.GInterpModal(ionM3parData, polyOrder, basisType)
    ionM3parDataInterp.interpolate(overwrite=True)
    grid, ionM3par = pg.data.select(ionM3parDataInterp.data, z1=z1)
    #ionM3perpData = pg.GData(ionM3perpFile[i])
    ionM3perpData = pg.GData(root+sub[i]+sim[i]+diag[3]+frame+'.gkyl')
    ionM3perpDataInterp = pg.GInterpModal(ionM3perpData, polyOrder, basisType)
    ionM3perpDataInterp.interpolate(overwrite=True)
    grid, ionM3perp = pg.data.select(ionM3perpDataInterp.data, z1=z1)
    M3_i_OD.append(np.squeeze(ionM3par)+np.squeeze(ionM3perp))
psi_OD = (grid[0][0:-1]+grid[0][1:])/2

# Make plots
lblSize = 14
ttlSize = 16
tickSize = 12
lgdSize = 12
txtSize = 14

xlim = np.array([[-1.0, 1.0], [0.154, 0.170], [0.154, 0.170]])
ylim_e = np.array([[-0.04, 0.04], [-0.02, 0.34], [-0.50, 0.05]])
ylim_i = np.array([[-1.50, 1.00], [-0.05, 1.00], [-1.50, 0.10]])

lblName = [r'$q_{\parallel e}~(\rho_e v_{te}^3)$',
           r'$q_{\parallel i}~(\rho_i v_{ti}^3)$']
ttlName = ['Parallel', 'Radial, inner divertor', 'Radial, outer divertor']

fig, axs = plt.subplot_mosaic([['a)', 'c)', 'e)'], 
                               ['b)', 'd)', 'f)']],
                              layout='constrained', figsize=[12.8,6.4])

for label, ax in axs.items():
    # Use ScaledTranslation to put the label
    # - at the top left corner (axes fraction (0, 1)),
    # - offset 20 pixels left and 7 pixels up (offset points (-20, +7)),
    # i.e. just outside the axes.
    ax.text(
        0.0, 1.0, label, transform=(
            ax.transAxes + ScaledTranslation(-48/72, +2/72, fig.dpi_scale_trans)),
        fontsize=ttlSize, va='bottom', fontfamily='serif')

axs['a)'].plot(theta1, M3_e[0]/2.0/norm_e, color='tab:blue', label=testName[0])
axs['a)'].plot(theta2, M3_e[1]/2.0/norm_e, color='tab:orange', label=testName[1])
axs['a)'].plot(theta2, M3_e[2]/2.0/norm_e, color='tab:orange', linestyle=':', label=testName[2])
axs['a)'].set_xlim(xlim[0,0], xlim[0,1])
axs['a)'].set_ylim(ylim_e[0,0], ylim_e[0,1])
axs['a)'].set_ylabel(lblName[0], fontsize=lblSize)
axs['a)'].set_title(ttlName[0], fontsize=ttlSize)
axs['a)'].tick_params(labelsize=tickSize)
#axs['a)'].legend(fontsize=lgdSize, frameon=False, ncol=2, loc='lower right')
axs['a)'].legend(fontsize=lgdSize, frameon=False, loc='lower right')

axs['b)'].plot(theta1, M3_i[0]/2.0/norm_i, color='tab:blue', label=testName[0])
axs['b)'].plot(theta2, M3_i[1]/2.0/norm_i, color='tab:orange', label=testName[1])
axs['b)'].plot(theta2, M3_i[2]/2.0/norm_i, color='tab:orange', linestyle=':', label=testName[2])
axs['b)'].set_xlim(xlim[0,0], xlim[0,1])
axs['b)'].set_ylim(ylim_i[0,0], ylim_i[0,1])
axs['b)'].set_xlabel(r'normalized poloidal arc length $\theta~(\pi)$', fontsize=lblSize)
axs['b)'].set_ylabel(lblName[1], fontsize=lblSize)
axs['b)'].tick_params(labelsize=tickSize)

axs['c)'].plot(psi_ID, M3_e_ID[0]/2.0/norm_e, color='tab:blue', label=testName[0])
axs['c)'].plot(psi_ID, M3_e_ID[1]/2.0/norm_e, color='tab:orange', label=testName[1])
axs['c)'].plot(psi_ID, M3_e_ID[2]/2.0/norm_e, color='tab:orange', linestyle=':', label=testName[2])
axs['c)'].set_xlim(xlim[1,0], xlim[1,1])
axs['c)'].set_ylim(ylim_e[1,0], ylim_e[1,1])
#axs['c)'].set_ylabel(lblName[0], fontsize=lblSize)
axs['c)'].set_title(ttlName[1], fontsize=ttlSize)
axs['c)'].tick_params(labelsize=tickSize)
#axs['c)'].legend(fontsize=lgdSize, frameon=False, ncol=2, loc='upper right')

axs['d)'].plot(psi_ID, M3_i_ID[0]/2.0/norm_i, color='tab:blue', label=testName[0])
axs['d)'].plot(psi_ID, M3_i_ID[1]/2.0/norm_i, color='tab:orange', label=testName[1])
axs['d)'].plot(psi_ID, M3_i_ID[2]/2.0/norm_i, color='tab:orange', linestyle=':', label=testName[2])
axs['d)'].set_xlim(xlim[1,0], xlim[1,1])
axs['d)'].set_ylim(ylim_i[1,0], ylim_i[1,1])
axs['d)'].set_xlabel(r'$\psi$', fontsize=lblSize)
#axs['d)'].set_ylabel(lblName[1], fontsize=lblSize)
axs['d)'].tick_params(labelsize=tickSize)

axs['e)'].plot(psi_OD, M3_e_OD[0]/2.0/norm_e, color='tab:blue', label=testName[0])
axs['e)'].plot(psi_OD, M3_e_OD[1]/2.0/norm_e, color='tab:orange', label=testName[1])
axs['e)'].plot(psi_OD, M3_e_OD[2]/2.0/norm_e, color='tab:orange', linestyle=':', label=testName[2])
axs['e)'].vlines(psi_src, ylim_e[2,0], ylim_e[2,1], 'k', '--')
axs['e)'].set_xlim(xlim[2,0], xlim[2,1])
axs['e)'].set_ylim(ylim_e[2,0], ylim_e[2,1])
#axs['e)'].set_ylabel(lblName[0], fontsize=lblSize)
axs['e)'].set_title(ttlName[2], fontsize=ttlSize)
axs['e)'].tick_params(labelsize=tickSize)
#axs['e)'].legend(fontsize=lgdSize, frameon=False, ncol=2, loc='upper right')

axs['f)'].plot(psi_OD, M3_i_OD[0]/2.0/norm_i, color='tab:blue', label=testName[0])
axs['f)'].plot(psi_OD, M3_i_OD[1]/2.0/norm_i, color='tab:orange', label=testName[1])
axs['f)'].plot(psi_OD, M3_i_OD[2]/2.0/norm_i, color='tab:orange', linestyle=':', label=testName[2])
axs['f)'].vlines(psi_src, ylim_i[2,0], ylim_i[2,1], 'k', '--')
axs['f)'].set_xlim(xlim[2,0], xlim[2,1])
axs['f)'].set_ylim(ylim_i[2,0], ylim_i[2,1])
axs['f)'].set_xlabel(r'$\psi$', fontsize=lblSize)
#axs['f)'].set_ylabel(lblName[1], fontsize=lblSize)
axs['f)'].tick_params(labelsize=tickSize)

plt.show()
#plt.savefig('heat_flux_profiles.png')
