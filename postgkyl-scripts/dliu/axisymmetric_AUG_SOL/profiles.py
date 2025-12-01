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
diag = ['-elc_MaxwellianMoments_', '-ion_MaxwellianMoments_', '-field_']
frame = '20'

polyOrder = 1
basisType = 'ms'
testName = ['BGK, low', 'BGK, high', 'LBD, high']

mp = 1.672621637e-27
mi = 2.014*mp
me = 9.10938215e-31
eV = 1.602176487e-19
psi_src = 0.1574

z0 = '0.162' # radial center of the domain
ne = []
Te = []
Ti = []
Phi = []
for i in range(len(sim)):
    #elcMomData = pg.GData(elcMomFile[i])
    elcMomData = pg.GData(root+sub[i]+sim[i]+diag[0]+frame+'.gkyl')
    elcMomDataInterp = pg.GInterpModal(elcMomData, polyOrder, basisType)
    elcMomDataInterp.interpolate(0, overwrite=True)
    grid, elcDen = pg.data.select(elcMomDataInterp.data, z0=z0)
    ne.append(np.squeeze(elcDen))
    #elcMomData = pg.GData(elcMomFile[i])
    elcMomData = pg.GData(root+sub[i]+sim[i]+diag[0]+frame+'.gkyl')
    elcMomDataInterp = pg.GInterpModal(elcMomData, polyOrder, basisType)
    elcMomDataInterp.interpolate(2, overwrite=True)
    grid, elcTemp = pg.data.select(elcMomDataInterp.data, z0=z0)
    Te.append(np.squeeze(elcTemp))
    #ionMomData = pg.GData(ionMomFile[i])
    ionMomData = pg.GData(root+sub[i]+sim[i]+diag[1]+frame+'.gkyl')
    ionMomDataInterp = pg.GInterpModal(ionMomData, polyOrder, basisType)
    ionMomDataInterp.interpolate(2, overwrite=True)
    grid, ionTemp = pg.data.select(ionMomDataInterp.data, z0=z0)
    Ti.append(np.squeeze(ionTemp))
    #fieldData = pg.GData(fieldFile[i])
    fieldData = pg.GData(root+sub[i]+sim[i]+diag[2]+frame+'.gkyl')
    fieldDataInterp = pg.GInterpModal(fieldData, polyOrder, basisType)
    fieldDataInterp.interpolate(overwrite=True)
    grid, field = pg.data.select(fieldDataInterp.data, z0=z0)
    Phi.append(np.squeeze(field))
    if i==0:
        theta1 = (grid[1][0:-1]+grid[1][1:])/2.0/np.pi
        #print(np.shape(theta1), theta1)
    if i==1:
        theta2 = (grid[1][0:-1]+grid[1][1:])/2.0/np.pi

z1 = '1.57079632679'
ne_IMP = []
Te_IMP = []
Ti_IMP = []
Phi_IMP = []
for i in range(len(sim)):
    #elcMomData = pg.GData(elcMomFile[i])
    elcMomData = pg.GData(root+sub[i]+sim[i]+diag[0]+frame+'.gkyl')
    elcMomDataInterp = pg.GInterpModal(elcMomData, polyOrder, basisType)
    elcMomDataInterp.interpolate(0, overwrite=True)
    grid, elcDen = pg.data.select(elcMomDataInterp.data, z1=z1)
    ne_IMP.append(np.squeeze(elcDen))
    #elcMomData = pg.GData(elcMomFile[i])
    elcMomData = pg.GData(root+sub[i]+sim[i]+diag[0]+frame+'.gkyl')
    elcMomDataInterp = pg.GInterpModal(elcMomData, polyOrder, basisType)
    elcMomDataInterp.interpolate(2, overwrite=True)
    grid, elcTemp = pg.data.select(elcMomDataInterp.data, z1=z1)
    Te_IMP.append(np.squeeze(elcTemp))
    #ionMomData = pg.GData(ionMomFile[i])
    ionMomData = pg.GData(root+sub[i]+sim[i]+diag[1]+frame+'.gkyl')
    ionMomDataInterp = pg.GInterpModal(ionMomData, polyOrder, basisType)
    ionMomDataInterp.interpolate(2, overwrite=True)
    grid, ionTemp = pg.data.select(ionMomDataInterp.data, z1=z1)
    Ti_IMP.append(np.squeeze(ionTemp))
    #fieldData = pg.GData(fieldFile[i])
    fieldData = pg.GData(root+sub[i]+sim[i]+diag[2]+frame+'.gkyl')
    fieldDataInterp = pg.GInterpModal(fieldData, polyOrder, basisType)
    fieldDataInterp.interpolate(overwrite=True)
    grid, field = pg.data.select(fieldDataInterp.data, z1=z1)
    Phi_IMP.append(np.squeeze(field))
psi_IMP = (grid[0][0:-1]+grid[0][1:])/2

z1 = '-1.57079632679'
ne_OMP = []
Te_OMP = []
Ti_OMP = []
Phi_OMP = []
for i in range(len(sim)):
    #elcMomData = pg.GData(elcMomFile[i])
    elcMomData = pg.GData(root+sub[i]+sim[i]+diag[0]+frame+'.gkyl')
    elcMomDataInterp = pg.GInterpModal(elcMomData, polyOrder, basisType)
    elcMomDataInterp.interpolate(0, overwrite=True)
    grid, elcDen = pg.data.select(elcMomDataInterp.data, z1=z1)
    ne_OMP.append(np.squeeze(elcDen))
    #elcMomData = pg.GData(elcMomFile[i])
    elcMomData = pg.GData(root+sub[i]+sim[i]+diag[0]+frame+'.gkyl')
    elcMomDataInterp = pg.GInterpModal(elcMomData, polyOrder, basisType)
    elcMomDataInterp.interpolate(2, overwrite=True)
    grid, elcTemp = pg.data.select(elcMomDataInterp.data, z1=z1)
    Te_OMP.append(np.squeeze(elcTemp))
    #ionMomData = pg.GData(ionMomFile[i])
    ionMomData = pg.GData(root+sub[i]+sim[i]+diag[1]+frame+'.gkyl')
    ionMomDataInterp = pg.GInterpModal(ionMomData, polyOrder, basisType)
    ionMomDataInterp.interpolate(2, overwrite=True)
    grid, ionTemp = pg.data.select(ionMomDataInterp.data, z1=z1)
    Ti_OMP.append(np.squeeze(ionTemp))
    #fieldData = pg.GData(fieldFile[i])
    fieldData = pg.GData(root+sub[i]+sim[i]+diag[2]+frame+'.gkyl')
    fieldDataInterp = pg.GInterpModal(fieldData, polyOrder, basisType)
    fieldDataInterp.interpolate(overwrite=True)
    grid, field = pg.data.select(fieldDataInterp.data, z1=z1)
    Phi_OMP.append(np.squeeze(field))
psi_OMP = (grid[0][0:-1]+grid[0][1:])/2

# Make plots
lblSize = 14
ttlSize = 16
tickSize = 12
lgdSize = 12
txtSize = 14

lblName = [r'$n_e~(m^{-3})$', r'$T_e~(eV)$', r'$T_i~(eV)$', r'$\Phi~(V)$']
ttlName = ['Parallel', 'Radial, inboard midplane', 'Radial, outboard midplane']

xlim = np.array([[-1.0, 1.0], [0.154, 0.170], [0.154, 0.170]])
ylim_ne = [0.0, 2.0e19] # 2.2e19
ylim_Te = [0.0, 60.0]
ylim_Ti = [0.0, 90.0]
ylim_Phi = [0.0, 120.0]
fig, axs = plt.subplot_mosaic([['a)', 'e)', 'i)'], 
                               ['b)', 'f)', 'j)'], 
                               ['c)', 'g)', 'k)'], 
                               ['d)', 'h)', 'l)']],
                              layout='constrained', figsize=[12.8,12.8])

for label, ax in axs.items():
    # Use ScaledTranslation to put the label
    # - at the top left corner (axes fraction (0, 1)),
    # - offset 20 pixels left and 7 pixels up (offset points (-20, +7)),
    # i.e. just outside the axes.
    ax.text(
        0.0, 1.0, label, transform=(
            ax.transAxes + ScaledTranslation(-48/72, +2/72, fig.dpi_scale_trans)),
        fontsize=ttlSize, va='bottom', fontfamily='serif')

axs['a)'].plot(theta1, ne[0], color='tab:blue', label=testName[0])
axs['a)'].plot(theta2, ne[1], color='tab:orange', label=testName[1])
axs['a)'].plot(theta2, ne[2], color='tab:orange', linestyle=':', label=testName[2])
axs['a)'].set_xlim(xlim[0,0], xlim[0,1])
axs['a)'].set_ylim(ylim_ne[0], ylim_ne[1])
axs['a)'].set_ylabel(lblName[0], fontsize=lblSize)
axs['a)'].set_title(ttlName[0], fontsize=ttlSize)
axs['a)'].tick_params(labelsize=tickSize)
#axs['a)'].legend(fontsize=lgdSize, frameon=False, ncol=2, loc='upper right')
axs['a)'].legend(fontsize=lgdSize, frameon=False, loc='upper right')

axs['b)'].plot(theta1, Te[0]*me/eV, color='tab:blue', label=testName[0])
axs['b)'].plot(theta2, Te[1]*me/eV, color='tab:orange', label=testName[1])
axs['b)'].plot(theta2, Te[2]*me/eV, color='tab:orange', linestyle=':', label=testName[2])
axs['b)'].set_xlim(xlim[0,0], xlim[0,1])
axs['b)'].set_ylim(ylim_Te[0], ylim_Te[1])
axs['b)'].set_ylabel(lblName[1], fontsize=lblSize)
axs['b)'].tick_params(labelsize=tickSize)

axs['c)'].plot(theta1, Ti[0]*mi/eV, color='tab:blue', label=testName[0])
axs['c)'].plot(theta2, Ti[1]*mi/eV, color='tab:orange', label=testName[1])
axs['c)'].plot(theta2, Ti[2]*mi/eV, color='tab:orange', linestyle=':', label=testName[2])
axs['c)'].set_xlim(xlim[0,0], xlim[0,1])
axs['c)'].set_ylim(ylim_Ti[0], ylim_Ti[1])
axs['c)'].set_ylabel(lblName[2], fontsize=lblSize)
axs['c)'].tick_params(labelsize=tickSize)

axs['d)'].plot(theta1, Phi[0], color='tab:blue', label=testName[0])
axs['d)'].plot(theta2, Phi[1], color='tab:orange', label=testName[1])
axs['d)'].plot(theta2, Phi[2], color='tab:orange', linestyle=':', label=testName[2])
axs['d)'].set_xlim(xlim[0,0], xlim[0,1])
axs['d)'].set_ylim(ylim_Phi[0], ylim_Phi[1])
axs['d)'].set_xlabel(r'normalized poloidal arc length $\theta~(\pi)$', fontsize=lblSize)
axs['d)'].set_ylabel(lblName[3], fontsize=lblSize)
axs['d)'].tick_params(labelsize=tickSize)

axs['e)'].plot(psi_IMP, ne_IMP[0], color='tab:blue', label=testName[0])
axs['e)'].plot(psi_IMP, ne_IMP[1], color='tab:orange', label=testName[1])
axs['e)'].plot(psi_IMP, ne_IMP[2], color='tab:orange', linestyle=':', label=testName[2])
axs['e)'].set_xlim(xlim[1,0], xlim[1,1])
axs['e)'].set_ylim(ylim_ne[0], ylim_ne[1])
axs['e)'].set_title(ttlName[1], fontsize=ttlSize)
axs['e)'].tick_params(labelsize=tickSize)
#axs['e)'].legend(fontsize=lgdSize, frameon=False, ncol=2, loc='upper right') 

axs['f)'].plot(psi_IMP, Te_IMP[0]*me/eV, color='tab:blue', label=testName[0])
axs['f)'].plot(psi_IMP, Te_IMP[1]*me/eV, color='tab:orange', label=testName[1])
axs['f)'].plot(psi_IMP, Te_IMP[2]*me/eV, color='tab:orange', linestyle=':', label=testName[2])
axs['f)'].set_xlim(xlim[1,0], xlim[1,1])
axs['f)'].set_ylim(ylim_Te[0], ylim_Te[1])
axs['f)'].tick_params(labelsize=tickSize)

axs['g)'].plot(psi_IMP, Ti_IMP[0]*mi/eV, color='tab:blue', label=testName[0])
axs['g)'].plot(psi_IMP, Ti_IMP[1]*mi/eV, color='tab:orange', label=testName[1])
axs['g)'].plot(psi_IMP, Ti_IMP[2]*mi/eV, color='tab:orange', linestyle=':', label=testName[2])
axs['g)'].set_xlim(xlim[1,0], xlim[1,1])
axs['g)'].set_ylim(ylim_Ti[0], ylim_Ti[1])
axs['g)'].tick_params(labelsize=tickSize)

axs['h)'].plot(psi_IMP, Phi_IMP[0], color='tab:blue', label=testName[0])
axs['h)'].plot(psi_IMP, Phi_IMP[1], color='tab:orange', label=testName[1])
axs['h)'].plot(psi_IMP, Phi_IMP[2], color='tab:orange', linestyle=':', label=testName[2])
axs['h)'].set_xlim(xlim[1,0], xlim[1,1])
axs['h)'].set_ylim(ylim_Phi[0], ylim_Phi[1])
axs['h)'].set_xlabel(r'$\psi$', fontsize=lblSize)
axs['h)'].tick_params(labelsize=tickSize)

axs['i)'].plot(psi_OMP, ne_OMP[0], color='tab:blue', label=testName[0])
axs['i)'].plot(psi_OMP, ne_OMP[1], color='tab:orange', label=testName[1])
axs['i)'].plot(psi_OMP, ne_OMP[2], color='tab:orange', linestyle=':', label=testName[2])
axs['i)'].vlines(psi_src, ylim_ne[0], ylim_ne[1], 'k', '--')
axs['i)'].set_xlim(xlim[2,0], xlim[2,1])
axs['i)'].set_ylim(ylim_ne[0], ylim_ne[1])
#axs['i)'].set_ylabel(lblName[0], fontsize=lblSize)
axs['i)'].set_title(ttlName[2], fontsize=ttlSize)
axs['i)'].tick_params(labelsize=tickSize)
#axs['i)'].legend(fontsize=lgdSize, frameon=False, ncol=2, loc='upper right') 

axs['j)'].plot(psi_OMP, Te_OMP[0]*me/eV, color='tab:blue', label=testName[0])
axs['j)'].plot(psi_OMP, Te_OMP[1]*me/eV, color='tab:orange', label=testName[1])
axs['j)'].plot(psi_OMP, Te_OMP[2]*me/eV, color='tab:orange', linestyle=':', label=testName[2])
axs['j)'].vlines(psi_src, ylim_Te[0], ylim_Te[1], 'k', '--')
axs['j)'].set_xlim(xlim[2,0], xlim[2,1])
axs['j)'].set_ylim(ylim_Te[0], ylim_Te[1])
#axs['j)'].set_ylabel(lblName[1], fontsize=lblSize)
axs['j)'].tick_params(labelsize=tickSize)

axs['k)'].plot(psi_OMP, Ti_OMP[0]*mi/eV, color='tab:blue', label=testName[0])
axs['k)'].plot(psi_OMP, Ti_OMP[1]*mi/eV, color='tab:orange', label=testName[1])
axs['k)'].plot(psi_OMP, Ti_OMP[2]*mi/eV, color='tab:orange', linestyle=':', label=testName[2])
axs['k)'].vlines(psi_src, ylim_Ti[0], ylim_Ti[1], 'k', '--')
axs['k)'].set_xlim(xlim[2,0], xlim[2,1])
axs['k)'].set_ylim(ylim_Ti[0], ylim_Ti[1])
#axs['k)'].set_ylabel(lblName[2], fontsize=lblSize)
axs['k)'].tick_params(labelsize=tickSize)

axs['l)'].plot(psi_OMP, Phi_OMP[0], color='tab:blue', label=testName[0])
axs['l)'].plot(psi_OMP, Phi_OMP[1], color='tab:orange', label=testName[1])
axs['l)'].plot(psi_OMP, Phi_OMP[2], color='tab:orange', linestyle=':', label=testName[2])
axs['l)'].vlines(psi_src, ylim_Phi[0], ylim_Phi[1], 'k', '--')
axs['l)'].set_xlim(xlim[2,0], xlim[2,1])
axs['l)'].set_ylim(ylim_Phi[0], ylim_Phi[1])
axs['l)'].set_xlabel(r'$\psi$', fontsize=lblSize)
#axs['l)'].set_ylabel(lblName[3], fontsize=lblSize)
axs['l)'].tick_params(labelsize=tickSize)

#plt.show()
plt.savefig('profiles.png')
