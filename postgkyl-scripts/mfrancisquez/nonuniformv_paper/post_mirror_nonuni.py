#[ ........................................................... ]#
#[
#[ post processing for nononuniform v mirror 1x2v simulations
#[
#[ Manaure Francisquez.
#[ April 2025.
#[
#[ ........................................................... ]#


import numpy as np
import postgkyl as pg
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from matplotlib.patches import ConnectionPatch
import matplotlib as mpl
import sys
#[ Append postgkyl wrappers.
#sys.path.insert(0, '/home/manaurer/pets/gkeyll/postProcessingScripts/')
sys.path.insert(0, '/Users/mfrancis/Documents/codebits/pets/gkeyll/postProcessingScripts/')
import pgkylUtil as pgu
from scipy import special
from scipy import optimize
#import h5py

plot_fi_vspace_Adiabatic = True   #[ Plot ion velocity space at z=0, z=z_m and z=zMax.
plot_moms                = False   #[ Plot density, temperature and Upar.
plot_phi_t_comp          = False   #[ phi(z,t) and e*Delta phi/Te vs. t.

outDir = './'

outFigureFile    = True     #[ Output a figure file?.
figureFileFormat = '.png'    #[ Can be .png, .pdf, .ps, .eps, .svg.
saveData         = False    #.Indicate whether to save data in plot to HDF5 file.

#[ ............... End of user inputs (MAYBE) ..................... ]#

polyOrder = 1
basisType = 'ms'

eps0, mu0 = 8.8541878176204e-12, 1.2566370614359e-06
eV        = 1.602176487e-19
qe, qi    = -1.602176487e-19, 1.602176487e-19
me, mp    = 9.10938215e-31, 1.672621637e-27

mi        = 2.014*mp                         #[ Deuterium ion mass.
Te0       = 940*eV
n0        = 3.e19
B_p       = 0.53
beta      = 0.4                              #[ Ratio of plasma to magnetic pressure.
tau       = (B_p**2)*beta/(2*mu0*n0*Te0)-1    #[ Ti/Te ratio.
Ti0       = tau*Te0

#[ Thermal speeds.
vti = np.sqrt(Ti0/mi)
vte = np.sqrt(Te0/me)
c_s = np.sqrt(Te0/mi)

#[ Gyrofrequencies and gyroradii.
omega_ci = eV*B_p/mi
rho_s    = c_s/omega_ci

z_m = 0.983244   #[ Location of maximum B-field.

#[ Some RGB colors. These are MATLAB-like.
defaultBlue    = [0, 0.4470, 0.7410]
defaultOrange  = [0.8500, 0.3250, 0.0980]
defaultGreen   = [0.4660, 0.6740, 0.1880]
defaultPurple  = [0.4940, 0.1840, 0.5560]
defaultRed     = [0.6350, 0.0780, 0.1840]
defaultSkyBlue = [0.3010, 0.7450, 0.9330]
grey           = [0.5, 0.5, 0.5]
#[ Colors in a single array.
defaultColors = [defaultBlue,defaultOrange,defaultGreen,defaultPurple,defaultRed,defaultSkyBlue,grey,'black']

#[ LineStyles in a single array.
lineStyles = ['-','--',':','-.','None','None','None','None']
markers    = ['None','None','None','None','o','d','s','+']

#[ Some fontsizes used in plots.
xyLabelFontSize       = 17
titleFontSize         = 17
colorBarLabelFontSize = 17
tickFontSize          = 14
legendFontSize        = 14
textFontSize          = 16

#.Set the font size of the ticks to a given size.
def setTickFontSize(axIn,fontSizeIn):
  axIn.tick_params(axis='both', which='major', labelsize=fontSizeIn)

#.Plot vertical lines at +/- given x location.
def plot_verticalLinesPM(xIn, axIn):
  ymin, ymax = axIn.get_ylim()
  eps = 0.5*ymax,
  axIn.plot([xIn, xIn], [ymin-eps, ymax+eps], linestyle=":", color='grey')
  axIn.plot([-xIn, -xIn], [ymin-eps, ymax+eps], linestyle=":", color='grey')
  axIn.set_ylim(ymin, ymax)

#[ Plot a pcolormesh inset in a given frame.
def plot_inset_pcolormesh(ax, X, Y, data, size, coords, frame_color, vbounds):
  axins = ax.inset_axes(size)
  hpc = axins.pcolormesh(X, Y, data, cmap='inferno', vmin=vbounds[0], vmax=vbounds[1])
  axins.set_xlim(coords[0][0], coords[0][1])
  axins.set_ylim(coords[1][0], coords[1][1])
  axins.set_xticklabels([])
  axins.set_yticklabels([])
  ax.indicate_inset_zoom(axins, edgecolor=frame_color)
  for spine in axins.spines.values():
    spine.set_edgecolor(frame_color)
  return axins, hpc

#................................................................................#

if plot_fi_vspace_Adiabatic:
  #[ Plot slices of the ion distribution functions at z=0, z=z_m and z=zMax.

#  dataDir = '/scratch/gpfs/manaurer/gkeyll/nonuniformv/mirror/gz57_muMax3p0vt_vparMax16_muLinQuad_Nz192/'
  dataDir = './'

  fileName = 'gz57_1x2v_p1'    #.Root name of files to process.
  frame    = 190

  filePath = dataDir+fileName+'-ion'

  #[ Load the distribution times the Jacobian (J_v*f), and the Jacobian J_v, divide (J_vf)/J_v and interpolate.
  f_data = pg.GData(filePath+'_'+str(frame)+'.gkyl', mapc2p_vel_name=filePath + '_mapc2p_vel.gkyl')
  Jv_data = pg.GData(filePath + '_jacobvel.gkyl', mapc2p_vel_name=filePath + '_mapc2p_vel.gkyl')
  f_c = f_data.get_values()
  Jv_c = Jv_data.get_values()
  f_data._values = f_c/Jv_c
  dg = pg.GInterpModal(f_data, poly_order=polyOrder, basis_type="gkhyb")
  xInt_i, fIon = dg.interpolate()
  fIon = np.squeeze(fIon)

  ndim = len(xInt_i)
  nxInt_i = [np.size(xInt_i[d]) for d in range(ndim)]

  #[ Cell center coordinates
  xIntC_i = [np.zeros(np.size(xInt_i[d])) for d in range(ndim)]
  for d in range(len(xIntC_i)):
    xIntC_i[d] = 0.5*(xInt_i[d][:-1]+xInt_i[d][1:])

  nxIntC_i = [np.size(xIntC_i[d]) for d in range(ndim)]

  #[ Normalize velocity space
  xInt_i[1] = xInt_i[1]/vti
  xInt_i[2] = xInt_i[2]/(0.5*mi*(vti**2)/B_p)
  xIntC_i[1] = xIntC_i[1]/vti
  xIntC_i[2] = xIntC_i[2]/(0.5*mi*(vti**2)/B_p)

  #[ Get indices along z of slices we wish to plot:
  plotz = [0., z_m, xInt_i[0][-1]]
  plotzIdx = [np.argmin(np.abs(xIntC_i[0]-val)) for val in plotz]

  #[ Prepare figure.
  figProp9 = (10.4, 4.)
  ax9Pos   = [[0.06, 0.15, 0.305, 0.74],[0.375, 0.15, 0.305, 0.74],[0.69, 0.15, 0.305, 0.74],]
  ca9Pos   = [[0.07, 0.9, 0.285, 0.02],[0.385, 0.9, 0.285, 0.02],[0.70, 0.9, 0.285, 0.02]]
  fig9     = plt.figure(figsize=figProp9)
  ax9      = [fig9.add_axes(pos) for pos in ax9Pos]
  cbar_ax9 = [fig9.add_axes(pos) for pos in ca9Pos]

  #[ Create colorplot grid. Recall coordinates have to be nodal.
  Xnodal_i = [np.outer(xInt_i[1], np.ones(nxInt_i[2])),
              np.outer(np.ones(nxInt_i[1]), xInt_i[2])]

  hpl9b = list()
  hcb9b = list()
  #[ For insets:
  inset_size = [0.5, 0.4, 0.48, 0.57]
  ax9in = [0 for i in range(len(ax9))]
  hpl9bin = [0 for i in range(len(ax9))]

  sub_coords = [[0., 2.], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
#  extreme_vals = [0., np.amax(fIon[plotzIdx[1],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  extreme_vals = [0., np.amax(fIon[plotzIdx[0],:,:])]
  hpl9b.append(ax9[0].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[0],:,:],
                                 cmap='inferno', vmin=extreme_vals[0], vmax=extreme_vals[1]))
#  #[ Plot two separate pcolormeshes, one showing the grid.
#  hpl9b.append(ax9[0].pcolormesh(Xnodal_i[0][:nxIntC_i[1]//2+1,:], Xnodal_i[1][:nxIntC_i[1]//2+1,:], fIon[plotzIdx[0],:nxInt_i[1]//2,:],
#                                 cmap='inferno', vmin=extreme_vals[0], vmax=extreme_vals[1], edgecolors='grey', linewidth=0.1))
#  hpl9b.append(ax9[0].pcolormesh(Xnodal_i[0][nxIntC_i[1]//2:,:], Xnodal_i[1][nxIntC_i[1]//2:,:], fIon[plotzIdx[0],nxInt_i[1]//2:,:],
#                                 cmap='inferno', vmin=extreme_vals[0], vmax=extreme_vals[1]))
  ax9in[0], hpl9bin[0] = plot_inset_pcolormesh(ax9[0], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[0],:,:], \
                                               inset_size, sub_coords, "white", extreme_vals)

  hcb9b.append(plt.colorbar(hpl9b[0], ax=ax9[0], cax=cbar_ax9[0], orientation='horizontal'))
  hcb9b[0].ax.xaxis.set_ticks_position('top')
  hcb9b[0].ax.tick_params(labelsize=tickFontSize)
  ax9[0].set_ylabel(r'$\mu/[m_iv_{ti0}^2/(2B_p)]$', fontsize=xyLabelFontSize)

  sub_coords = [[-0.5, 1.5], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[1],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
#  hpl9b.append(ax9[1].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[1],:,:],
#                                 cmap='inferno', vmin=extreme_vals[0], vmax=extreme_vals[1]))
  #[ Plot two separate pcolormeshes, one showing the grid.
  hpl9b.append(ax9[1].pcolormesh(Xnodal_i[0][:nxIntC_i[1]//2+1,:], Xnodal_i[1][:nxIntC_i[1]//2+1,:], fIon[plotzIdx[1],:nxInt_i[1]//2,:],
                                 cmap='inferno', vmin=extreme_vals[0], vmax=extreme_vals[1], edgecolors='grey', linewidth=0.1))
  hpl9b.append(ax9[1].pcolormesh(Xnodal_i[0][nxIntC_i[1]//2:,:], Xnodal_i[1][nxIntC_i[1]//2:,:], fIon[plotzIdx[1],nxInt_i[1]//2:,:],
                                 cmap='inferno', vmin=extreme_vals[0], vmax=extreme_vals[1]))
  ax9in[1], hpl9bin[1] = plot_inset_pcolormesh(ax9[1], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[1],:,:], \
                                               inset_size, sub_coords, "white", extreme_vals)

  hcb9b.append(plt.colorbar(hpl9b[1], ax=ax9[1], cax=cbar_ax9[1], orientation='horizontal'))
  hcb9b[1].ax.xaxis.set_ticks_position('top')
  hcb9b[1].ax.tick_params(labelsize=tickFontSize)

  sub_coords = [[.75, 3.75], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[2],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  hpl9b.append(ax9[2].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[2],:,:],
                                 cmap='inferno', vmin=0., vmax=2.e-1*np.amax(fIon[plotzIdx[2],:,:])))
  ax9in[2], hpl9bin[2] = plot_inset_pcolormesh(ax9[2], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[2],:,:], \
                                               inset_size, sub_coords, "white", extreme_vals)

  hcb9b.append(plt.colorbar(hpl9b[2], ax=ax9[2], cax=cbar_ax9[2], orientation='horizontal'))
  hcb9b[2].ax.xaxis.set_ticks_position('top')
  hcb9b[2].ax.tick_params(labelsize=tickFontSize)

#  hpl9b[-1].set_clim(0.5*np.amin(fIon[plotzIdx[-1],:,:]),0.5*np.amax(fIon[plotzIdx[-1],:,:]))
  print(np.amin(fIon[plotzIdx[0],:,:]),np.amax(fIon[plotzIdx[0],:,:]))
  print(np.amin(fIon[plotzIdx[1],:,:]),np.amax(fIon[plotzIdx[1],:,:]))
  print(np.amin(fIon[plotzIdx[2],:,:]),np.amax(fIon[plotzIdx[2],:,:]))
#  hpl9b[0].set_clim(  -1., 7.5)
#  hpl9b[-1].set_clim(  -0.05, .4)
#  hpl9bin[-1].set_clim(-0.05, .4)


  for i in range(3):
    ax9[i].set_xlabel(r'$v_\parallel/v_{ti0}$', fontsize=xyLabelFontSize, labelpad=-2)
    setTickFontSize(ax9[0+i],tickFontSize)
  for i in range(1,3):
    plt.setp( ax9[i].get_yticklabels(), visible=False)
#  for i in range(3):
#    ax9[i].set_ylabel(r'$\mu/[m_iv_{ti0}^2/(2B_p)]$', fontsize=xyLabelFontSize)
#    setTickFontSize(ax9[0+i],tickFontSize)
#    hmagx = ax9[i].xaxis.get_offset_text().set_size(tickFontSize)
#  for i in range(0,2):
#    plt.setp( ax9[0+i].get_xticklabels(), visible=False)
#  ax9[-1].set_xlabel(r'$v_\parallel/v_{ti}$', fontsize=xyLabelFontSize)

  plt.text(0.03, 0.89, r'(a) $f_i(z=0)$', fontsize=textFontSize, color='white', transform=ax9[0].transAxes)
  plt.text(0.03, 0.89, r'(b) $f_i(z=z_m)$', fontsize=textFontSize, color='white', transform=ax9[1].transAxes)
  plt.text(0.03, 0.89, r'(c) $f_i(z=L_z/2)$', fontsize=textFontSize, color='white', transform=ax9[2].transAxes)

  plt.text(0.85, 0.85, r'', fontsize=textFontSize, color='white', transform=ax9[0].transAxes)
  plt.text(0.85, 0.85, r'', fontsize=textFontSize, color='white', transform=ax9[1].transAxes)
  plt.text(0.85, 0.85, r'', fontsize=textFontSize, color='white', transform=ax9[2].transAxes)

#  fileName = 'gk57-wham1x2v-nonuniformmu'    #.Root name of files to process.
  figName = fileName+'_fiCenterThroatWall_'+str(frame)
  if outFigureFile:
    fig9.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")
    h5f.create_dataset('vpar',       np.shape(Xnodal_i[0]),           dtype='f8', data=Xnodal_i[0])
    h5f.create_dataset('mu',         np.shape(Xnodal_i[1]),           dtype='f8', data=Xnodal_i[1])
    h5f.create_dataset('fi_zeq0',    np.shape(fIon[plotzIdx[0],:,:]), dtype='f8', data=fIon[plotzIdx[0],:,:])
    h5f.create_dataset('fi_zeqzm',   np.shape(fIon[plotzIdx[1],:,:]), dtype='f8', data=fIon[plotzIdx[1],:,:])
    h5f.create_dataset('fi_zeqLzD2', np.shape(fIon[plotzIdx[2],:,:]), dtype='f8', data=fIon[plotzIdx[2],:,:])
    h5f.close()

#................................................................................#

if plot_phi_t_comp:
  #[ Plot phi(z,t) and e*Delta phi/Te vs. t for all three models.

  #[ Linear polarization, nonlinear polarization, and adiabatic electrons.
  dataDir = ['/scratch/gpfs/manaurer/gkeyll/mirror/gk57/',
             '/scratch/gpfs/manaurer/gkeyll/nonuniformv/mirror/gz57_muMax3p0vt_vparMax16_muLinQuad_Nz192/',]
  fileName = ['gk57-wham1x2v','gz57_1x2v_p1']    #[ Root name of files to process.
  nFrames = [192, 400] #[ Number of frames in each sim.
  fileExt = ['bp','gkyl']
  fieldFile = ['_phi_','-field_']

  figName  = 'gk-wham1x2v_ePhiDTe_vs_time_comp'

  #[ Prepare figure.
  figProp12 = (6.4,6.8)
  ax12Pos   = [[0.12, 0.690, 0.725, 0.285],
               [0.12, 0.395, 0.725, 0.285],
               [0.12, 0.100, 0.725, 0.285]]
  cax12Pos  = [[0.855, 0.690, 0.02, 0.285],
               [0.855, 0.395, 0.02, 0.285],]
  fig12     = plt.figure(figsize=figProp12)
  ax12      = [fig12.add_axes(pos) for pos in ax12Pos]
  cbar_ax12 = [fig12.add_axes(pos) for pos in cax12Pos]

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

#  #[ Place a line at the analytic Pastukhov expectation (A_p=0.5 and A_p=1):
#  #[ 4.5494485
#  timeLims = [0., 100.]
#  ax12[-1].plot([timeLims[0],timeLims[-1]],[4.5494485,4.5494485],color='grey',linestyle=lineStyles[3])
#  plt.text(1.01, 0.81, r'$A_p=0.5$', fontsize=14, color='grey', transform=ax12[-1].transAxes)
#  #[ 5.115795
#  ax12[-1].plot([timeLims[0],timeLims[-1]],[5.115795,5.115795],color='grey',linestyle=lineStyles[3])
#  plt.text(1.01, 0.91, r'$A_p=1$', fontsize=14, color='grey', transform=ax12[-1].transAxes)

  hpl12 = list()
  hcb12 = list()
  hpl12b = list()
  timeLims = [1e10, -1e10]
  for sI in range(len(fileName)):
  
    phiFile = dataDir[sI]+fileName[sI]+fieldFile[sI]    #.Complete file name.
    times = 1.e6*pgu.getTimeStamps(phiFile,fileExt[sI],0,nFrames[sI]-1)
  
    phiFile = dataDir[sI]+fileName[sI]+fieldFile[sI]+'%d.'+fileExt[sI]    #.Complete file name.
  
    #[ Load the grid.
    xInt, _, nxInt, lxInt, dxInt, _ = pgu.getGrid(phiFile % 0,polyOrder,basisType)
    xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(phiFile % 0,polyOrder,basisType,location='center')
  
    phi = np.zeros((nFrames[sI],nxIntC[0]))
    Te = Te0*np.ones((nFrames[sI],nxIntC[0]))
  
    eDeltaPhiDTe = np.zeros(nFrames[sI])
    for fI in range(nFrames[sI]):
      TeMid = 0.5*(Te[fI,nxIntC[0]//2-1]+Te[fI,nxIntC[0]//2])
      phi[fI,:] = (eV/TeMid)*np.squeeze(pgu.getInterpData(phiFile % fI, polyOrder, basisType))
      eDeltaPhiDTe[fI] = np.interp(0., xIntC[0], phi[fI,:])-np.interp(z_m, xIntC[0], phi[fI,:])
  
    #[ Create colorplot grid. Recall coordinates have to be nodal.
    timesC  = 0.5*(times[1:]+times[0:-1])
    Xnodal = [np.outer(np.append(np.append([timesC[0]-(timesC[1]-timesC[0])],timesC),timesC[-1]+(timesC[-1]-timesC[-2])), \
                       np.ones(np.shape(xInt[0]))), \
              np.outer(np.ones(np.size(timesC)+2),xInt[0])]

    timeLims = [min(timeLims[0],np.amin(Xnodal[0])), max(timeLims[-1],np.amax(Xnodal[0]))]
  
    hpl12.append(ax12[sI].pcolormesh(Xnodal[0], Xnodal[1], phi, cmap='inferno'))
    hcb12.append(plt.colorbar(hpl12[sI], ax=ax12[sI], cax=cbar_ax12[sI]))
    if saveData:
      elcStr = 'boltz'
      if sI==1:
        elcStr = 'nonlinPol'
      h5f.create_dataset('ePhiDTe_'+elcStr,         np.shape(phi), dtype='f8', data=phi)
      h5f.create_dataset('ePhiDTe_'+elcStr+'_time', np.shape(Xnodal[0]), dtype='f8', data=Xnodal[0])
      h5f.create_dataset('ePhiDTe_'+elcStr+'_z',    np.shape(Xnodal[1]), dtype='f8', data=Xnodal[1])

    #[ Plot the potential drop to the throat over time:
    hpl12b.append(ax12[2].plot(times, eDeltaPhiDTe, color=defaultColors[sI], linestyle=lineStyles[sI]))

    if saveData:
      elcStr = 'boltz'
      if sI==1:
        elcStr = 'nonlinPol'
      elif sI==2:
        elcStr = 'linPol'
      h5f.create_dataset('eDeltaPhiDTe_'+elcStr,         np.shape(eDeltaPhiDTe), dtype='f8', data=eDeltaPhiDTe)
      h5f.create_dataset('eDeltaPhiDTe_'+elcStr+'_time', np.shape(times), dtype='f8', data=times)

  for sI in range(2):
    hcb12[sI].set_label('$e\phi(z{=}0)/T_{e0}$', rotation=90, labelpad=0, fontsize=colorBarLabelFontSize)
    hcb12[sI].ax.tick_params(labelsize=tickFontSize)
    plt.setp( ax12[sI].get_xticklabels(), visible=False)
    ax12[sI].set_ylabel(r'$z$ (m)', fontsize=xyLabelFontSize, labelpad=-4)
  ax12[-1].set_xlabel(r'Time ($\mu$s)', fontsize=xyLabelFontSize)
  ax12[-1].set_ylabel(r'$e\Delta \phi/T_{e0}$', fontsize=xyLabelFontSize)
  ax12[-1].legend([hpl12b[0][0], hpl12b[1][0]],['uniform',r'nonuniform'], fontsize=legendFontSize, frameon=False)
  plt.text(0.7, 0.88, r'Uniform', fontsize=14, color='black', fontweight='regular', transform=ax12[0].transAxes)
  plt.text(0.7, 0.88, r'Nonuniform' , fontsize=14, color='white', fontweight='regular', transform=ax12[1].transAxes)
  plt.text(0.025, 0.05, r'(a)', fontsize=textFontSize, color='white', transform=ax12[0].transAxes)
  plt.text(0.025, 0.05, r'(b)', fontsize=textFontSize, color='white', transform=ax12[1].transAxes)
  plt.text(0.025, 0.05, r'(c)', fontsize=textFontSize, color='black', transform=ax12[2].transAxes)
  for i in range(len(ax12)):
    ax12[i].set_xlim( timeLims[0], timeLims[-1]*1.005 )
    setTickFontSize(ax12[i],tickFontSize) 
  
  if outFigureFile:
    fig12.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f.close()

#................................................................................#

if plot_moms:
  #[ Plot density, temperature and u_parallel along the field line for the kinetic elc sim,
  #[ with and without force softening.

  dataDir = ['/scratch/gpfs/manaurer/gkeyll/mirror/gk57/',
             '/scratch/gpfs/manaurer/gkeyll/nonuniformv/mirror/gz57_muMax3p0vt_vparMax16_muLinQuad_Nz192/',]
  fileName = ['gk57-wham1x2v_ion_gridDiagnostics_%d.bp',
              'gz57_1x2v_p1-ion_BiMaxwellianMoments_%d.gkyl']    #[ Root name of files to process.
  nFrames = [192, 400] #[ Number of frames in each sim.
  fileExt = ['bp','gkyl']
  fieldFile = ['_phi_','-field_']
  
  #.Root name of files to process.
  frame = 192

  figName = 'gk57-wham1x2v_M0TempUpar_'+str(frame)

  c_s = np.sqrt(Te0/mi)
  
  #[ Prepare figure.
  figProp16 = (7.,8.5)
  ax16Pos   = [[0.15, 0.750, 0.77, 0.215],
               [0.15, 0.525, 0.77, 0.215],
               [0.15, 0.300, 0.77, 0.215],
               [0.15, 0.075, 0.77, 0.215]]
  fig16     = plt.figure(figsize=figProp16)
  ax16      = [fig16.add_axes(d) for d in ax16Pos]

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  hpl16 = list()
  for fI in range(len(dataDir)):
    fName = dataDir[fI]+fileName[fI]    #.Complete file name.
  
    if fI == 0:
      #[ Load the electron density, temperatures and flows.
      den   = np.squeeze(pgu.getInterpData(fName % frame, polyOrder, basisType, varName='M0'))
      upar  = (1./c_s)*np.squeeze(pgu.getInterpData(fName % frame, polyOrder, basisType, varName='Upar'))
      tpar  = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % frame, polyOrder, basisType, varName='Tpar'))
      tperp = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % frame, polyOrder, basisType, varName='Tperp'))

      #[ Load the grid.
      xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % frame,polyOrder,basisType,location='center',varName='M0')
    else:
      den   = np.squeeze(pgu.getInterpData(fName % frame, polyOrder, basisType, comp=0))
      upar  = (1./c_s)*np.squeeze(pgu.getInterpData(fName % frame, polyOrder, basisType, comp=1))
      tpar  = (1.e-3/eV)*mi*np.squeeze(pgu.getInterpData(fName % frame, polyOrder, basisType, comp=2))
      tperp = (1.e-3/eV)*mi*np.squeeze(pgu.getInterpData(fName % frame, polyOrder, basisType, comp=3))

      #[ Load the grid.
      xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % frame,polyOrder,basisType,location='center')

    hpl16.append(ax16[0].semilogy(xIntC[0], den, color=defaultColors[fI], linestyle=lineStyles[fI], marker=markers[fI], markevery=12, markersize=4))
    hpl16.append(ax16[1].plot(xIntC[0], upar,    color=defaultColors[fI], linestyle=lineStyles[fI], marker=markers[fI], markevery=12, markersize=4))
    hpl16.append(ax16[2].plot(xIntC[0], tpar,    color=defaultColors[fI], linestyle=lineStyles[fI], marker=markers[fI], markevery=12, markersize=4))
    hpl16.append(ax16[3].plot(xIntC[0], tperp,   color=defaultColors[fI], linestyle=lineStyles[fI], marker=markers[fI], markevery=12, markersize=4))

    if saveData:
      fsStr = ''
      if fI == 1:
        fsStr = '_fs'
      h5f.create_dataset('den'+fsStr,   np.shape(den),      dtype='f8', data=den)
      h5f.create_dataset('upar'+fsStr,  np.shape(upar),     dtype='f8', data=upar)
      h5f.create_dataset('Tpar'+fsStr,  np.shape(tpar),     dtype='f8', data=tpar)
      h5f.create_dataset('Tperp'+fsStr, np.shape(tperp),    dtype='f8', data=tperp)

  ax16[0].set_ylim(1e14, 1e20)
  for i in range(len(ax16)):
    ax16[i].set_xlim( xIntC[0][0], xIntC[0][-1] )
    setTickFontSize(ax16[i],tickFontSize) 
    ax16[i].tick_params(axis='y', labelcolor='black', labelsize=tickFontSize)
    hmag = ax16[i].yaxis.get_offset_text().set_size(tickFontSize)
  for i in range(3):
    plt.setp( ax16[i].get_xticklabels(), visible=False)
  ax16[0].set_ylabel(r'$n_i$ (m$^{-3}$)', fontsize=xyLabelFontSize, color='black', labelpad=0)
  ax16[1].set_ylabel(r'$u_{\parallel i}/c_{se0}$', fontsize=xyLabelFontSize, color='black', labelpad=-2)
  ax16[2].set_ylabel(r'$T_{\parallel i}$ (keV)', fontsize=xyLabelFontSize, color='black')
  ax16[3].set_ylabel(r'$T_{\perp i}$ (keV)', fontsize=xyLabelFontSize, color='black')
  plt.text(0.025, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax16[0].transAxes)
  plt.text(0.025, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax16[1].transAxes)
  plt.text(0.025, 0.85, r'(c)', fontsize=textFontSize, color='black', transform=ax16[2].transAxes)
  plt.text(0.025, 0.85, r'(d)', fontsize=textFontSize, color='black', transform=ax16[3].transAxes)
  ax16[0].legend([hpl16[0][0], hpl16[4][0]],['uniform','nonuniform'], fontsize=legendFontSize, frameon=False)

  for i in range(len(ax16)):
    plot_verticalLinesPM(z_m, ax16[i])  #[ Indicate location of max B.
  ax16[3].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)

  if outFigureFile:
    plt.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f.create_dataset('z',           np.shape(xIntC[0]), dtype='f8', data=xIntC[0])
    h5f.close()

#................................................................................#
