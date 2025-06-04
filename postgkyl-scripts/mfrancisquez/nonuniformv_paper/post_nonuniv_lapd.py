#[ ........................................................... ]#
#[
#[ post processing for nonuniform velocity LAPD 3x2v simulations.
#[
#[
#[ Manaure Francisquez.
#[ April 2025
#[
#[ ........................................................... ]#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from matplotlib.patches import ConnectionPatch
import matplotlib as mpl
import sys
#[ Append postgkyl wrappers.
sys.path.insert(0, '/global/homes/m/mana/pets/gkeyll/postProcessingScripts/')
import pgkylUtil as pgu
import postgkyl as pg

plot_nT_r            = False  #[ Radial n and T profiles.
plot_n_rtheta_comp   = False  #[ r-theta plot of n for various biases.
plot_vE_theta_r_comp = True  #[ Azimuthal component of the ExB drift vs. r for one or more simulations.
plot_nT_r_comp       = False  #[ Radial n and T profiles for one or more simulations.

outDir = './'

outFigureFile    = True     #[ Output a figure file?.
figureFileFormat = '.eps'    #[ Can be .png, .pdf, .ps, .eps, .svg.
saveData         = False    #.Indicate whether to save data in plot to HDF5 file.

#[ ............... End of user inputs (MAYBE) ..................... ]#

polyOrder = 1
basisType = 'ms'

eps0, mu0 = 8.8541878176204e-12, 1.2566370614359e-06
eV        = 1.602176487e-19
qe, qi    = -1.602176487e-19, 1.602176487e-19
me, mp    = 9.10938215e-31, 1.672621637e-27

mRat      = 400       #[ Ion to electron mass ratio.
mi        = 3.973*mp  #[ Helium ion mass.
me        = mi/mRat
Te0       = 6*eV
Ti0       = 1*eV
n0        = 2.e18
B0        = 0.0398
gas_gamma = 5.0/3.0

r_s = 0.25

#[ Thermal speeds.
vti = np.sqrt(Ti0/mi)
vte = np.sqrt(Te0/me)
c_s = np.sqrt(Te0/mi)

#[ Gyrofrequencies and gyroradii.
omega_ci = eV*B0/mi
rho_s    = c_s/omega_ci

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
markers    = ['None','None','None','None','+','o','d','s']

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

#................................................................................#

if plot_nT_r:
  #[ Plot radial profiles of density and temperature.
#  dataDir = '/scratch/gpfs/tqian/mirror/lapd/32/'
#  fileName = 'ls3-lapd-3x2v-p1'    #.Root name of files to process.
  dataDir = '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like/'
  fileName = 'gk_lapd_nonuniv_3x2v_p1'    #.Root name of files to process.

  #[ Initial and final frame, and max radius, to compute stats for.
  initFrame = 400
  endFrame  = 600
  avgz      = [-4.49, 4.49]

  fName = dataDir+fileName+'-%s_BiMaxwellianMoments_%d.gkyl'    #.Complete file name.

  #[ Load the grid.
  xIntC_e, _, nxIntC_e, lxIntC_e, dxIntC_e, _ = pgu.getGrid(fName % ('elc',0),polyOrder,basisType,location='center')

  #[ Indices in between which we will average.
  avgzIdx = [np.argmin(np.abs(xIntC_e[2]-val)) for val in avgz]

  m0_avg = np.zeros(nxIntC_e[0])
  te_avg = np.zeros(nxIntC_e[0])
  ti_avg = np.zeros(nxIntC_e[0])
  for nFr in range(initFrame,endFrame+1):

    m0 =         np.squeeze(pgu.getInterpData(fName % ('elc',nFr), polyOrder, basisType, comp=0))
    te = (me/eV)*np.squeeze(pgu.getInterpData(fName % ('elc',nFr), polyOrder, basisType, comp=2))
    ti = (mi/eV)*np.squeeze(pgu.getInterpData(fName % ('ion',nFr), polyOrder, basisType, comp=2))
    #[ There may be transient large negative temperatures. Floor them.
    te = np.where(te>0., te, 0.)
    ti = np.where(ti>0., ti, 0.)

    #[ Average over theta and z.
    m0_avg += np.average(m0[:,:,avgzIdx[0]:avgzIdx[1]+1],axis=(1,2))
    te_avg += np.average(te[:,:,avgzIdx[0]:avgzIdx[1]+1],axis=(1,2))
    ti_avg += np.average(ti[:,:,avgzIdx[0]:avgzIdx[1]+1],axis=(1,2))

  #[ Divide by the number of frames.
  m0_avg = m0_avg/(endFrame-initFrame+1)
  te_avg = te_avg/(endFrame-initFrame+1)
  ti_avg = ti_avg/(endFrame-initFrame+1)

  #[ Prepare figure.
  figProp1 = (12., 3.6)
  ax1Pos   = [[0.095, 0.15, 0.25, 0.78],[0.4075, 0.15, 0.25, 0.78],[0.74, 0.15, 0.25, 0.78],]
  fig1     = plt.figure(figsize=figProp1)
  ax1      = [fig1.add_axes(pos) for pos in ax1Pos]

  hpl1a, hpl1b, hpl1c = list(), list(), list()
  hpl1a.append(ax1[0].plot(xIntC_e[0]*100., m0_avg))
  hpl1b.append(ax1[1].plot(xIntC_e[0]*100., te_avg))
  hpl1c.append(ax1[2].plot(xIntC_e[0]*100., ti_avg))

  ax1[0].set_ylabel(r'$\left\langle n_e\right\rangle_{\theta,z,t}~(\mathrm{m}^{-3})$', fontsize=xyLabelFontSize)
  ax1[1].set_ylabel(r'$\left\langle T_e\right\rangle_{\theta,z,t}$ (eV)', fontsize=xyLabelFontSize)
  ax1[2].set_ylabel(r'$\left\langle T_i\right\rangle_{\theta,z,t}$ (eV)', fontsize=xyLabelFontSize)
  for i in range(3):
    ax1[i].set_xlabel(r'$r$ (cm)', fontsize=xyLabelFontSize, labelpad=-2)
    ax1[i].set_xlim(6.8, xIntC_e[0][-1]*100.)
    ax1[i].set_ylim(0., ax1[i].get_ylim()[1])
#    ax1[i].axvspan(0., r_s*100., alpha=0.2, color='grey')
    ax1[i].plot([r_s*100, r_s*100],[0., ax1[i].get_ylim()[1]], linestyle='--', color='grey')
    setTickFontSize(ax1[0+i],tickFontSize)
    hmagx = ax1[i].yaxis.get_offset_text().set_size(tickFontSize)
  plt.text(0.82, 0.88, r'(a)', fontsize=textFontSize, color='black', fontweight='regular', transform=ax1[0].transAxes)
  plt.text(0.82, 0.88, r'(b)', fontsize=textFontSize, color='black', fontweight='regular', transform=ax1[1].transAxes)
  plt.text(0.82, 0.88, r'(c)', fontsize=textFontSize, color='black', fontweight='regular', transform=ax1[2].transAxes)

  if outFigureFile:
    fig1.savefig(outDir+fileName+'_nT_profiles'+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_n_rtheta_comp:
  #[ Plot n on the r-theta plane for a couple of biases.
  #[ shi-like_bias2p025/  shi-like_bias9p825/  shi-like_biasm3p075/ shi-like_biasm7p275/

  plot_z = 0.0

  species = 'elc'
  frame = [
    600,
    300,
    300,
  ]
  dataDir = [
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like/',
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like_biasm7p275/',
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like_bias9p825/',
  ]

  simName = 'gk_lapd_nonuniv_3x2v_p1-'
  c2pFile = simName+'mapc2p.gkyl'
  fileName = simName+species+'_BiMaxwellianMoments_%d.gkyl'

  #[ Prepare figure.
  figProp2 = (10., 3.8)
  ax2Pos   = [[0.06, 0.18, 0.3, 0.7],[0.335, 0.18, 0.3, 0.7],[0.61, 0.18, 0.3, 0.7],]
  cax2Pos  = [0.9, 0.18, 0.02, 0.7]
  fig2     = plt.figure(figsize=figProp2)
  ax2      = [fig2.add_axes(pos) for pos in ax2Pos]
  cbar_ax2 = fig2.add_axes(cax2Pos)

  #[ Find min and max values.
  m0_eEx = [9.99e29, -9.99e29]
  for fI in range(len(dataDir)):
    fName = dataDir[fI]+fileName    #.Complete file name.

    #[ Load the density.
    m0_data = pg.GData(fName % frame[fI], mapc2p_name=dataDir[fI]+c2pFile)
    dg = pg.GInterpModal(m0_data, poly_order=polyOrder, basis_type=basisType)
    xInt_e, m0_e = dg.interpolate()
    xInt_e = np.squeeze(xInt_e)
    m0_e = np.squeeze(m0_e)

    #[ z index of the slice to plot.
    plot_zIdx = np.argmin(np.abs(xInt_e[2]-plot_z))

    m0_eSlice = m0_e[:,:,plot_zIdx]

    m0_eEx = [np.amin([m0_eEx[0],np.amin(m0_eSlice)]), np.amax([m0_eEx[1],np.amax(m0_eSlice)])]

  #[ Plot density.
  hpl2 = list()
  for fI in range(len(dataDir)):
    fName = dataDir[fI]+fileName    #.Complete file name.

    #[ Load the density.
    m0_data = pg.GData(fName % frame[fI], mapc2p_name=dataDir[fI]+c2pFile)
    dg = pg.GInterpModal(m0_data, poly_order=polyOrder, basis_type=basisType)
    xInt_e, m0_e = dg.interpolate()
    xInt_e = np.squeeze(xInt_e)
    m0_e = np.squeeze(m0_e)

    #[ z index of the slice to plot.
    plot_zIdx = np.argmin(np.abs(xInt_e[2]-plot_z))

    m0_eSlice = m0_e[:,:,plot_zIdx]
    Xnodal = [xInt_e[0,:,:,plot_zIdx], xInt_e[1,:,:,plot_zIdx]]

    hpl2.append(ax2[fI].pcolormesh(Xnodal[0], Xnodal[1], m0_eSlice, cmap='inferno', vmin=m0_eEx[0], vmax=m0_eEx[1]))
    ax2[fI].set_aspect('equal')

  hcb2 = plt.colorbar(hpl2[-1], ax=ax2[-1], cax=cbar_ax2)
  hcb2.ax.tick_params(labelsize=tickFontSize)
  hcb2.set_label('$n_e$ (m$^{-3}$)', rotation=90, labelpad=0, fontsize=colorBarLabelFontSize)
  hcb2.ax.yaxis.get_offset_text().set_fontsize(tickFontSize)
  ax2[0].set_ylabel(r'$Y$ (m)', fontsize=xyLabelFontSize, labelpad=-4)
  for sI in range(1,len(ax2)):
    plt.setp( ax2[sI].get_yticklabels(), visible=False)
  for sI in range(len(ax2)):
    ax2[sI].set_xlabel(r'$X$ (m)', fontsize=xyLabelFontSize, labelpad=2)
#  plt.text(0.7, 0.88, r'Uniform', fontsize=14, color='black', fontweight='regular', transform=ax2[0].transAxes)
#  plt.text(0.7, 0.88, r'Nonuniform' , fontsize=14, color='white', fontweight='regular', transform=ax2[1].transAxes)
  bbox_style = dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.65)
  plt.text(0.62, 0.88, r'$\phi_{\mathrm{wall}}=0$', fontsize=textFontSize, color='black', bbox=bbox_style, transform=ax2[0].transAxes)
  plt.text(0.3, 0.88, r'$\phi_{\mathrm{wall}}=-7.28$ V', fontsize=textFontSize, color='black', bbox=bbox_style, transform=ax2[1].transAxes)
  plt.text(0.4, 0.88, r'$\phi_{\mathrm{wall}}=9.83$ V', fontsize=textFontSize, color='black', bbox=bbox_style, transform=ax2[2].transAxes)
  plt.text(0.025, 0.05, r'(a)', fontsize=textFontSize, color='black', transform=ax2[0].transAxes)
  plt.text(0.025, 0.05, r'(b)', fontsize=textFontSize, color='black', transform=ax2[1].transAxes)
  plt.text(0.025, 0.05, r'(c)', fontsize=textFontSize, color='black', transform=ax2[2].transAxes)
  for i in range(len(ax2)):
#    ax2[i].set_xlim( timeLims[0], timeLims[-1]*1.005 )
    setTickFontSize(ax2[i],tickFontSize)

  if outFigureFile:
    fig2.savefig(outDir+simName+'_n_snapshot_bias_comp'+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_vE_theta_r_comp:
  #[ Plot the azimuthal component of the ExB drift for biased and unbiased.
  #[ Recall:
  #[   hat{r} = hat{x}cos(theta) + hat{y}sin(theta)
  #[   hat{theta} = -hat{x}sin(theta) + hat{y}cos(theta)
  #[   r = sqrt(x^2 + y^2)
  #[   theta = arctan(y/x)
  dataDir = [
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like_biasm7p275/',
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like_biasm3p075/',
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like/',
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like_bias2p025/',
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like_bias9p825/',
  ]

  simName = 'gk_lapd_nonuniv_3x2v_p1-'

  legendStrs = [
    r'$\phi_{\mathrm{wall}}=-7.28$ V',
    r'$\phi_{\mathrm{wall}}=-3.08$ V',
    r'$\phi_{\mathrm{wall}}=0$',
    r'$\phi_{\mathrm{wall}}=2.03$ V',
    r'$\phi_{\mathrm{wall}}=9.83$ V'
  ]

  #[ Initial and final frame, and max radius, to compute stats for.
  initFrame = [
    200,
    200,
    500,
    200,
    200,
  ]
  endFrame  = [
    300,
    300,
    600,
    300,
    300,
  ]
  avgz = [-4.49, 4.49]

  fName = [dataDir[i]+simName+'field_%d.gkyl' for i in range(len(dataDir))]    #.Complete file name.

  #[ Load the grid.
  xInt_e, _, nxInt_e, lxInt_e, dxInt_e, _ = pgu.getGrid(fName[0] % 0,polyOrder,basisType)
  xIntC_e, _, nxIntC_e, lxIntC_e, dxIntC_e, _ = pgu.getGrid(fName[0] % 0,polyOrder,basisType,location='center')

  #[ Indices in between which we will average.
  avgzIdx = [np.argmin(np.abs(xIntC_e[2]-val)) for val in avgz]

  vE_theta = np.zeros(nxIntC_e)

  #[ Prepare figure.
  figProp1 = (6., 3.6)
  ax1Pos   = [[0.14, 0.15, 0.84, 0.78],]
  fig1     = plt.figure(figsize=figProp1)
  ax1      = [fig1.add_axes(pos) for pos in ax1Pos]
  hpl1a = list()
  hForLegend = list()

  for fI in range(len(dataDir)):
    vE_theta_avg = np.zeros(nxIntC_e[0])
    for nFr in range(initFrame[fI],endFrame[fI]+1):

      phi = np.squeeze(pgu.getInterpData(fName[fI] % nFr, polyOrder, basisType))

      Ex = - np.gradient(phi, xIntC_e[0], axis=0)

      #[ v_E = ExB/B^2 = hat{x}Ey/B0 - hat{y}Ex/B0.
      for k in range(nxIntC_e[2]):
        vE_theta[:,:,k] = - Ex[:,:,k]/B0

      #[ Average over theta and z.
      vE_theta_avg += np.average(vE_theta[:,:,avgzIdx[0]:avgzIdx[1]+1],axis=(1,2))

    #[ Divide by the number of frames.
    vE_theta_avg = vE_theta_avg/(endFrame[fI]-initFrame[fI]+1)

    #[ Plot it with the same orientation as D. Schaffner, et al PRL 109, 135002 (2012).
    hpl1a.append(ax1[0].plot(xIntC_e[0]*100, -vE_theta_avg*100/1e5, color=defaultColors[fI], linestyle=lineStyles[fI], marker=markers[fI]))
    hForLegend.append(hpl1a[-1][0])

  #[ Place arrow indicating IDD/EDD:
  ax1[0].annotate('', xy=(11., 1.8), xytext=(11, 0.25),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=0.5))
  plt.text(0.07, 0.58, r'IDD', fontsize=textFontSize, color='black', transform=ax1[0].transAxes)
  ax1[0].annotate('', xy=(11., -1.8), xytext=(11, -0.25),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=0.5))
  plt.text(0.07, 0.35, r'EDD', fontsize=textFontSize, color='black', transform=ax1[0].transAxes)

  ax1[0].plot([0., 100.], [0., 0.], linestyle=':', color='grey')
  ax1[0].set_ylabel(r'$\left\langle v_{E,\theta}\right\rangle_{\theta,z,t}~(10^5~\mathrm{cm/s})$', fontsize=xyLabelFontSize)
  ax1[0].set_xlim(10., 35.0)
  ax1[0].set_ylim(-4.0, 4.0)
#  ax1[0].axvspan(0., r_s*100., alpha=0.2, color='grey')
  ax1[0].set_xlabel(r'$r$ (cm)', fontsize=xyLabelFontSize, labelpad=-2)
#  ax1[0].set_ylim(0., ax1[0].get_ylim()[1])
  setTickFontSize(ax1[0],tickFontSize)
  hmagx = ax1[0].yaxis.get_offset_text().set_size(tickFontSize)
  ax1[0].legend(hForLegend,legendStrs, fontsize=legendFontSize, frameon=False)

  if outFigureFile:
    fig1.savefig(outDir+simName+'_vE_theta_vs_r_bias_comp'+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_nT_r_comp:
  #[ Plot radial profiles of density and temperature for biased and un-biased cases.
  dataDir = [
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like_biasm7p275/',
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like_biasm3p075/',
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like/',
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like_bias2p025/',
    '/pscratch/sd/m/mana/gkeyll/nonuniv/lapd/shi-like_bias9p825/',
  ]

  simName = 'gk_lapd_nonuniv_3x2v_p1-'

  legendStrs = [
    r'$\phi_{\mathrm{wall}}=-7.28$ V',
    r'$\phi_{\mathrm{wall}}=-3.08$ V',
    r'$\phi_{\mathrm{wall}}=0$',
    r'$\phi_{\mathrm{wall}}=2.03$ V',
    r'$\phi_{\mathrm{wall}}=9.83$ V'
  ]

  #[ Initial and final frame, and max radius, to compute stats for.
  initFrame = [
    200,
    200,
    500,
    200,
    200,
  ]
  endFrame  = [
    300,
    300,
    600,
    300,
    300,
  ]
  avgz = [-4.49, 4.49]

  fName = [dataDir[i]+simName+'%s_BiMaxwellianMoments_%d.gkyl' for i in range(len(dataDir))]    #.Complete file name.

  #[ Load the grid.
  xInt_e, _, nxInt_e, lxInt_e, dxInt_e, _ = pgu.getGrid(fName[0] % ('elc',0),polyOrder,basisType)
  xIntC_e, _, nxIntC_e, lxIntC_e, dxIntC_e, _ = pgu.getGrid(fName[0] % ('elc',0),polyOrder,basisType,location='center')

  #[ Indices in between which we will average.
  avgzIdx = [np.argmin(np.abs(xIntC_e[2]-val)) for val in avgz]

  #[ Prepare figure.
  figProp1 = (12., 3.6)
  ax1Pos   = [[0.08, 0.15, 0.25, 0.78],[0.40, 0.15, 0.25, 0.78],[0.72, 0.15, 0.25, 0.78],]
  fig1     = plt.figure(figsize=figProp1)
  ax1      = [fig1.add_axes(pos) for pos in ax1Pos]
  hpl1a, hpl1b, hpl1c = list(), list(), list()
  hForLegend = list()

  for fI in range(len(dataDir)):

    m0_avg = np.zeros(nxIntC_e[0])
    te_avg = np.zeros(nxIntC_e[0])
    ti_avg = np.zeros(nxIntC_e[0])

    for nFr in range(initFrame[fI],endFrame[fI]+1):

      m0 =         np.squeeze(pgu.getInterpData(fName[fI] % ('elc',nFr), polyOrder, basisType, comp=0))
      te = (me/eV)*np.squeeze(pgu.getInterpData(fName[fI] % ('elc',nFr), polyOrder, basisType, comp=2))
      ti = (mi/eV)*np.squeeze(pgu.getInterpData(fName[fI] % ('ion',nFr), polyOrder, basisType, comp=2))
      #[ There may be transient large negative temperatures. Floor them.
      te = np.where(te>0., te, 0.)
      ti = np.where(ti>0., ti, 0.)

      #[ Average over theta and z.
      m0_avg += np.average(m0[:,:,avgzIdx[0]:avgzIdx[1]+1],axis=(1,2))
      te_avg += np.average(te[:,:,avgzIdx[0]:avgzIdx[1]+1],axis=(1,2))
      ti_avg += np.average(ti[:,:,avgzIdx[0]:avgzIdx[1]+1],axis=(1,2))

    #[ Divide by the number of frames.
    m0_avg = m0_avg/(endFrame[fI]-initFrame[fI]+1)
    te_avg = te_avg/(endFrame[fI]-initFrame[fI]+1)
    ti_avg = ti_avg/(endFrame[fI]-initFrame[fI]+1)

    hpl1a.append(ax1[0].plot(xIntC_e[0]*100., m0_avg))
    hpl1b.append(ax1[1].plot(xIntC_e[0]*100., te_avg))
    hpl1c.append(ax1[2].plot(xIntC_e[0]*100., ti_avg))
    hForLegend.append(hpl1a[-1][0])

  ax1[0].set_ylabel(r'$n_e~(\mathrm{m}^{-3})$', fontsize=xyLabelFontSize)
  ax1[1].set_ylabel(r'$T_e$ (eV)', fontsize=xyLabelFontSize)
  ax1[2].set_ylabel(r'$T_i$ (eV)', fontsize=xyLabelFontSize)
  ax1[-0].legend(hForLegend,legendStrs, fontsize=legendFontSize, frameon=False)
  for i in range(3):
    ax1[i].set_xlim(0., xIntC_e[0][-1]*100.)
#    ax1[i].axvspan(0., r_s*100., alpha=0.2, color='grey')
    ax1[i].plot([r_s*100, r_s*100],[0., ax1[i].get_ylim()[1]], linestyle='--', color='grey')
    ax1[i].set_xlabel(r'$r$ (cm)', fontsize=xyLabelFontSize, labelpad=-2)
    ax1[i].set_ylim(0., ax1[i].get_ylim()[1])
    setTickFontSize(ax1[0+i],tickFontSize)
    hmagx = ax1[i].yaxis.get_offset_text().set_size(tickFontSize)

  if outFigureFile:
    fig1.savefig(outDir+'lapd-3x2v-p1_nT_profiles'+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#
