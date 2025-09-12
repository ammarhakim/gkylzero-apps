#[ ........................................................... ]#
#[
#[ Check particle balance
#[
#[ Manaure Francisquez.
#[ Dec 2024
#[
#[ ........................................................... ]#


import numpy as np
import postgkyl as pg
import matplotlib.pyplot as plt
import sys
import pgkylUtil as pgu
import os

dataDir = './simdata/sh12/'
outDir = './'

simName = 'sh12'
cdim = 2
blocks = [0,1,2,3,4,5,6,7,8,9,10,11]

outFigureFile    = False   #[ Output a figure file?.
figureFileFormat = '.png'  #[ Can be .png, .pdf, .ps, .eps, .svg.
saveData         = False   #.Indicate whether to save data in plot to HDF5 file.


#[ ............... End of user inputs (MAYBE) ..................... ]#

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

#.Set the font size of the ticks to a given size.
def setTickFontSize(axIn,fontSizeIn):
  axIn.tick_params(axis='both',labelsize=fontSizeIn)
  offset_txt = axIn.yaxis.get_offset_text() # Get the text object
  offset_txt.set_size(fontSizeIn) # # Set the size.
  offset_txt = axIn.xaxis.get_offset_text() # Get the text object
  offset_txt.set_size(fontSizeIn) # # Set the size.

#[ Strings for selecting boundary.
bcDir = ['x', 'y', 'z']
bcEdge = ['lower','upper']

#[ ............... End common utilities ..................... ]#

#[ Read the M0 moment of df/dt, the source and the particle fluxes.

dataPath = dataDir + simName

species = 'elc'

for bI in range(len(blocks)):
  file = dataPath + '_b' + str(blocks[bI]) + '-' + species

  time_fdot, fdot_pb = pgu.readDynVector(file + '_fdot_integrated_moms.gkyl')
  time_bflux_lo, bflux_lo_pb = list(range(cdim)), list(range(cdim))
  time_bflux_up, bflux_up_pb = list(range(cdim)), list(range(cdim))
  for d in range(cdim):
    loread = file + '_bflux_' + bcDir[d] + 'lower_integrated_HamiltonianMoments.gkyl'
    upread = file + '_bflux_' + bcDir[d] + 'upper_integrated_HamiltonianMoments.gkyl'
    if os.path.exists(loread):
        time_bflux_lo[d], bflux_lo_pb[d] = pgu.readDynVector(loread)
    else:
        time_bflux_lo[d], bflux_lo_pb[d] = None, None
    if os.path.exists(upread):
        time_bflux_up[d], bflux_up_pb[d] = pgu.readDynVector(upread)
    else:
        time_bflux_up[d], bflux_up_pb[d] = None, None

  sread = file + '_source_integrated_moms.gkyl'
  if os.path.exists(sread):
    time_src, src_pb = pgu.readDynVector(sread)
  else:
    time_src, src_pb = None, None
  
  if(src_pb is not None):
    src_pb[0,:] = 0.0 #[ Set source=0 at t=0 since we don't have fdot and bflux then.
  
  #[ Select the M0 moment.
  fdot_pb = fdot_pb[:,0]
  for d in range(cdim):
    if(bflux_lo_pb[d] is not None):
        bflux_lo_pb[d] = bflux_lo_pb[d][:,0]
    if(bflux_up_pb[d] is not None):
        bflux_up_pb[d] = bflux_up_pb[d][:,0]
  if (src_pb is not None):
    src_pb = src_pb[:,0]
  
  bflux_tot_pb = np.zeros(np.size(bflux_lo_pb[-1])) #[ Total boundary flux loss.
  for d in range(cdim):
    if(bflux_lo_pb[d] is not None):
        bflux_tot_pb += bflux_lo_pb[d]
    if(bflux_up_pb[d] is not None):
        bflux_tot_pb+=bflux_up_pb[d]

  #[ Add over blocks.
  if bI == 0:
    fdot = fdot_pb
    bflux_tot = bflux_tot_pb
  else:
    fdot += fdot_pb
    bflux_tot += bflux_tot_pb

  if bI==10:
    src = src_pb
  elif bI==11:
    src += src_pb

mom_err = src - bflux_tot - fdot #[ Error.

#[ Plot each contribution.
figProp1a = (7.5, 4.5)
ax1aPos   = [0.09, 0.15, 0.87, 0.8]
fig1a     = plt.figure(figsize=figProp1a)
ax1a      = fig1a.add_axes(ax1aPos)

hpl1a = list()
hpl1a.append(ax1a.plot([-1.0,1.0], [0.0,0.0], color='grey', linestyle=':', linewidth=1))
hpl1a.append(ax1a.plot(time_fdot, fdot, color=defaultColors[0], linestyle=lineStyles[0], linewidth=2))
hpl1a.append(ax1a.plot(time_bflux_lo[-1], -bflux_tot, color=defaultColors[1], linestyle=lineStyles[1], linewidth=2))
hpl1a.append(ax1a.plot(time_src, src, color=defaultColors[2], linestyle=lineStyles[2], linewidth=2))
hpl1a.append(ax1a.plot(time_fdot, mom_err, color=defaultColors[3], linestyle=lineStyles[3], linewidth=2))
ax1a.set_xlabel(r'Time ($s$)',fontsize=xyLabelFontSize, labelpad=+4)
ax1a.set_ylabel(r'0-th  moment',fontsize=xyLabelFontSize, labelpad=0)
ax1a.set_xlim( time_fdot[0], time_fdot[-1] )
legendStrings = [r'$\dot{f}$',r'$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$',r'$\mathcal{S}$',
r'$E_{\dot{\mathcal{N}}}=\mathcal{S}-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f-\dot{f}$',
]
ax1a.legend([hpl1a[i][0] for i in range(1,len(hpl1a))], legendStrings, fontsize=legendFontSize, frameon=False)
setTickFontSize(ax1a,tickFontSize)

if outFigureFile:
  plt.savefig(outFileRoot+fileSuffix+'_fld_z0eq0p0'+figureFileFormat)
else:
  plt.show()

#[ Plot the error normalized for different time steps.
#cfl_dirs = ['cfl0p125/', 'cfl0p25/', 'cfl0p5/']
#legendStrings = [r'CFL=0.125', r'CFL=0.25', r'CFL=0.5']
cfl_dirs = ['']
legendStrings = [r'CFL=1.0']

eV = 1.602e-19
me = 0.91e-30
Lz = 4.0
n0 = 7.0e18
N0 = 9e17 #[ Reference number of particles
Te = 40.0*eV
vte = np.sqrt(Te/me)
tau = 0.5*Lz/vte #[ Transit time.

figProp2a = (7.5, 4.5)
ax2aPos   = [0.09, 0.15, 0.87, 0.8]
fig2a     = plt.figure(figsize=figProp2a)
ax2a      = fig2a.add_axes(ax2aPos)

hpl2a = list()
hpl2a.append(ax2a.plot([-1.0,1.0], [0.0,0.0], color='grey', linestyle=':', linewidth=1))
ylabelString = ""

for dI in range(len(cfl_dirs)):
  dataPath = dataDir + cfl_dirs[dI] + simName
  
  time_dt, dt = pgu.readDynVector(dataPath + '-dt.gkyl')

  for bI in range(len(blocks)):
    file = dataPath + '_b' + str(blocks[bI]) + '-' + species
  
    time_fdot, fdot_pb = pgu.readDynVector(file + '_fdot_integrated_moms.gkyl')
    time_bflux_lo, bflux_lo_pb = list(range(cdim)), list(range(cdim))
    time_bflux_up, bflux_up_pb = list(range(cdim)), list(range(cdim))
    for d in range(cdim):
      loread = file + '_bflux_' + bcDir[d] + 'lower_integrated_HamiltonianMoments.gkyl'
      upread = file + '_bflux_' + bcDir[d] + 'upper_integrated_HamiltonianMoments.gkyl'
      if os.path.exists(loread):
        time_bflux_lo[d], bflux_lo_pb[d] = pgu.readDynVector(loread)
      else:
        time_bflux_lo[d], bflux_lo_pb[d] = None,None
      if os.path.exists(upread):
        time_bflux_up[d], bflux_up_pb[d] = pgu.readDynVector(upread)
      else:
        time_bflux_up[d], bflux_up_pb[d] = None,None
    sread = file + '_source_integrated_moms.gkyl'
    fread = file + '_integrated_moms.gkyl'
    if os.path.exists(sread):
        time_src, src_pb = pgu.readDynVector(sread)
    else:
        time_src, src_pb = None, None
    if os.path.exists(fread):
        time_distf, distf_pb = pgu.readDynVector(fread)
    else:
        time_distf, distf_pb = None, None
    
    #[ Select the M0 moment and remove the t=0 data point.
    fdot_pb = fdot_pb[1:,0]
    for d in range(cdim):
      if  bflux_lo_pb[d] is not None:
        bflux_lo_pb[d] = bflux_lo_pb[d][1:,0]
      if  bflux_up_pb[d] is not None:
        bflux_up_pb[d] = bflux_up_pb[d][1:,0]
    if src_pb is not None:
        src_pb = src_pb[1:,0]
    distf_pb = distf_pb[1:,0]
    
    bflux_tot_pb = np.zeros(np.size(bflux_lo_pb[-1])) #[ Total boundary flux loss.
    for d in range(cdim):
        if(bflux_lo_pb[d] is not None):
            bflux_tot_pb += bflux_lo_pb[d]
        if(bflux_up_pb[d] is not None):
            bflux_tot_pb+=bflux_up_pb[d]

    #[ Add over blocks.
    if bI == 0:
      fdot = fdot_pb
      bflux_tot = bflux_tot_pb
      distf = distf_pb
    else:
      fdot += fdot_pb
      bflux_tot += bflux_tot_pb
      distf += distf_pb

    if bI == 10:
      src = src_pb
    elif bI == 11:
      src += src_pb

  mom_err = src - bflux_tot - fdot #[ Error.
  #print(time_fdot[1], time_bflux_lo[0][1], time_bflux_up[0][1], time_src[1], time_distf[1])
  #print(mom_err[0], src[0], bflux_tot[0], fdot[0])

#  mom_err_norm = mom_err
#  ylabelString = r'$E_{\dot{\mathcal{N}}}$'
  mom_err_norm = mom_err *dt/distf
  ylabelString = r'$E_{\dot{\mathcal{N}}}~\Delta t/\mathcal{N}$'
#  mom_err_norm = mom_err*tau/N0
#  ylabelString = r'$E_{\dot{\mathcal{N}}}~\tau/\mathcal{N}_0$'
  print(np.mean(mom_err_norm))
  
  hpl2a.append(ax2a.plot(time_dt, mom_err_norm, color=defaultColors[dI], linestyle=lineStyles[dI], linewidth=2))

ax2a.set_xlabel(r'Time ($s$)',fontsize=xyLabelFontSize, labelpad=+4)
ax2a.set_ylabel(ylabelString,fontsize=xyLabelFontSize, labelpad=0)
ax2a.set_xlim( time_fdot[0], time_fdot[-1] )
ax2a.legend([hpl2a[i][0] for i in range(1,len(hpl2a))], legendStrings, fontsize=legendFontSize, frameon=False)
setTickFontSize(ax2a,tickFontSize)

if outFigureFile:
  plt.savefig(outFileRoot+fileSuffix+'_fld_z0eq0p0'+figureFileFormat)
else:
  plt.show()

