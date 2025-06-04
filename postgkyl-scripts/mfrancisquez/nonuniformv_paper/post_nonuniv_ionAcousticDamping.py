import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/mfrancis/Documents/codebits/pets/gkeyll/postProcessingScripts/')
import pgkylUtil as pgu
from scipy.optimize import curve_fit

#[ ..........................................................................#
#[ 
#[ Post process nonuniform ion acoustic damping tests.
#[ 
#[ Manaure Francisquez.
#[ 
#[ ..........................................................................#

dataDir = './'
outDir  = './'

simName = 'gk_ion_sound_adiabatic_elc_1x2v_p1'    #[ Root name of files to process.

plot_individual_fits   = False  #[ Linear fit to field energy for each sim.
plot_rates_vs_k        = False  #[ Damping rates vs. k.
plot_field_energy_comp = False  #[ Field energy for uni and nonuni.
plot_distF             = True  #[ Plot the distribution function.
plot_vmap              = False  #[ Plot velocity mapping.

outFigureFile    = True    #[ Output a figure file?.
figureFileFormat = '.eps'  #[ Can be .png, .pdf, .ps, .eps, .svg.
saveData         = False    #[ Indicate whether to save data in plot to HDF5 file.

#..................... NO MORE USER INPUTS BELOW (maybe) ....................#

basisType = 'ms'   #[ 'ms': modal serendipity, or 'ns': nodal serendipity.
polyOrder = 1

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
markers    = ['None','+','o','None','None','None','d','s']

#[ Some fontsizes used in plots.
xyLabelFontSize       = 17
titleFontSize         = 17
colorBarLabelFontSize = 17
tickFontSize          = 14
legendFontSize        = 14
textFontSize          = 16

#[ Set the font size of the ticks to a given size.
def setTickFontSize(axIn,fontSizeIn):
  axIn.tick_params(axis='both', which='major', labelsize=fontSizeIn)

pgu.checkMkdir(outDir)

def lineFit(x,a,b):
  return np.multiply(x,a) + b

#.This function finds the index of the grid point nearest to a given fix value.
def findNearestIndex(array,value):
  return (np.abs(array-value)).argmin()
#...end of findNearestIndex function...#

#.Function to find the local maxima of the field energy oscillations.
def locatePeaks(timeIn,energyIn,interval,floor):
  nT  = np.shape(timeIn)[0]
  tLo = findNearestIndex(timeIn,interval[0])
  tUp = findNearestIndex(timeIn,interval[1])

  energyMaxima      = np.empty(1)
  energyMaximaTimes = np.empty(1)
  for it in range(tLo,tUp):
    if (energyIn[it]>energyIn[it-1]) and (energyIn[it]>energyIn[it+1]) and (energyIn[it]>floor):
      energyMaxima      = np.append(energyMaxima,energyIn[it])
      energyMaximaTimes = np.append(energyMaximaTimes,timeIn[it])

  #.Don't return random first value introduced by np.empty.
  return energyMaximaTimes[1:], energyMaxima[1:]
#...end of locatePeaks function...#

#.......................................................................#

if plot_individual_fits:

  grid_types = ['uni/','nonuni/']

  file_names = ['wavek0p125/','wavek0p25/','wavek0p375/','wavek0p5/','wavek0p75/']

  waveks = [0.125, 0.25, 0.375, 0.5, 0.75]

  damp_calc_time_range = [
    [[0.50, 45.0],[1.0,26.5],[1.0,19.0],[1.0,13.5],[1.0,9.0]],
    [[1.0, 45.0],[1.0,29.5],[1.0,20.0],[1.0,15.0],[1.0,10.0]],
  ]

  sim_name = 'gk_ion_sound_adiabatic_elc_1x2v_p1-'

  for gI in range(len(grid_types)):

    for fi in range(len(file_names)):

      dir_name = grid_types[gI]+file_names[fi]
      
      print(' -> Processing '+dir_name+' <-')

      pgu.checkMkdir(dir_name+'/post/')

      #.Field energy.
      fName = dir_name + sim_name +'field_energy.gkyl'    #.Complete file name.
      time, field_energy = pgu.readDynVector(fName)

      #.Fit a line to the local maxima.
      #.Suppose the data obeys E^2 = (E0^2)*e^(-2*gamma*t). Take the log10 of both sides to get
      #.log10(E^2) = log10(E0^2)-2*gamma*log10(e)*t.
      #.So we can fit a linear polynomial to the log10(E^2) vs. t data to get gamma.
      fieldEMaximaTs, fieldEMaxima = locatePeaks(time,field_energy,damp_calc_time_range[gI][fi],0.0)
      fieldEMaximaLog10            = np.log10(fieldEMaxima)
      #.Fit a line to all the local maxima.
      poptMaxima, _  = curve_fit(lineFit, fieldEMaximaTs, fieldEMaximaLog10)
      gamma1         = -0.5*poptMaxima[0]/np.log10(np.exp(1))
      print('  gamma from line fit to maxima:       ', gamma1)
      #.Fit a line to even local maxima.
      poptMaximaE, _ = curve_fit(lineFit, fieldEMaximaTs[0::2], fieldEMaximaLog10[0::2])
      gammaE         = -0.5*poptMaximaE[0]/np.log10(np.exp(1))
      print('  gamma from line fit to even maxima:  ', gammaE)
#      #.Fit a line to odd local maxima.
#      print(np.shape(fieldEMaximaTs[1::2]), np.shape(fieldEMaximaLog10[1::2]))
#      poptMaximaO, _ = curve_fit(lineFit, fieldEMaximaTs[1::2], fieldEMaximaLog10[1::2])
#      gammaO         = -0.5*poptMaximaO[0]/np.log10(np.exp(1))
#      print('  gamma from line fit to odd maxima:  ', gammaO)

      #.Prepare figure.
      figProp2a = [6,4]
      ax2aPos   = [0.19, 0.14, 0.78, 0.8]
      fig2      = plt.figure(figsize=(figProp2a[0],figProp2a[1]))
      ax2a      = fig2.add_axes(ax2aPos)

      #.Plot field energy.
      hpl2a1 = ax2a.semilogy(time,field_energy,color=defaultBlue,linestyle=lineStyles[0])
      #.Plot line fit to maxima.
      hpl2a2 = ax2a.plot([fieldEMaximaTs[0],fieldEMaximaTs[-1]],
                         np.power(10,lineFit([fieldEMaximaTs[0],fieldEMaximaTs[-1]],*poptMaxima)),
                         color=defaultOrange,linestyle=lineStyles[1])
      #.Plot line fit to even maxima.
      hpl2a3 = ax2a.plot([fieldEMaximaTs[0],fieldEMaximaTs[-1]],
                         np.power(10,lineFit([fieldEMaximaTs[0],fieldEMaximaTs[-1]],*poptMaximaE)),
                         color=defaultGreen,linestyle=lineStyles[2])
#      #.Plot line fit to odd maxima.
#      hpl2a4 = ax2a.plot([fieldEMaximaTs[0],fieldEMaximaTs[-1]],
#                         np.power(10,lineFit([fieldEMaximaTs[0],fieldEMaximaTs[-1]],*poptMaximaO)),
#                         color=defaultPurple,linestyle=lineStyles[3])
#      ax2a.axis( (np.amin(time), np.amax(time), np.amin(field_energy)/10.0, np.amax(fieldE)*10.0) )
      ax2a.set_xlabel(r'$\omega_{ci} t$', fontsize=xyLabelFontSize)
      ax2a.set_ylabel(r'Field energy', fontsize=xyLabelFontSize)
#      ax2a.legend((hpl2a2[0],hpl2a3[0],hpl2a4[0]),
      ax2a.legend((hpl2a2[0],hpl2a3[0]),
                   (r'$\gamma$='+'{:9.3e}'.format(gamma1),
                    r'$\gamma$='+'{:9.3e}'.format(gammaE)+' (even maxima)',
 #                   r'$\gamma$='+'{:9.3e}'.format(gammaO)+' (odd maxima)'),
 ),
                    frameon=False, loc='upper right', fontsize=legendFontSize)
      setTickFontSize(ax2a,tickFontSize)
      if outFigureFile:
        plt.savefig(dir_name+'/post/'+simName+'_field_energy_fits'+figureFileFormat)
        plt.clf()
      else:
        plt.show()

#.................. FINISHED. PLOT INDIVIDUAL FITS ...........................#

if plot_rates_vs_k:

  grid_types = ['uni/','nonuni/']

  file_names = ['wavek0p125/','wavek0p25/','wavek0p375/','wavek0p5/','wavek0p75/']

  waveks = [0.125, 0.25, 0.375, 0.5, 0.75]

  damp_calc_time_range = [
    [[1.0, 45.0],[1.0,26.5],[1.0,19.0],[1.0,13.5],[1.0,9.0]],
    [[1.0, 45.0],[1.0,29.5],[1.0,20.0],[1.0,15.0],[1.0,10.0]],
  ]

  sim_name = 'gk_ion_sound_adiabatic_elc_1x2v_p1-'

  figProp4a = [6.2,3.2]
  ax4aPos   = [0.14, 0.18, 0.83, 0.79]
  fig4      = plt.figure(figsize=(figProp4a[0],figProp4a[1]))
  ax4a      = fig4.add_axes(ax4aPos)

  hpl4a = list()
  for gI in range(len(grid_types)):

    gamma = np.zeros(len(waveks))

    for fi in range(len(file_names)):

      dir_name = grid_types[gI]+file_names[fi]
      
      print(' -> Processing '+dir_name+' <-')

      #.Field energy.
      fName = dir_name + sim_name +'field_energy.gkyl'    #.Complete file name.
      time, field_energy = pgu.readDynVector(fName)

      #.Fit a line to the local maxima.
      #.Suppose the data obeys E^2 = (E0^2)*e^(-2*gamma*t). Take the log10 of both sides to get
      #.log10(E^2) = log10(E0^2)-2*gamma*log10(e)*t.
      #.So we can fit a linear polynomial to the log10(E^2) vs. t data to get gamma.
      fieldEMaximaTs, fieldEMaxima = locatePeaks(time,field_energy,damp_calc_time_range[gI][fi],0.0)
      fieldEMaximaLog10            = np.log10(fieldEMaxima)
      #.Fit a line to even local maxima.
      poptMaximaE, _ = curve_fit(lineFit, fieldEMaximaTs[0::2], fieldEMaximaLog10[0::2])
      gamma[fi]      = -0.5*poptMaximaE[0]/np.log10(np.exp(1))
      print('  gamma from line fit to even maxima:  ', gamma[fi])

    #.Plot field energy.
    hpl4a.append(ax4a.plot(waveks,gamma,color=defaultColors[gI],linestyle=lineStyles[gI],marker=markers[gI+1]))

#  ax4a.axis( (np.amin(time), np.amax(time), np.amin(field_energy)/10.0, np.amax(fieldE)*10.0) )
  ax4a.set_xlabel(r'$k_\parallel\rho_{i}$', fontsize=xyLabelFontSize, labelpad=-1)
  ax4a.set_ylabel(r'Damping rate, $\gamma/(k_\parallel v_{ti})$', fontsize=xyLabelFontSize)
  fig4.text(0.85, 0.9, '(b)', fontsize=textFontSize, color='black', transform=ax4a.transAxes)
  ax4a.legend((hpl4a[0][0],hpl4a[1][0]),(r'uniform',r'nonuniform'),
                frameon=False, loc='lower right', fontsize=legendFontSize)
  setTickFontSize(ax4a,tickFontSize)
  if outFigureFile:
    plt.savefig(outDir+simName+'_gamma_vs_k'+figureFileFormat)
    plt.clf()
  else:
    plt.show()

#............................................................................#

if plot_field_energy_comp:

  grid_types = ['uni/','nonuni/']

  file_name = 'wavek0p25/'

  waveks = 0.25

  sim_name = 'gk_ion_sound_adiabatic_elc_1x2v_p1-'

  #.Prepare figure.
  figProp2a = [6.2,3.2]
  ax2aPos   = [0.14, 0.18, 0.83, 0.79]
  fig2      = plt.figure(figsize=(figProp2a[0],figProp2a[1]))
  ax2a      = fig2.add_axes(ax2aPos)

  hpl2a = list()
  for gI in range(len(grid_types)):

    dir_name = grid_types[gI]+file_name
    
    #.Field energy.
    fName = dir_name + sim_name +'field_energy.gkyl'    #.Complete file name.
    time, field_energy = pgu.readDynVector(fName)

    #.Plot field energy.
    hpl2a.append(ax2a.semilogy(time,field_energy,color=defaultColors[gI],linestyle=lineStyles[gI]))

#  ax2a.axis( (np.amin(time), np.amax(time), np.amin(field_energy)/10.0, np.amax(fieldE)*10.0) )
  ax2a.set_xlim( np.amin(time), 30.0 )

  fig2.text(0.85, 0.9, '(a)', fontsize=textFontSize, color='black', transform=ax2a.transAxes)
  ax2a.set_xlabel(r'Time, $\omega_{ci}t$', fontsize=xyLabelFontSize, labelpad=-1)
  ax2a.set_ylabel(r'Normalized field energy', fontsize=xyLabelFontSize)
  ax2a.legend((hpl2a[0][0],hpl2a[1][0]),('uniform','nonuniform'),
              frameon=False, loc='lower left', fontsize=legendFontSize)
  ax2a.yaxis.set_label_coords(-.12,0.45)
  setTickFontSize(ax2a,tickFontSize)
  if outFigureFile:
    plt.savefig(outDir+simName+'field_energy_k0p25'+figureFileFormat)
  else:
    plt.show()

#............................................................................#

if plot_distF:
  #[ Plot initial distribution function.

  test = 'nonuni/wavek0p5/'
  fileName = test+'/'+simName+'-ion'

  figProp3a = (7., 4.0)
  ax3aPos   = [[0.09, 0.14, 0.73, 0.82]]
  cax3aPos  = [[0.83, 0.14, 0.02, 0.82]]
  fig3a     = plt.figure(figsize=figProp3a)
  ax3a      = [fig3a.add_axes(ax3aPos[0])]
  cbar_ax3a = [fig3a.add_axes(cax3aPos[0])]

  filePath = dataDir+fileName

  #[ Load the distribution times the Jacobian (J_v*f), and the Jacobian J_v, divide (J_vf)/J_v and interpolate.
  f_data = pg.GData(filePath + '_0.gkyl', mapc2p_vel_name=filePath + '_mapc2p_vel.gkyl')
  Jv_data = pg.GData(filePath + '_jacobvel.gkyl', mapc2p_vel_name=filePath + '_mapc2p_vel.gkyl')
  f_c = f_data.get_values()
  Jv_c = Jv_data.get_values()
  f_data._values = f_c/Jv_c
  dg = pg.GInterpModal(f_data, poly_order=polyOrder, basis_type="gkhyb")
  xInt, fInit = dg.interpolate()

  fInit_vpar_mu = np.squeeze(fInit)[0,:,:]
  
  #[ 2D nodal mesh.
  Xnodal = [np.outer(xInt[1], np.ones(np.shape(xInt[2]))), \
            np.outer(np.ones(np.shape(xInt[1])), xInt[2])]

  hpl3a = list()
  hpl3a.append(ax3a[0].pcolormesh(Xnodal[0], Xnodal[1], fInit_vpar_mu, cmap='inferno', edgecolors='grey',
            linewidth=0.1))

  cbar = [plt.colorbar(hpl3a[0],ax=ax3a[0],cax=cbar_ax3a[0])]
  cbar[0].ax.tick_params(labelsize=tickFontSize)
  cmag = cbar[0].ax.yaxis.get_offset_text().set_size(tickFontSize)
  cbar[0].set_label(r'$f_i(v_\parallel,\mu,t=0)$', rotation=90, labelpad=8, fontsize=16)
  ax3a[0].set_xlabel(r'$v_\parallel/v_{ti0}$',fontsize=xyLabelFontSize, labelpad=-2)
  ax3a[0].set_ylabel(r'$\mu/[m_iv_{ti0}^2/(2B)]$',fontsize=xyLabelFontSize, labelpad=-2)
  setTickFontSize(ax3a[0],tickFontSize)
  hmag = ax3a[0].xaxis.get_offset_text().set_size(tickFontSize)
  hmag = ax3a[0].yaxis.get_offset_text().set_size(tickFontSize)
  plt.text(0.05, 0.88, r'(b)', fontsize=textFontSize, color='white', weight='regular', transform=ax3a[0].transAxes)

  if outFigureFile:
    fig3a.savefig(outDir+'nonuni_ion_acoustic_damping-fInit'+figureFileFormat)
#    fig3a.savefig(outDir+'fig1'+figureFileFormat)
  else:
    plt.show()

  if saveData:
    #.Save data to HDF5 file:
#    h5f = h5py.File(outDir+'vmConservation-distF_initEnd.h5', "w")
    h5f = h5py.File(outDir+'fig1.h5', "w")
    h5f.create_dataset('elc_0', (nxInt[1]-1,nxInt[2]-1), dtype='f8', data=fElcInit)
    h5f.create_dataset('vxNodal', (nxInt[1],), dtype='f8', data=xInt[1])
    h5f.create_dataset('vyNodal', (nxInt[1],), dtype='f8', data=xInt[2])
    h5f.close()

#............................................................................#

if plot_vmap:

  figProp3a = (6.2,3.2)
  ax3aPos   = [0.14, 0.18, 0.83, 0.79]
  fig3      = plt.figure(figsize=figProp3a)
  ax3a      = fig3.add_axes(ax3aPos)

  def vpar_map(cvpar, vpar_lin_fac_inv, vpar_pow, vpar_max):
    vp = 0.0
    if (np.abs(cvpar) <= 1.0/vpar_lin_fac_inv):
      vp = vpar_max*cvpar
    elif (cvpar < -1.0/vpar_lin_fac_inv):
      vp = -vpar_max*np.power(vpar_lin_fac_inv,vpar_pow-1)*np.power(np.abs(cvpar),vpar_pow)
    else:
      vp =  vpar_max*np.power(vpar_lin_fac_inv,vpar_pow-1)*np.power(np.abs(cvpar),vpar_pow)

    return vp
  #[ ............................................. ]#

  Nvpar = 32
  vpar_max_ion = 5.0
  vpar_lin_fac_inv = 2.0
  vpar_pow = 2

  vpar_min_ion_c = -1.0/np.power(vpar_lin_fac_inv,(vpar_pow-1)/vpar_pow)
  vpar_max_ion_c =  1.0/np.power(vpar_lin_fac_inv,(vpar_pow-1)/vpar_pow)

  vpar_c = np.linspace(vpar_min_ion_c, vpar_max_ion_c, Nvpar)

  vpar = np.zeros(Nvpar)
  vpar_uni = np.zeros(Nvpar)
  for i in range(Nvpar):
    vpar[i] = vpar_map(vpar_c[i], vpar_lin_fac_inv, vpar_pow, vpar_max_ion)
    vpar_uni[i] = vpar_c[i]*vpar_max_ion

  hpl3a = list()
  hpl3a.append(ax3a.plot(vpar_c, vpar, color=defaultColors[0]))
  hpl3a.append(ax3a.plot(vpar_c, vpar_uni, color='grey', linestyle='--'))
  ax3a.axis( (vpar_c[0], vpar_c[-1], -6.0, 6.0) )
  yTickLabels = [str(-6+2*i)+r'$v_{ti0}$' for i in range(7)]
  yTickLabels[3] = '0'
  plt.yticks([-6+2*i for i in range(7)], yTickLabels)

  ax3a.set_xlabel(r'Computational $v_\parallel$ coordinate, $\eta$',fontsize=xyLabelFontSize, labelpad=-2)
  ax3a.set_ylabel(r'$v_\parallel(\eta)$',fontsize=xyLabelFontSize, labelpad=-2)
  plt.text(0.03, 0.88, r'(a)', fontsize=textFontSize, color='black', weight='regular', transform=ax3a.transAxes)
  setTickFontSize(ax3a,tickFontSize)

  if outFigureFile:
    fig3.savefig(outDir+'nonuni_ion_acoustic_damping-vpar_map'+figureFileFormat)
  else:
    plt.show()
