#[ ........................................................... ]#
#[
#[ post processing for nononuniform v asdex 2x2v simulations
#[
#[ Manaure Francisquez.
#[ April 2025.
#[
#[ ........................................................... ]#


import numpy as np
import postgkyl as pg
import matplotlib.pyplot as plt
import sys
#[ Append postgkyl wrappers.
sys.path.insert(0, '/home/manaurer/pets/gkeyll/postProcessingScripts/')
import pgkylUtil as pgu
from math import atan2,degrees

plot_ICs         = False  #[ Initial conditions.
plot_FinalCs     = False  #[ Final conditions.
plot_upari       = False  #[ Ion parallel velocity.
plot_jpar_plates = True  #[ Parallel current at the plates.

outDir = './'

outFigureFile    = False     #[ Output a figure file?.
figureFileFormat = '.eps'    #[ Can be .png, .pdf, .ps, .eps, .svg.
saveData         = False    #.Indicate whether to save data in plot to HDF5 file.

psi_axis = -9.276977e-02
psi_sep = 0.1497542827844
psi_min = 1.5218558641356e-01
psi_max = 1.6954424549161e-01

#[ ............... End of user inputs (MAYBE) ..................... ]#
0
polyOrder = 1
basisType = 'ms'

eps0, mu0 = 8.8541878176204e-12, 1.2566370614359e-06
eV        = 1.602176487e-19
qe, qi    = -1.602176487e-19, 1.602176487e-19
me, mp    = 9.10938215e-31, 1.672621637e-27

mi        = 2.014*mp                         #[ Deuterium ion mass.
Te0       = 37.5*eV
Ti0       = 38.0*eV
n0        = 0.4e19
B0        = (1.937830e+00+3.930574e+00)/2.0

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

#[ Normalized poloidal flux coordinate rho.
def rho_psi(psi, psisep, psi_ax):
  return np.sqrt((psi - psi_ax) / (psisep - psi_ax))

#[ Flux in terms of normalized poloidal flux rho.
def psi_rho(rho, psisep, psi_ax):
  return np.power(rho,2)*(psisep - psi_ax) + psi_ax

#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):
  ax = line.axes
  xdata = line.get_xdata()
  ydata = line.get_ydata()

  if (x < xdata[0]) or (x > xdata[-1]):
      print('x label location is outside data range!')
      return

  #Find corresponding y co-ordinate and angle of the line
  ip = 1
  for i in range(len(xdata)):
      if x < xdata[i]:
          ip = i
          break

  y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

  if not label:
      label = line.get_label()

  if align:
      #Compute the slope
      dx = xdata[ip] - xdata[ip-1]
      dy = ydata[ip] - ydata[ip-1]
      ang = degrees(atan2(dy,dx))

      #Transform to screen co-ordinates
      pt = np.array([x,y]).reshape((1,2))
      trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

  else:
      trans_angle = 0

  #Set a bunch of keyword arguments
  if 'color' not in kwargs:
      kwargs['color'] = line.get_color()

  if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
      kwargs['ha'] = 'center'

  if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
      kwargs['va'] = 'center'

  if 'backgroundcolor' not in kwargs:
      kwargs['backgroundcolor'] = ax.get_facecolor()

  if 'clip_on' not in kwargs:
      kwargs['clip_on'] = True

  if 'zorder' not in kwargs:
      kwargs['zorder'] = 2.5

  ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):
  ax = lines[0].axes
  labLines = []
  labels = []

  #Take only the lines which have labels other than the default ones
  for line in lines:
      label = line.get_label()
      if "_line" not in label:
          labLines.append(line)
          labels.append(label)

  if xvals is None:
      xmin,xmax = ax.get_xlim()
      xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

  for line,x,label in zip(labLines,xvals,labels):
      labelLine(line,x,label,align,**kwargs)

#................................................................................#

if plot_ICs:
  #[ Plot initial conditions.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/nonuniformv/asdex/gaug7/'

  den_exp_file = './michels2022data/Michels_2022-fig6_density_exp.csv'
  Te_exp_file = './michels2022data/Michels_2022-fig8_Te_exp.csv'

  fileName = 'gk_aug36190_sol_2x2v_p1'    #.Root name of files to process.
  frame    = 0

  filePath = dataDir+fileName+'-%s_BiMaxwellianMoments_'+str(frame)+'.gkyl'

  #[ Load the grid.
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(filePath % 'elc',polyOrder,basisType,location='center')

  #[ Get indices along z of slices we wish to plot:
  plotz = -np.pi/2
  plotzIdx = np.argmin(np.abs(xIntC[1]-plotz))

  #[ Load the data.
  den    = np.squeeze(pgu.getInterpData(filePath % 'elc', polyOrder, basisType, comp=0))
  temp_e = (me/eV)*np.squeeze((pgu.getInterpData(filePath % 'elc', polyOrder, basisType, comp=2)+
    2.0*pgu.getInterpData(filePath % 'elc', polyOrder, basisType, comp=2))/3.0)
  temp_i = (mi/eV)*np.squeeze((pgu.getInterpData(filePath % 'ion', polyOrder, basisType, comp=2)+
    2.0*pgu.getInterpData(filePath % 'ion', polyOrder, basisType, comp=2))/3.0)

  denSlice = den[:,plotzIdx]
  temp_eSlice = temp_e[:,plotzIdx]
  temp_iSlice = temp_i[:,plotzIdx]

  #[ Prepare figure.
  figProp9 = (6.4, 10.)
  ax9Pos   = [[0.12, 0.68, 0.86, 0.28],
              [0.12, 0.38, 0.86, 0.28],
              [0.12, 0.08, 0.86, 0.28],]
  fig9     = plt.figure(figsize=figProp9)
  ax9      = [fig9.add_axes(pos) for pos in ax9Pos]

  #[ Plot ICs.
  hpl9a = list()
  hpl9a.append(ax9[0].plot(xIntC[0], denSlice, color=defaultColors[0], linestyle=lineStyles[0], marker=markers[0]))

  hpl9b = list()
  hpl9b.append(ax9[1].plot(xIntC[0], temp_eSlice, color=defaultColors[0], linestyle=lineStyles[0], marker=markers[0]))

  hpl9c = list()
  hpl9c.append(ax9[2].plot(xIntC[0], temp_iSlice, color=defaultColors[0], linestyle=lineStyles[0], marker=markers[0]))

  #[ Plot experimental data.
  den_x_exp = np.loadtxt(open(den_exp_file),delimiter=',')
  den_exp_x = np.sort(den_x_exp[:,0])[-1::-1]
  den_exp_vals = 1e19*np.sort(den_x_exp[:,1])
  den_exp_psi = psi_rho(den_exp_x, psi_sep, psi_axis)

  Te_x_exp = np.loadtxt(open(Te_exp_file),delimiter=',')
  Te_exp_x = np.sort(Te_x_exp[:,0])[-1::-1]
  Te_exp_vals = np.sort(Te_x_exp[:,1])
  Te_exp_psi = psi_rho(Te_exp_x, psi_sep, psi_axis)

  ax9[0].plot(den_exp_psi,den_exp_vals, linestyle=lineStyles[1], color='grey')
  ax9[1].plot(Te_exp_psi,Te_exp_vals, linestyle=lineStyles[1], color='grey')

  ax9[2].set_xlabel(r'Poloidal flux, $\psi$ (T m$^{2}$)', fontsize=xyLabelFontSize, labelpad=-2)
  ax9[0].set_ylabel(r'$n_s(t=0)$ (m$^{-3}$)', fontsize=xyLabelFontSize)
  ax9[1].set_ylabel(r'$T_e(t=0)$ (eV)', fontsize=xyLabelFontSize)
  ax9[2].set_ylabel(r'$T_i(t=0)$ (eV)', fontsize=xyLabelFontSize)
  ax9[0].yaxis.get_offset_text().set_size(tickFontSize)
  ax9[0].set_ylim(0.0, 0.80e19)
  ax9[1].set_ylim(0.0, 80.0)
  ax9[2].set_ylim(0.0, 70.0)
  for i in range(3):
    setTickFontSize(ax9[i],tickFontSize)
    ax9[i].set_xlim(xIntC[0][0], xIntC[0][-1])
  for i in range(0,2):
    plt.setp( ax9[i].get_xticklabels(), visible=False)

  plt.text(0.92, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax9[0].transAxes)
  plt.text(0.92, 0.85, r'(c)', fontsize=textFontSize, color='black', transform=ax9[1].transAxes)
  plt.text(0.92, 0.85, r'(d)', fontsize=textFontSize, color='black', transform=ax9[2].transAxes)

  if outFigureFile:
    fig9.savefig(outDir+fileName+'-BiMaxwellianMoments_'+str(frame)+'_z1eqmpiD2'+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#


if plot_FinalCs:
  #[ Plot initial conditions.

  dataDir = [
    '/scratch/gpfs/manaurer/gkeyll/nonuniformv/asdex/gaug11/',
    '/scratch/gpfs/manaurer/gkeyll/nonuniformv/asdex/gaug12/',
    '/scratch/gpfs/manaurer/gkeyll/nonuniformv/asdex/gaug13/',
  ]

  Dcoeff = [
    '0.2',
    '0.3',
    '0.4',
  ]

  den_exp_file = './michels2022data/Michels_2022-fig6_density_exp.csv'
  Te_exp_file = './michels2022data/Michels_2022-fig8_Te_exp.csv'

  fileName = 'gk_aug36190_sol_2x2v_p1'    #.Root name of files to process.
#  frame    = 748
#  frame    = 952
  frame    = 2000

  #[ Prepare figure.
  figProp9 = (12.4, 3.2)
  ax9Pos   = [[0.045, 0.18, 0.28, 0.74],[0.38, 0.18, 0.28, 0.74],[0.715, 0.18, 0.28, 0.74],]
  fig9     = plt.figure(figsize=figProp9)
  ax9      = [fig9.add_axes(pos) for pos in ax9Pos]
  
  hpl9a = list()
  hpl9b = list()
  hpl9c = list()
  for fI in range(len(dataDir)):
    filePath = dataDir[fI]+fileName+'-%s_BiMaxwellianMoments_'+str(frame)+'.gkyl'
  
    #[ Load the grid.
    xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(filePath % 'elc',polyOrder,basisType,location='center')
  
    #[ Get indices along z of slices we wish to plot:
    plotz = -np.pi/2
    plotzIdx = np.argmin(np.abs(xIntC[1]-plotz))
  
    #[ Load the data.
    den    = np.squeeze(pgu.getInterpData(filePath % 'elc', polyOrder, basisType, comp=0))
    temp_e = (me/eV)*np.squeeze((pgu.getInterpData(filePath % 'elc', polyOrder, basisType, comp=2)+
      2.0*pgu.getInterpData(filePath % 'elc', polyOrder, basisType, comp=2))/3.0)
    temp_i = (mi/eV)*np.squeeze((pgu.getInterpData(filePath % 'ion', polyOrder, basisType, comp=2)+
      2.0*pgu.getInterpData(filePath % 'ion', polyOrder, basisType, comp=2))/3.0)
  
    denSlice = den[:,plotzIdx]
    temp_eSlice = temp_e[:,plotzIdx]
    temp_iSlice = temp_i[:,plotzIdx]
  
    #[ Plot ICs.
    hpl9a.append(ax9[0].plot(xIntC[0], denSlice, color=defaultColors[fI], linestyle=lineStyles[0], marker=markers[0], label=Dcoeff[fI]))
  
    hpl9b.append(ax9[1].plot(xIntC[0], temp_eSlice, color=defaultColors[fI], linestyle=lineStyles[0], marker=markers[0], label=Dcoeff[fI]))
  
    hpl9c.append(ax9[2].plot(xIntC[0], temp_iSlice, color=defaultColors[fI], linestyle=lineStyles[0], marker=markers[0], label=Dcoeff[fI]))

  labelLines(ax9[0].get_lines(), backgroundcolor="none")
  labelLines(ax9[1].get_lines(), backgroundcolor="none")
  labelLines(ax9[2].get_lines(), backgroundcolor="none")

  #[ Plot experimental data.
  den_x_exp = np.loadtxt(open(den_exp_file),delimiter=',')
  den_exp_x = np.sort(den_x_exp[:,0])[-1::-1]
  den_exp_vals = 1e19*np.sort(den_x_exp[:,1])
  den_exp_psi = psi_rho(den_exp_x, psi_sep, psi_axis)

  Te_x_exp = np.loadtxt(open(Te_exp_file),delimiter=',')
  Te_exp_x = np.sort(Te_x_exp[:,0])[-1::-1]
  Te_exp_vals = np.sort(Te_x_exp[:,1])
  Te_exp_psi = psi_rho(Te_exp_x, psi_sep, psi_axis)

  ax9[0].plot(den_exp_psi,den_exp_vals, linestyle=lineStyles[1], color='grey')
  ax9[1].plot(Te_exp_psi,Te_exp_vals, linestyle=lineStyles[1], color='grey')

  ax9[0].set_ylabel(r'$n_e$ (m$^{-3}$)', fontsize=xyLabelFontSize, labelpad=-2)
  ax9[1].set_ylabel(r'$T_e$ (eV)', fontsize=xyLabelFontSize, labelpad=-2)
  ax9[2].set_ylabel(r'$T_i$ (eV)', fontsize=xyLabelFontSize, labelpad=-2)
  ax9[0].yaxis.get_offset_text().set_size(tickFontSize)
  ax9[0].set_ylim(0.0, 0.80e19)
  ax9[1].set_ylim(0.0, 80.0)
  ax9[2].set_ylim(0.0, 80.0)
  for i in range(3):
    ax9[i].set_xlabel(r'Poloidal flux, $\psi$ (T m$^{2}$)', fontsize=xyLabelFontSize, labelpad=-2)
    setTickFontSize(ax9[i],tickFontSize)
    ax9[i].set_xlim(xIntC[0][0], xIntC[0][-1])

  plt.text(0.87, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax9[0].transAxes)
  plt.text(0.87, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax9[1].transAxes)
  plt.text(0.87, 0.85, r'(c)', fontsize=textFontSize, color='black', transform=ax9[2].transAxes)

  if outFigureFile:
    fig9.savefig(outDir+fileName+'-BiMaxwellianMoments_'+str(frame)+'_z1eqmpiD2'+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_upari:
  #[ Plot the ion parallel velocity normalized to the sound speed.

#  dataDir = '/scratch/gpfs/manaurer/gkeyll/nonuniformv/asdex/gaug7/'
  dataDir = '/scratch/gpfs/manaurer/gkeyll/nonuniformv/asdex/gaug11/'

  fileName = 'gk_aug36190_sol_2x2v_p1'    #.Root name of files to process.
#  frame    = 748
#  frame    = 952
  frame    = 2000

  filePath = dataDir+fileName+'-%s_BiMaxwellianMoments_'+str(frame)+'.gkyl'

  #[ Load the grid.
  xInt, _, nxInt, lxInt, dxInt, _ = pgu.getGrid(filePath % 'elc',polyOrder,basisType)
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(filePath % 'elc',polyOrder,basisType,location='center')

  #[ Load the data.
  upar_i = np.squeeze(pgu.getInterpData(filePath % 'ion', polyOrder, basisType, comp=1))
  temp_e = me*np.squeeze((pgu.getInterpData(filePath % 'elc', polyOrder, basisType, comp=2)+
    2.0*pgu.getInterpData(filePath % 'elc', polyOrder, basisType, comp=2))/3.0)
  temp_i = mi*np.squeeze((pgu.getInterpData(filePath % 'ion', polyOrder, basisType, comp=2)+
    2.0*pgu.getInterpData(filePath % 'ion', polyOrder, basisType, comp=2))/3.0)

  cs = np.sqrt((temp_e+temp_i)/mi)

  Ma = upar_i/cs

  #[ Prepare figure.
  figProp9 = (6.5, 4.)
  ax9Pos   = [[0.12, 0.145, 0.72, 0.84],]
  ca9Pos   = [[0.845, 0.145, 0.02, 0.84],]
  fig9     = plt.figure(figsize=figProp9)
  ax9      = [fig9.add_axes(pos) for pos in ax9Pos]
  cbar_ax9 = [fig9.add_axes(pos) for pos in ca9Pos]

  #[ Create colorplot grid. Recall coordinates have to be nodal.
  Xnodal = [np.outer(xInt[0], np.ones(nxInt[1])),
            np.outer(np.ones(nxInt[0]), xInt[1])]

  hpl9 = list()
  hpl9.append(ax9[0].pcolormesh(Xnodal[0], Xnodal[1], Ma, cmap='inferno'))

  hcb9 = list()
  hcb9.append(plt.colorbar(hpl9[0], ax=ax9[0], cax=cbar_ax9[0]))
  hcb9[0].ax.tick_params(labelsize=tickFontSize)

  ax9[0].set_xticks(np.linspace(xInt[0][0], xInt[0][-1], 6))
  ax9[0].set_xlabel(r'Poloidal flux, $\psi$ (T m$^{2}$)', fontsize=xyLabelFontSize, labelpad=0)
  ax9[0].set_ylabel(r'Poloidal arc length, $\theta$', fontsize=xyLabelFontSize, labelpad=-2)
  hcb9[0].set_label('$u_{\parallel i}/c_s$', rotation=90, labelpad=0, fontsize=colorBarLabelFontSize)
  setTickFontSize(ax9[0],tickFontSize)
  plt.text(0.9, 0.9, '(a)', fontsize=textFontSize, color='black', transform=ax9[0].transAxes)

  if outFigureFile:
    fig9.savefig(outDir+fileName+'-ion_upari_'+str(frame)+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_jpar_plates:
  #[ Plot the parallel current at the plates.

#  dataDir = '/scratch/gpfs/manaurer/gkeyll/nonuniformv/asdex/gaug7/'
  dataDir = '/scratch/gpfs/manaurer/gkeyll/nonuniformv/asdex/gaug11/'

  fileName = 'gk_aug36190_sol_2x2v_p1'    #.Root name of files to process.
#  frame    = 748
#  frame    = 952
  frame    = 2000

  filePath = dataDir+fileName+'-%s_M1_'+str(frame)+'.gkyl'

  #[ Load the grid.
  xInt, _, nxInt, lxInt, dxInt, _ = pgu.getGrid(filePath % 'elc',polyOrder,basisType)
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(filePath % 'elc',polyOrder,basisType,location='center')

  #[ Load the data.
  M1_i = np.squeeze(pgu.getInterpData(filePath % 'ion', polyOrder, basisType))
  M1_e = np.squeeze(pgu.getInterpData(filePath % 'elc', polyOrder, basisType))

  jpar_in = qi*M1_i[:,-1] + qe*M1_e[:,-1]
  jpar_out = qi*M1_i[:,0] + qe*M1_e[:,0]

  #[ Prepare figure.
  figProp9 = (6.4, 4.)
  ax9Pos   = [[0.145, 0.14, 0.81, 0.85],]
  fig9     = plt.figure(figsize=figProp9)
  ax9      = [fig9.add_axes(pos) for pos in ax9Pos]

  #[ Plot ICs.
  hpl9a = list()
  hpl9a.append(ax9[0].plot(xIntC[0], jpar_in/1e6, color=defaultColors[0], linestyle=lineStyles[0], marker=markers[0]))
  hpl9a.append(ax9[0].plot(xIntC[0], jpar_out/1e6, color=defaultColors[1], linestyle=lineStyles[1], marker=markers[1]))

  ax9[0].set_xticks(np.linspace(xIntC[0][0], xIntC[0][-1], 6))
  ax9[0].set_xlabel(r'Poloidal flux, $\psi$ (T m$^2$)', fontsize=xyLabelFontSize, labelpad=-2)
  ax9[0].set_ylabel(r'$j_\parallel(\theta=\pm\pi)$ (MA/m$^2$)', fontsize=xyLabelFontSize, labelpad=-2)
#  ax9[0].yaxis.get_offset_text().set_size(tickFontSize)
  ax9[0].set_xlim(xIntC[0][0], xIntC[0][-1])
  setTickFontSize(ax9[0],tickFontSize)
  plt.text(0.9, 0.9, r'(b)', fontsize=textFontSize, color='black', transform=ax9[0].transAxes)

  if outFigureFile:
    fig9.savefig(outDir+fileName+'-jpar_'+str(frame)+'_z1eqpmpi'+figureFileFormat)
    plt.close()
  else:
    plt.show()
