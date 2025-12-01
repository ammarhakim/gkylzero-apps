import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#import sys
#import h5py
#import os

#...........................................................................#
#.
#.Post process cross-BGK validation test.
#.
#.Dingyun Liu.
#.
#...........................................................................#

dataDir  = '/scratch/gpfs/dingyunl/g0-main/gkylzero/output_cross/'
outDir  = '/scratch/gpfs/dingyunl/g0-main/gkylzero/output_cross/'
tests = ['gk_bgk_im', 'gk_bgk', 'gk_lbo']

plot_tempRelax     = True       #.Plot temperature relaxation.
plot_uparRelax     = True       #.Plot parallel speed relaxation.
plot_uparRelaxComp = False       #.Plot parallel speed relaxation for several tests.
plot_uparRelaxCompSeparate = False       #.Make separate plots for ions and elctrons.

outFigureFile    = True    #.Output a figure file?.
figureFileFormat = '.png'  #.Can be .png, .pdf, .ps, .eps, .svg.
saveData         = True    #.Indicate whether to save data in plot to HDF5 file.

#..................... NO MORE USER INPUTS BELOW (maybe) ....................#

mp = 1.672621637e-27  #.Proton mass (kg).
me = 9.10938215e-31   #.Electron mass (kg).
eV = 1.602176487e-19  #.Proton charge (C).
eps0 = 8.854187817620389850536563031710750260608e-12  #.Vacuum permitivity (F/m).
hbar = 6.62606896e-34/(2.*np.pi)  #.Planck's h bar (J s).
B0 = 1.0  #.Magnetic field amplitude (T).

basisType = 'ms'   #.'ms': modal serendipity, or 'ns': nodal serendipity.

#.Some RGB colors. These are MATLAB-like.
defaultBlue    = [0, 0.4470, 0.7410]
defaultOrange  = [0.8500, 0.3250, 0.0980]
defaultGreen   = [0.4660, 0.6740, 0.1880]
defaultPurple  = [0.4940, 0.1840, 0.5560]
defaultRed     = [0.6350, 0.0780, 0.1840]
defaultSkyBlue = [0.3010, 0.7450, 0.9330]
grey           = [0.5, 0.5, 0.5]
#.Colors in a single array.
defaultColors = [defaultBlue,defaultOrange,defaultGreen,defaultPurple,defaultRed,defaultSkyBlue,grey,'black']

#.LineStyles in a single array.
lineStyles = ['-','--',':','-.','None','None','None','None']
markers    = ['None','+','o','None','None','None','d','s']

#.Some fontsizes used in plots.
xyLabelFontSize       = 14
titleFontSize         = 16
tickFontSize          = 12
legendFontSize        = 12
textFontSize          = 14

#.Set the font size of the ticks to a given size.
def setTickFontSize(axIn,fontSizeIn):
  for tick in axIn.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontSizeIn)
  for tick in axIn.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontSizeIn)

#pgu.checkMkdir(outDir)

def coulombLog_sr(qs, qr, ms, mr, ns, nr, vts, vtr):
  #.Coulomb logarithm in Gkeyll (see online documentation):
  m_sr, u_sr = ms*mr/(ms+mr), np.sqrt(3.*np.power(vts,2)+3.*np.power(vtr,2))
  omega_ps, omega_pr = np.sqrt(ns*(qs**2)/(ms*eps0)), np.sqrt(nr*(qr**2)/(mr*eps0))
  omega_cs, omega_cr = np.abs(qs*B0/ms), np.abs(qr*B0/mr)
  rMax = 1./np.sqrt( np.divide((np.power(omega_ps,2)+omega_cs**2),(np.power(vts,2)+3.*np.power(vts,2)))
                    +np.divide((np.power(omega_pr,2)+omega_cr**2),(np.power(vtr,2)+3.*np.power(vts,2))) )
  rMin = np.maximum(np.divide(np.abs(qs*qr),(4.*np.pi*eps0*m_sr*np.power(u_sr,2))), np.divide(hbar,(2.*np.exp(0.5)*u_sr*m_sr)))
  return 0.5*np.log(1. + np.power((rMax/rMin),2))

def nu_ss(qs, ms, ns, vts): #.Like-species collision frequency.
  logLambda = coulombLog_sr(qs, qs, ms, ms, ns, ns, vts, vts)
  return (1./np.sqrt(2))*(qs**4)*ns*logLambda/(3.*((2.*np.pi)**(3./2.))*(eps0**2)*(ms**2)*np.power(vts,3))

def nuIso_sr(qs, qr, ms, mr, ns, nr, Tpars, Tperps, Tparr, Tperpr):  #.Isotropization rate (p. 33 in NRL)
  Ts, Tr   = (2.*Tperps+Tpars)/3., (2.*Tperpr+Tparr)/3.
  vts, vtr = np.sqrt(Ts/ms), np.sqrt(Tr/mr)
  A = Tperps/Tpars - 1.
  if np.isscalar(A):
    A = np.array([A])
  aF = np.zeros(np.size(A))
  for i in range(np.size(A)):
    if A[i] < 0.:
      aF[i] = np.arctanh(np.sqrt(-A[i]))/np.sqrt(-A[i])
    else:
      aF[i] = np.arctan(np.sqrt(A[i]))/np.sqrt(A[i])
  logLambda = 23.-np.log(np.sqrt(ns/1e6)/((Ts/eV)**1.5)) #coulombLog_sr(qs, qr, ms, mr, ns, nr, vts, vtr)
  return (2.*np.sqrt(np.pi)*((qs*qr)**2)*ns*logLambda/(((4.*np.pi*eps0)**2)*np.sqrt(ms*np.power(Tpars,3)))) \
        *np.power(A,-2)*(-3.+(A+3.)*aF)

def nu_S(qs, qr, ms, mr, ns, nr, vts, vtr): #.Flow relaxation frequency (eq.29 in Hager).
  if qs > 0.:
    logLambda = coulombLog_sr(qs, qr, ms, mr, ns, nr, vts, vtr)
    return ((qs*qr)**2)*ns*logLambda/(3.*((2.*np.pi)**(3./2.))*(eps0**2)*(mr**2)*np.power(vtr,3))
  else:
    logLambda = coulombLog_sr(qr, qs, mr, ms, nr, ns, vtr, vts)
    return ((qs*qr)**2)*nr*logLambda/(3.*((2.*np.pi)**(3./2.))*(eps0**2)*(ms**2)*np.power(vts,3))

#.......................................................................#
#.Compute the ion-ion collision frequency.
mi  = 2.014*mp                #.Ion mass (kg).
qi, qe  = eV, -eV             #.Charge (C).
den = 7.0e19           #.Density (1/m^3).
TiPar0 = 200.0*eV
TiPerp0 = 200.0*1.3*eV
TeIC = 300*eV*(1+1.3*2)/3.0 
TiIC = 200*eV*(1+1.3*2)/3.0
vtiIC = np.sqrt(TiIC/mi) 
cs = np.sqrt(TeIC/mi)
print('den=%10.8e, TiPar0=%10.8e, TiPerp0=%10.8e\n' % (den, TiPar0/eV, TiPerp0/eV))

#.......................................................................#
if plot_tempRelax:
  #.Plot the temperatures over time.

  figProp1a = (6.0, 4.5)
  #ax1aPos   = [[0.115, 0.13, 0.86, 0.83]]
  ax1aPos   = [[0.12, 0.12, 0.86, 0.80]]
  #figProp1a = (6.5, 6.5)
  #ax1aPos   = [[0.115, 0.13, 0.86, 0.83]]
  fig1a     = plt.figure(figsize=figProp1a)
  ax1a      = fig1a.add_axes(ax1aPos[0])

  #.Load and plot cross-BGK data.
  moms = np.load(dataDir+'moms.npy')
  time = np.load(dataDir+'time.npy')
 
  for i in [0,2]:
    Upare = moms[i,0,1] / moms[i,0,0]
    Upari = moms[i,1,1] / moms[i,1,0]
    Te = me/3.0 * (moms[i,0,2]-moms[i,0,1]*Upare) / moms[i,0,0]
    Ti = mi/3.0 * (moms[i,1,2]-moms[i,1,1]*Upari) / moms[i,1,0]
    TePerp = 1/2.0 * me/moms[i,0,0] * moms[i,0,4]
    TiPerp = 1/2.0 * mi/moms[i,1,0] * moms[i,1,4]
    TePar = 3*Te - 2*TePerp
    TiPar = 3*Ti - 2*TiPerp
    vti = np.sqrt(Ti/mi)          #.Ion thermal speed (m/s).
    vte = np.sqrt(Te/me)          #.Electron thermal speed (m/s).

    nu_ii = nu_ss(qi, mi, den, vtiIC) 
    #nu_ii = nuIso_sr(qi, qi, mi, mi, den, den, TiPar0, TiPerp0, TiPar0, TiPerp0)
    #print("nuRat = ", nu_ii)
    #print("Time = ", time*nu_ii)
  
    ax1a.semilogx(nu_ii*(time+0.), (1./eV)*TePar,  color=defaultColors[0], linestyle=lineStyles[i])
    ax1a.semilogx(nu_ii*(time+0.), (1./eV)*TePerp, color=defaultColors[1], linestyle=lineStyles[i])
    ax1a.semilogx(nu_ii*(time+0.), (1./eV)*TiPar,  color=defaultColors[2], linestyle=lineStyles[i])
    ax1a.semilogx(nu_ii*(time+0.), (1./eV)*TiPerp, color=defaultColors[3], linestyle=lineStyles[i])

  ax1a.legend([r'$T_{\parallel e,BGK,im}$',r'$T_{\perp e,BGK,im}$',r'$T_{\parallel i,BGK,im}$',r'$T_{\perp i,BGK,im}$',r'$T_{\parallel e,LBD,ex}$',r'$T_{\perp e,LBD,ex}$',r'$T_{\parallel i,LBD,ex}$',r'$T_{\perp i,LBD,ex}$'], fontsize=legendFontSize, frameon=False, bbox_to_anchor=(0.5, 0.54), ncol=2, columnspacing=2.0)

  ax1a.axis( (0.001, 500.0, 180, 400) )
  ax1a.text(-0.12, 1.02, 'b)', ha='left', va='bottom', transform=ax1a.transAxes, fontsize=titleFontSize, fontfamily='serif')
  ax1a.set_xlabel(r'time $(\nu_{ii}^{-1})$',fontsize=xyLabelFontSize, labelpad=-1)
  ax1a.set_ylabel(r'temperature $(eV)$',fontsize=xyLabelFontSize, labelpad=-1)
  #setTickFontSize(ax1a,tickFontSize)
  ax1a.xaxis.set_tick_params(labelsize=tickFontSize)
  ax1a.yaxis.set_tick_params(labelsize=tickFontSize)

  if outFigureFile:
    fig1a.savefig(outDir+'temperature_paper'+figureFileFormat)
  else:
    plt.show()

#.......................................................................#

if plot_uparRelax:
  #.Plot the parallel mean flow speed over time.

  figProp2a = (6.0, 4.5)
  #ax2aPos   = [[0.115, 0.13, 0.86, 0.83]]
  ax2aPos   = [[0.12, 0.12, 0.86, 0.80]]
  fig2a     = plt.figure(figsize=figProp2a)
  ax2a      = fig2a.add_axes(ax2aPos[0])

  #.Load and plot cross-BGK data.
  moms = np.load(dataDir+'moms.npy')
  time = np.load(dataDir+'time.npy')  

  for i in [0,2]:
    Upare = moms[i,0,1] / moms[i,0,0] / cs
    Upari = moms[i,1,1] / moms[i,1,0] / cs
    Te = me/3.0 * (moms[i,0,2]-moms[i,0,1]*Upare) / moms[i,0,0]
    Ti = mi/3.0 * (moms[i,1,2]-moms[i,1,1]*Upari) / moms[i,1,0]
    TePerp = 1/2.0 * me/moms[i,0,0] * moms[i,0,4]
    TiPerp = 1/2.0 * mi/moms[i,1,0] * moms[i,1,4]
    TePar = 3*Te - 2*TePerp
    TiPar = 3*Ti - 2*TiPerp
    vti = np.sqrt(Ti/mi)          #.Ion thermal speed (m/s).
    vte = np.sqrt(Te/me)          #.Electron thermal speed (m/s).

    nu_ii = nu_ss(qi, mi, den, vtiIC)
    #nu_ii = nuIso_sr(qi, qi, mi, mi, den, den, TiPar0, TiPerp0, TiPar0, TiPerp0)

    ax2a.semilogx(nu_ii*(time+0.), Upare, color=defaultColors[0], linestyle=lineStyles[i])
    ax2a.semilogx(nu_ii*(time+0.), Upari, color=defaultColors[1], linestyle=lineStyles[i])

  # zoom in to see upari
  axins = inset_axes(ax2a, width='100%', height='100%', loc=4,
                     bbox_to_anchor=(0.55, 0.1, 0.4, 0.3), bbox_transform=ax2a.transAxes)
  for i in range(len(tests)):
    Upari = moms[i,1,1] / moms[i,1,0] / cs
    Upare = moms[i,0,1] / moms[i,0,0] / cs
    axins.semilogx(nu_ii*(time+0.), Upare, color=defaultColors[0], linestyle=lineStyles[i])
    axins.semilogx(nu_ii*(time+0.), Upari, color=defaultColors[1], linestyle=lineStyles[i])
  axins.set_xlim(1.0e-3, 1.0)
  axins.set_ylim(0.011, 0.0112)
  mark_inset(ax2a, axins, loc1=2, loc2=4, fc="none", ec="0.5")

  ax2a.legend([r'$u_{\parallel e,BGK,im}$',r'$u_{\parallel i,BGK,im}$', r'$u_{\parallel e,LBD,ex}$',r'$u_{\parallel i,LBD,ex}$'], fontsize=legendFontSize, frameon=False, loc='upper right', ncol=2)

  ax2a.axis( (0.001, 500.0, 0., 0.5) ) #7.0e4
  ax2a.text(-0.12, 1.02, 'a)', ha='left', va='bottom', transform=ax1a.transAxes, fontsize=titleFontSize, fontfamily='serif')
  ax2a.set_xlabel(r'time $(\nu_{ii}^{-1})$',fontsize=xyLabelFontSize, labelpad=-1)
  ax2a.set_ylabel(r'mean parallel velocity $(c_s)$',fontsize=xyLabelFontSize, labelpad=-1)
  ax2a.set_yticks(np.linspace(0.,0.5,6)) #np.linspace(0.,6e4,7)
  #setTickFontSize(ax2a,tickFontSize)
  ax2a.xaxis.set_tick_params(labelsize=tickFontSize)
  ax2a.yaxis.set_tick_params(labelsize=tickFontSize)

  if outFigureFile:
    fig2a.savefig(outDir+'upar_paper'+figureFileFormat)
  else:
    plt.show()
