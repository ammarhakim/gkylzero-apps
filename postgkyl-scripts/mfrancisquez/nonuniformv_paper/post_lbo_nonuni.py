import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/mfrancis/Documents/codebits/pets/gkeyll/postProcessingScripts/')
import pgkylUtil as pgu
from matplotlib.legend_handler import HandlerBase

#[ ..........................................................................#
#[ 
#[ Post process nonuniform LBO tests.
#[ 
#[ Manaure Francisquez.
#[ 
#[ ..........................................................................#

dataDir = './'
outDir  = './'

simName = 'gk_lbo_relax_nonuniv_1x2v_p1'    #[ Root name of files to process.
polyOrder = 1

B0 = 1.0 #[ Magnetic field amplitude (T).

plot_tempRelax = False #[ Plot temperature relaxation.
plot_Merrors   = True #[ Plot conservation errors as a function of resolution.
plot_distF     = False #[ Plot the distribution function.

outFigureFile    = False    #[ Output a figure file?.
figureFileFormat = '.png'  #[ Can be .png, .pdf, .ps, .eps, .svg.
saveData         = False    #[ Indicate whether to save data in plot to HDF5 file.

#..................... NO MORE USER INPUTS BELOW (maybe) ....................#

mp = 1.672621637e-27  #[ Proton mass (kg).
me = 9.10938215e-31   #[ Electron mass (kg).
eV = 1.602176487e-19  #[ Proton charge (C).
eps0 = 8.854187817620389850536563031710750260608e-12  #[ Vacuum permitivity (F/m).
hbar = 6.62606896e-34/(2.*np.pi)  #[ Planck's h bar (J s).

basisType = 'ms'   #[ 'ms': modal serendipity, or 'ns': nodal serendipity.

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

def coulombLog_sr(qs, qr, ms, mr, ns, nr, vts, vtr):
  #[ Coulomb logarithm in Gkeyll (see online documentation):
  m_sr, u_sr = ms*mr/(ms+mr), np.sqrt(3.*np.power(vts,2)+3.*np.power(vtr,2))
  omega_ps, omega_pr = np.sqrt(ns*(qs**2)/(ms*eps0)), np.sqrt(nr*(qr**2)/(mr*eps0))
  omega_cs, omega_cr = np.abs(qs*B0/ms), np.abs(qr*B0/mr)
  rMax = 1./np.sqrt( np.divide((np.power(omega_ps,2)+omega_cs**2),(np.power(vts,2)+3.*np.power(vts,2)))
                    +np.divide((np.power(omega_pr,2)+omega_cr**2),(np.power(vtr,2)+3.*np.power(vts,2))) )
  rMin = np.maximum(np.divide(np.abs(qs*qr),(4.*np.pi*eps0*m_sr*np.power(u_sr,2))), np.divide(hbar,(2.*np.exp(0.5)*u_sr*m_sr)))
  return 0.5*np.log(1. + np.power((rMax/rMin),2))

def nu_ss(qs, ms, ns, vts): #[ Like-species collision frequency.
  logLambda = coulombLog_sr(qs, qs, ms, ms, ns, ns, vts, vts)
  return (1./np.sqrt(2))*(qs**4)*ns*logLambda/(3.*((2.*np.pi)**(3./2.))*(eps0**2)*(ms**2)*np.power(vts,3))

def nuIso_sr(qs, qr, ms, mr, ns, nr, Tpars, Tperps, Tparr, Tperpr):  #[ Isotropization rate (p. 33 in NRL)
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

def nu_S(qs, qr, ms, mr, ns, nr, vts, vtr): #[ Flow relaxation frequency (eq.29 in Hager).
  if qs > 0.:
    logLambda = coulombLog_sr(qs, qr, ms, mr, ns, nr, vts, vtr)
    return ((qs*qr)**2)*ns*logLambda/(3.*((2.*np.pi)**(3./2.))*(eps0**2)*(mr**2)*np.power(vtr,3))
  else:
    logLambda = coulombLog_sr(qr, qs, mr, ms, nr, ns, vtr, vts)
    return ((qs*qr)**2)*nr*logLambda/(3.*((2.*np.pi)**(3./2.))*(eps0**2)*(ms**2)*np.power(vts,3))

#.A handler class used for multi-line legends in plots.
class AnyObjectHandler(HandlerBase):
  def create_artists(self, legend, orig_handle,
                     x0, y0, width, height, fontsize, trans):
    l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                    linestyle=orig_handle[0], marker=orig_handle[1], color=orig_handle[2])
    l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
                    linestyle=orig_handle[3], marker=orig_handle[4], color=orig_handle[5])
    return [l1, l2]

#.......................................................................#

if plot_tempRelax:
  #[ Plot the temperatures over time.
  tests = ['glb0','glb26']

  figProp1a = (6.5, 4.5)
  ax1aPos   = [[0.115, 0.13, 0.86, 0.83]]
  fig1a     = plt.figure(figsize=figProp1a)
  ax1a      = fig1a.add_axes(ax1aPos[0])

  for dI in range(len(tests)):
    #[ Load and plot Gkeyll data.
    fileRoot = dataDir+'/'+tests[dI]+'/'+simName+'-%s_BiMaxwellianMoments_'
  
    nFrames = 1+pgu.findLastFrame(fileRoot % 'elc','gkyl')
  
    m0e = np.zeros(nFrames)
    TePar, TePerp = np.zeros(nFrames), np.zeros(nFrames)
  
    time = pgu.getTimeStamps(fileRoot % 'elc','gkyl',0,nFrames-1)  #[ Load times.
  
    for fI in range(nFrames):  #.Load temperatures.
      m0e[fI]    = np.squeeze(pgu.getInterpData(fileRoot % 'elc' +str(fI)+'.gkyl',polyOrder,basisType,comp=0))[0]
      TePar[fI]  = me*np.squeeze(pgu.getInterpData(fileRoot % 'elc' +str(fI)+'.gkyl',polyOrder,basisType,comp=2))[0]
      TePerp[fI] = me*np.squeeze(pgu.getInterpData(fileRoot % 'elc' +str(fI)+'.gkyl',polyOrder,basisType,comp=3))[0]
  
    #[ Compute the elc-elc collision frequency.
    qi, qe = eV, -eV              #[ Charge (C).
    den = m0e                     #[ Density (1/m^3).
    Te  = (2.*TePerp + TePar)/3.  #[ Electron temperature (J).
    vte = np.sqrt(Te/me)          #[ Electron thermal speed (m/s).
    nu_ee = nu_ss(qe, me, den[0], vte[0])
  
    hpl1a = list()
    hpl1a.append(ax1a.plot(nu_ee*(time+0.), (1/eV)*TePar,  color=defaultColors[dI], linestyle=lineStyles[0]))
    hpl1a.append(ax1a.plot(nu_ee*(time+0.), (1/eV)*TePerp, color=defaultColors[dI], linestyle=lineStyles[1]))

#  ax1a.legend([r'$T_{\parallel e}$',r'$T_{\perp e}$'], fontsize=legendFontSize, frameon=False, loc='center')
  legendStrings = ['Uniform','Nonuniform',r'$T_{\parallel e}$',r'$T_{\perp e}$']
  ax1a.legend([(lineStyles[0], 'None',defaultColors[0],lineStyles[1],'None',defaultColors[0]), \
                  (lineStyles[0], 'None',defaultColors[1],lineStyles[1],'None',defaultColors[1]), \
                  (lineStyles[0], 'None',defaultColors[0],lineStyles[0],'None',defaultColors[1]), \
                  (lineStyles[1], 'None',defaultColors[0],lineStyles[1],'None',defaultColors[1])], legendStrings, \
                 handler_map={tuple: AnyObjectHandler()}, fontsize=legendFontSize, frameon=False, loc='lower right')

  ax1a.axis( (0.0, 5.0, 290, 400) )
  ax1a.set_xlabel(r'Time, $\nu_{ee}t$',fontsize=xyLabelFontSize, labelpad=-1)
  ax1a.set_ylabel(r'Temperature (eV)',fontsize=xyLabelFontSize, labelpad=-1)
  fig1a.text(0.9, 0.9, r'(b)', fontsize=textFontSize, color='black', weight='regular', transform=ax1a.transAxes)
  setTickFontSize(ax1a,tickFontSize)

  if outFigureFile:
    fig1a.savefig(outDir+'nonuni_lbo-tempRelax'+figureFileFormat)
#    fig1a.savefig(outDir+'fig6a'+figureFileFormat)
  else:
    plt.show()

  #[ Save data to HDF5 file:
  if saveData:
    figName = 'fig6c'
    opName = 'lbo-et'
    h5f = h5py.File(outDir+figName+'.h5', "w")
    h5f.create_dataset(outDir+opName+'_Tepar',  (np.size(TePar ),), dtype='f8', data=(1./eV)*TePar)
    h5f.create_dataset(outDir+opName+'_Teperp', (np.size(TePerp),), dtype='f8', data=(1./eV)*TePerp)
    h5f.create_dataset(outDir+opName+'_time', (np.size(time),), dtype='f8', data=nu_ii*(time+0.))
    h5f.close()

#.......................................................................#

if plot_Merrors:
  #[ Plot the errors in M0, M1 and M2 as a function of velocity space resolution.
  tests = ['glb'+str(i) for i in range(1,26)] 

  figProp1a = (12.0, 4.0)
  ax1aPos   = [[0.060, 0.16, 0.25, 0.77], [0.37, 0.16, 0.25, 0.77], [0.68, 0.16, 0.25, 0.77]]
  cax1aPos  = [[0.315, 0.16, 0.02, 0.77], [0.625, 0.16, 0.02, 0.77], [0.935, 0.16, 0.02, 0.77], ]
  fig1a     = plt.figure(figsize=figProp1a)
  ax1a      = [fig1a.add_axes(ax1aPos[0]), fig1a.add_axes(ax1aPos[1]), fig1a.add_axes(ax1aPos[2])]
  cbar_ax1a = [fig1a.add_axes(cax1aPos[0]), fig1a.add_axes(cax1aPos[1]), fig1a.add_axes(cax1aPos[2])]

  #[ Create the Nvpar and Nmu arrays.
  Nvpars = np.array([4, 6, 12, 18, 24])
  Nmus = np.array([4, 6, 12, 18, 24])
  #[ Turn to nodal.
  Nv = np.size(Nvpars)
  Nm = np.size(Nmus)
  Nvpars_ex = np.insert(Nvpars,[0,Nv],[Nvpars[0]-(Nvpars[1]-Nvpars[0]),Nvpars[Nv-1]+(Nvpars[Nv-1]-Nvpars[Nv-2])])
  Nmus_ex = np.insert(Nmus,[0,Nm],[Nmus[0]-(Nmus[1]-Nmus[0]),Nmus[Nm-1]+(Nmus[Nm-1]-Nmus[Nm-2])])
  Nvpars_nod = 0.5*(Nvpars_ex[:-1]+Nvpars_ex[1:])
  Nmus_nod = 0.5*(Nmus_ex[:-1]+Nmus_ex[1:])
  #[ Create 2D version of nodal for pcolormesh.
  Nvpar2D, Nmu2D = np.meshgrid(Nvpars_nod,Nmus_nod,indexing='ij')
  
  #[ Plot conservation errors as a function of Nx.
  hpl1a = list()
  m0Error, m1Error, m2Error = np.zeros((Nv,Nm)), np.zeros((Nv,Nm)), np.zeros((Nv,Nm))
  for dI in range(len(tests)):  #[ Loop over tests.
    fileRoot = dataDir+'/'+tests[dI]+'/'+simName+'-elc_'

    #[ Get the grid and indices in Nvpar-Nmu grid
    x, dim, nx, lx, dx, _ = pgu.getRawGrid(fileRoot+'0.gkyl', location='center')
    nv_idx, nm_idx = pgu.findNearestIndex(Nvpars, nx[1]), pgu.findNearestIndex(Nmus, nx[2])
  
    #[ Get integrated moments of electrons and ions.
    _, int_moms = pgu.readDynVector(fileRoot+'integrated_moms.gkyl') 
    intM0 = int_moms[:,0]
    intM1 = int_moms[:,1]
    intM2 = int_moms[:,2] + int_moms[:,3]
  
    m0Error[nv_idx,nm_idx] = np.abs(intM0[-1]-intM0[0])/intM0[0]/(np.size(intM0)-1)
    m1Error[nv_idx,nm_idx] = np.abs((intM1[-1]-intM1[0])/intM1[0])/(np.size(intM1)-1)
    m2Error[nv_idx,nm_idx] = np.abs(intM2[-1]-intM2[0])/intM2[0]/(np.size(intM2)-1)
  
  hpl1a.append(ax1a[0].pcolormesh(Nvpar2D, Nmu2D, m0Error))
  hpl1a.append(ax1a[1].pcolormesh(Nvpar2D, Nmu2D, m1Error))
  hpl1a.append(ax1a[2].pcolormesh(Nvpar2D, Nmu2D, m2Error))

  cbar = list()
  for s in range(3):
    cbar.append(plt.colorbar(hpl1a[s],ax=ax1a[s],cax=cbar_ax1a[s]))
    cbar[s].ax.tick_params(labelsize=tickFontSize)
    cmag = cbar[s].ax.yaxis.get_offset_text().set_size(tickFontSize)

  if saveData:
    h5f.create_dataset('errorM0_'+opName[oI]+'_dv'+str(vI+1)+'_p'+str(pI+1), (len(numCellsV),), dtype='f8', data=m0Error)
    h5f.create_dataset('errorM1_'+opName[oI]+'_dv'+str(vI+1)+'_p'+str(pI+1), (len(numCellsV),), dtype='f8', data=m1Error)
    h5f.create_dataset('errorM2_'+opName[oI]+'_dv'+str(vI+1)+'_p'+str(pI+1), (len(numCellsV),), dtype='f8', data=m2Error)

  if saveData:
    h5f.create_dataset('Nv', (len(numCellsV),), dtype='f8', data=numCellsV)
    h5f.close()

  ax1a[0].set_ylabel(r'$N_\mu$',fontsize=titleFontSize)
  ax1a[0].set_title(r'$E_{r,M_0}$',fontsize=xyLabelFontSize)
  ax1a[1].set_title(r'$E_{r,M_1}$',fontsize=xyLabelFontSize)
  ax1a[2].set_title(r'$E_{r,M_2}$',fontsize=xyLabelFontSize)
  fig1a.text(0.06, 0.9, '(a)', fontsize=textFontSize, color='white', weight='bold', transform=ax1a[0].transAxes)
  fig1a.text(0.06, 0.9, '(b)', fontsize=textFontSize, color='white', weight='bold', transform=ax1a[1].transAxes)
  fig1a.text(0.06, 0.9, '(c)', fontsize=textFontSize, color='white', weight='bold', transform=ax1a[2].transAxes)
  xlabels = [str(Nvpars[i]) for i in range(Nv)]
  ylabels = [str(Nmus[i]) for i in range(Nm)]
  for m in range(3):
    ax1a[m].set_xlabel(r'$N_{v_{\parallel}}$',fontsize=titleFontSize)
    ax1a[m].set_xticks(Nvpars, labels=xlabels)
    ax1a[m].set_yticks(Nmus, labels=ylabels)
    setTickFontSize(ax1a[m],tickFontSize)
  for m in range(2):
    plt.setp( ax1a[m+1].get_yticklabels(), visible=False)
  
  if outFigureFile:
    fig1a.savefig(outDir+'nonuni_lbo-intMerrorVsNv'+figureFileFormat)
    #fig1a.savefig(outDir+'fig4'+figureFileFormat)
  else:
    plt.show()

#.......................................................................#

if plot_distF:
  #[ Plot initial distribution function.

  test = 'glb7'
  fileName = test+'/'+simName+'-elc'

  figProp3a = (7., 5.0)
  ax3aPos   = [[0.07, 0.12, 0.75, 0.83]]
  cax3aPos  = [[0.83, 0.12, 0.02, 0.83]]
  fig3a     = plt.figure(figsize=figProp3a)
  ax3a      = [fig3a.add_axes(ax3aPos[0])]
  cbar_ax3a = [fig3a.add_axes(cax3aPos[0])]

  filePath = dataDir+fileName

  #[ Load the distribution times the Jacobian (J_v*f), and the Jacobian J_v, divide (J_vf)/J_v and interpolate.
  f_data = pg.GData(filePath + '_0.gkyl', mapc2p_vel_name=filePath + '_mapc2p_vel.gkyl')
  Jv_data = pg.GData(filePath + '_jacobvel.gkyl', mapc2p_vel_name=filePath + '_mapc2p_vel.gkyl')
  fElc_c = f_data.get_values()
  Jv_c = Jv_data.get_values()
  f_data._values = fElc_c/Jv_c
  dg = pg.GInterpModal(f_data, poly_order=polyOrder, basis_type="gkhyb")
  xInt, fElcInit = dg.interpolate()

  fElcInit_vpar_mu = np.squeeze(fElcInit)[0,:,:]
  
  #[ 2D nodal mesh.
  Xnodal = [np.outer(xInt[1], np.ones(np.shape(xInt[2]))), \
            np.outer(np.ones(np.shape(xInt[1])), xInt[2])]

  hpl3a = list()
  hpl3a.append(ax3a[0].pcolormesh(Xnodal[0], Xnodal[1], fElcInit_vpar_mu, cmap='inferno', edgecolors='grey',
            linewidth=0.1))

  cbar = [plt.colorbar(hpl3a[0],ax=ax3a[0],cax=cbar_ax3a[0])]
  cbar[0].ax.tick_params(labelsize=tickFontSize)
  cmag = cbar[0].ax.yaxis.get_offset_text().set_size(tickFontSize)
  cbar[0].set_label(r'$f_e(v_\parallel,\mu,t=0)$', rotation=90, labelpad=8, fontsize=16)
  ax3a[0].set_xlabel(r'$v_\parallel$ (m/s)',fontsize=xyLabelFontSize, labelpad=-2)
  ax3a[0].set_ylabel(r'$\mu$ (J/T)',fontsize=xyLabelFontSize, labelpad=-2)
  setTickFontSize(ax3a[0],tickFontSize)
  hmag = ax3a[0].xaxis.get_offset_text().set_size(tickFontSize)
  hmag = ax3a[0].yaxis.get_offset_text().set_size(tickFontSize)
  plt.text(0.9, 0.9, r'(a)', fontsize=textFontSize, color='white', weight='regular', transform=ax3a[0].transAxes)

  if outFigureFile:
    fig3a.savefig(outDir+'nonuni_lbo-fElcInit'+figureFileFormat)
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
