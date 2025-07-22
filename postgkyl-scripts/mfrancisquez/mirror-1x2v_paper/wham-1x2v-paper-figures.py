#[ ........................................................... ]#
#[ 
#[ pre and post processing for WHAM 1x2v simulations
#[
#[ Simulations of interest:
#[ - gk55: longest running (to t=32 mu sec) kin elc sim.
#[ - gk57: adiabatic elc.
#[ - gk71: adiabatic elc (softened).
#[ - gk77: adiabatic elc (softened), 1.5x Nvpar.
#[ - gk75: kin elc w/ ExB term, nonlinear pol  (softened), 1.5x Nvpar.
#[ 
#[ 
#[ Manaure Francisquez.
#[ February 2021.
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
sys.path.insert(0, '/home/manaurer/pets/gkeyll/postProcessingScripts/')
import pgkylUtil as pgu
from scipy import special
from scipy import optimize
import h5py

plot_modelEq              = False   #[ Plot the model equilibrium.
plot_fi_zmu               = False   #[ Plot f_i(z,mu) at some vpar.
plot_fi_zmu_chi           = False   #[ Plot f_i(z,mu) and the softening function chi(z,mu).
plot_pastukhovPhi         = False   #[ Plots with the estimate of the Pastukhov potential.
plot_phiAdiabatic         = False   #[ Plot e*phi(z,t)/Te0 and e*phi(z=0,t)/Te0 for the adiabatic elc sim.
plot_momAdiabatic         = False   #[ Plot density, temperature and Upar for the adiabatic elc sim.
plot_fi_vspace_Adiabatic  = False   #[ Plot ion velocity space at z=0, z=z_m and z=zMax for adiabatic elc sim.
plot_phi_forceSoft        = False   #[ Plot comparison of phi profile in force softening test (adiabatic).
plot_fi_vspace_forceSoft  = False   #[ Plot ion velocity space at z=0, z=z_m and z=zMax for adiabatic elc sim.
plot_phi_forceSoftChiComp = False   #[ Plot phi profile for various chi softening factors.
plot_kperpScan            = False   #[ Plot e*phi/Te for all kperp*rhos, and/or e*Delta phi/Te0 vs. kperp*rhos.
plot_f_vspace_kinetic     = False   #[ Plot velocity space at z=0, z=z_m and z=zMax of kinetic elc sim.
plot_phiKinetic           = False   #[ Plot phi(z,t), and phi(z) at the last frame of kinetic elc sim.
plot_momKinElc            = False   #[ Plot density, temperature and Upar for the kinetic elc sim.
plot_f_vspace_nonlinPol   = False   #[ Plot velocity space at z=0, z=z_m and z=zMax of kinetic elc sim w/ nonlinear pol.
plot_momKinElc_nonlinPol  = False   #[ Plot density, temperature and Upar for the kinetic elc sim.
plot_phiNonlinPol         = False   #[ Plot phi(z,t), and phi(z) at the last frame of kinetic elc with nonlinear pol sim.
plot_mom_comp             = False   #[ Plot density, temperature and Upar for the kinetic and adiabatic elc sim.
plot_phi_t_comp           = False   #[ phi(z,t) and e*Delta phi/Te vs. t for all three models.
plot_resScan              = False   #[ Plot M0, Temp, phi(z), e*phi/Te for all various resolutions.
print_noNegCells          = False   #[ Print the number of negative cells in a file.
plot_mom_ic               = True   #[ Plot the initial moments.
plot_mom_ic_final         = False   #[ Plot the initial and final (for the kinetic electron nonlin pol sim) moments.

plot_mom_forceSoftComp = False   #[ Plot density, temperature and Upar for the adiabatic elc sims w/ & w/o force softening.
plot_momAdiabaticOld      = False   #[ Plot density, temperature and Upar for the adiabatic elc sim.
plot_M0           = False   #[ Plot density.
plot_Temp         = False   #[ Plot temperature.
plot_intM0        = False   #[ Plot integrated density.
plot_M0TempUpar   = False   #[ Plot density, temperature and Upar.
plot_PparPperp    = [False, False]   #[ Plot Ppar and Pperp, and/or their ratio.
plot_adiabaticElc = False   #[ Plot M0, Temp, phi(z), e*phi/Te for kinetic and adiabatic electrons.
plot_nonuniformz  = False

#outDir = '/Users/manaure/Documents/gkeyll/publications/wham1x2v/'
outDir = '/home/manaurer/gkeyll/code/gkyl-sims/mirror/'

outFigureFile    = False     #[ Output a figure file?.
figureFileFormat = '.eps'    #[ Can be .png, .pdf, .ps, .eps, .svg.
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
print("vti0 = ",vti)
print("mui0 = ",0.5*mi*(vti**2)/B_p)

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
  for tick in axIn.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontSizeIn)
  for tick in axIn.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontSizeIn)

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

#[ Compute the magnetic field (radial, axial and total amplitudes),
#[ given 1D arrays of R and Z coordinates, Rin and Zin.
def Bfield_RZ(Rin, Zin, modelParsIn):
  RmIn = np.outer(Rin, np.ones(np.size(Zin)))
  ZmIn = np.outer(np.ones(np.size(Rin)), Zin)

  if modelParsIn["model"] == 0:
    a_2   = modelParsIn["a_2"]
    a_3   = modelParsIn["a_3"]
    d_1   = modelParsIn["d_1"]
    d_2   = modelParsIn["d_2"]
    A     = modelParsIn["A"]
    D     = modelParsIn["D"]
    mcB   = modelParsIn["mcB"]
    alpha = modelParsIn["alpha"]
  
    BRad = -(1./2.)*RmIn*mcB*( (2.*a_2*ZmIn+np.sign(ZmIn)*3.*a_3*np.power(ZmIn,2))*A*np.sin(a_2*np.power(ZmIn,2)+a_3*np.power(np.abs(ZmIn),3)) \
                              +(np.sign(ZmIn)*d_1+2.*d_2*ZmIn)*D*np.sin(d_1*np.abs(ZmIn)+d_2*np.power(ZmIn,2)) )
    BZ   = mcB*(alpha-A*np.cos(a_2*np.power(ZmIn,2)+a_3*np.power(np.abs(ZmIn),3)) \
                     -D*np.cos(d_1*np.abs(ZmIn)+d_2*np.power(ZmIn,2)))

  elif modelParsIn["model"] == 1:
    mcB   = modelParsIn["mcB"]
    gamma = modelParsIn["gamma"]
    Z_m   = modelParsIn["Z_m"]
  
    BRad = -0.5*RmIn*mcB*(-2.*(ZmIn-Z_m)/( np.pi*(gamma**3)*((1.+((ZmIn-Z_m)/gamma)**2)**2)) \
                          -2.*(ZmIn+Z_m)/( np.pi*(gamma**3)*((1.+((ZmIn+Z_m)/gamma)**2)**2)) )
  
    BZ   = mcB*( 1./(np.pi*gamma*(1.+((ZmIn-Z_m)/gamma)**2)) \
                +1./(np.pi*gamma*(1.+((ZmIn+Z_m)/gamma)**2)) )

  Bmag = np.sqrt(np.power(BRad,2) + np.power(BZ,2))

  return BRad, BZ, Bmag
#[ Finished Bfield_RZ ....................... ]#

#[ Alternative tperp calculation
def calcTperp(fileRoot, species, frame, **kwargs):
  distfFile = fileRoot+'_%s_%d.bp'
  diagFile  = fileRoot+'_%s_gridDiagnostics_%d.bp'

  if 'gridPars' in kwargs:
    xIntC  = kwargs['gridPars']['xIntC']
    nxIntC = kwargs['gridPars']['nxIntC']
  else:
    xIntC, _, nxIntC, _, _, _ = pgu.getGrid(distfFile % (species,frame),polyOrder,basisType,location='center')

  muPhase = np.tile(np.tile(xIntC[2], (nxIntC[1],1)), (nxIntC[0],1,1))

  bmag = 0.
  if 'bmag' in kwargs:
    bmag = kwargs['bmag']
  else:
    geoFile = fileRoot+'_allGeo_0.bp'
    bmag    = np.squeeze(pgu.getInterpData(geoFile,polyOrder,basisType,location='center',varName='bmag'))

  jacobGeoInv = 0.
  if 'jacobGeoInv' in kwargs:
    jacobGeoInv = kwargs['jacobGeoInv']
  else:
    geoFile     = fileRoot+'_allGeo_0.bp'
    jacobGeoInv = np.squeeze(pgu.getInterpData(geoFile,polyOrder,basisType,location='center',varName='jacobGeoInv'))

  distf       = np.squeeze(pgu.getInterpData(distfFile % (species,frame), polyOrder, basisType))
  den         = np.squeeze(pgu.getInterpData(diagFile % (species,frame), polyOrder, basisType, varName='M0'))
  if species=='ion':
    mass = mi
  elif species=='elc':
    mass = me

  if 'phaseFac' in kwargs:
    return (2.*np.pi/(mass*den))*bmag*jacobGeoInv \
           *np.trapz(np.trapz(kwargs['phaseFac']*muPhase*distf, xIntC[2], axis=2), xIntC[1], axis=1)
  else:
    return (2.*np.pi/(mass*den))*bmag*jacobGeoInv \
           *np.trapz(np.trapz(muPhase*distf, xIntC[2], axis=2), xIntC[1], axis=1)
#[ Finished calcTperp ....................... ]#

#[ Compute the force-softening factor Pi(z,mu) = (1/B)*int_0^z chi*dB/dz:
def forceSofteningPi_zvparmu(fileRoot, species, frame, dBdzFile):
  chiFile  = fileRoot+'_%s_mirrorPenalty_0.bp'    #.Complete file name.
  geoFile  = fileRoot+'_allGeo_0.bp'
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fileRoot+'_%s_%d.bp' % (species,frame),polyOrder,basisType,location='center')
  fsChi = np.squeeze(pgu.getInterpData(chiFile % (species), polyOrder, basisType))
  dBdz  = np.squeeze(pgu.getInterpData(dBdzFile, polyOrder, basisType))
  bmag  = np.squeeze(pgu.getInterpData(geoFile,polyOrder,basisType,location='center',varName='bmag'))
  if not np.array_equal(np.shape(fsChi),nxIntC):
    #[ It must be that ghost cells were included. Remove them.
    fsChi = fsChi[2:-2,2:-2,2:-2]

  integrand = np.squeeze(fsChi[:,0,:])*np.repeat(dBdz[:,np.newaxis], nxIntC[2], axis=1)
  fsPi_zmu  = np.zeros((nxIntC[0],nxIntC[2]))
  #[ We will only integrate in z>0 to save time, and then reflect.
  for j in range(nxIntC[2]):
     for i in range(nxIntC[0]//2+1,nxIntC[0]):
       fsPi_zmu[i,j] = (1./bmag[i])*( B_p+np.trapz(integrand[nxIntC[0]//2:i,j], xIntC[0][nxIntC[0]//2:i]) )
  #[ Reflect Pi.
  fsPi_zmu[:nxIntC[0]//2-1,:] = fsPi_zmu[-1:nxIntC[0]//2:-1,:]
  #[ Impose BC:
  fsPi_zmu[nxIntC[0]//2-1:nxIntC[0]//2+1,:] = 1.0
  #[ Repeat in the vpar direction.
  return np.repeat(fsPi_zmu[:,np.newaxis,:], nxIntC[1], axis=1)
#[ Finished forceSofteningPi_zvparmu ....................... ]#
        
#[ ................................................................................ ]#

if plot_modelEq:
  #[ Plot the model equilibrium.

  Rmin, Rmax = 0.0, 0.36    #[ Minium/maximum radius (m).
  Zmin, Zmax = -2.501, 2.5   #[ Minimum/maximum axial coordinate measured from device center (m).

  numR, numZ = 100, 100      #[ Number of points along R and Z.

  #[ Set of Z and R coordinates for field-line starting points
  fieldLineStarts = [np.zeros(16), np.linspace(Rmin,0.1,16)]

  modelPars = {}
  ##[ Parameters for analytic model based on cosines.
  #modelPars["model"] = 0
  #modelPars["mcB"]   = 3.1279               #.Reference magnetic field (T).
  #modelPars["A"]     = 2.5
  #modelPars["D"]     = -1.7
  #modelPars["alpha"] = 1.45
  #modelPars["a_2"]   = 1.24*np.pi
  #modelPars["a_3"]   = -1.19
  #modelPars["d_1"]   = 0.246
  #modelPars["d_2"]   = 0.06
  #[ Parameters for analytic model based on Cauchy distributions.
  modelPars["model"] = 1
  modelPars["mcB"]   =6.51292   #8.3    #6.51292  #6.8051
  modelPars["gamma"] =0.124904  #0.1575 #0.124904 #0.141368
  modelPars["Z_m"]   =0.98      #0.98   #0.98

  #[ The following is to use John Wright's routines to read/write EQDSK.
  sys.path.append('/Users/manaure/Documents/gkeyll/code/gkyl-sims/mirror/equilibrium/plasma')
  import equilibrium_process as eqdsk
  #[ The following is to call Pleiades.
  sys.path.insert(0, '/Users/manaure/Documents/gkeyll/code/gkyl-sims/mirror/equilibrium/pleiades')
  from pleiades import RectMesh, compute_equilibrium, write_eqdsk
  from pleiades.configurations import WHAM

  R, Z = np.linspace(Rmin,Rmax,numR+1), np.linspace(Zmin,Zmax,numZ+1)

  BRad, BZ, Bmag = Bfield_RZ(R, Z, modelPars)

  print(" max Bmag = ",np.amax(Bmag))
  print(" min Bmag = ",np.amin(Bmag))
  print(" B_p      = ",Bmag[0,0])

  Rgrid, Zgrid = np.mgrid[Rmin:Rmax:complex(0,numR+1), Zmin:Zmax:complex(numZ+1)]

  figProp1 = (10., 6.)
  ax1Pos   = [ [0.08, 0.53, 0.815, 0.42], \
               [0.08, 0.1, 0.815, 0.42] ]
  cax1Pos  = [ [0.905, 0.53, 0.02, 0.42], \
               [0.905, 0.1, 0.02, 0.42] ]
  fig1     = plt.figure(figsize=figProp1)
  ax1      = [ fig1.add_axes(ax1Pos[0]), \
               fig1.add_axes(ax1Pos[1]) ]
  cbar_ax1 = [ fig1.add_axes(cax1Pos[0]), \
                fig1.add_axes(cax1Pos[1]) ]

  #[ Field lines.
  hpl1a = list()
  #[ Color proportional to B.
  seedPoints = np.array([fieldLineStarts[0].tolist(),fieldLineStarts[1].tolist()])
  hpl1a.append(ax1[0].streamplot(Zgrid, Rgrid, BZ, BRad, color=Bmag, start_points=seedPoints.T, density=20))
  ax1[0].set_title(r'Field lines (color $\propto B$)', fontsize=titleFontSize)

  hcb1a = plt.colorbar(hpl1a[0].lines,ax=ax1[0],cax=cbar_ax1[0])
  hcb1a.set_label('$B$ (T)', rotation=90, labelpad=+8, fontsize=colorBarLabelFontSize)
  hcb1a.ax.tick_params(labelsize=tickFontSize)
  plt.setp( ax1[0].get_xticklabels(), visible=False)
  ax1[0].set_ylabel(r'$R$ (m)', fontsize=xyLabelFontSize, labelpad=+0)
  plt.text(0.75, 0.85, r'Gkeyll', fontsize=16, color='black', transform=ax1[0].transAxes)
  plt.text(0.05, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax1[0].transAxes)
  ax1[0].set_xlim(Zmin,Zmax)
  ax1[0].set_ylim(-0.01,0.38)
  setTickFontSize(ax1[0],tickFontSize)
  #.Plot vertical lines where Gkeyll domain terminates.
  #hpl1a.append(ax1[0].plot([-2.3,-2.3],[-1.,1.], linestyle='--', color=grey))
  #hpl1a.append(ax1[0].plot([ 2.3, 2.3],[-1.,1.], linestyle='--', color=grey))

  #[ Plot the magnetic field obtained by Pleiades.
  #[ Make sure the configuration file pleiades/pleiades/configurations/wham.py
  #[ has the desired parameters. At first we intended to use J. Pizzo's phase-1 parameters (2020-01-29).
  #[   z0 = 0.98
  #[   dr, dz = 0.01377, 0.01229
  #[   nr, nz = 23, 6
  #[   r0 = 0.06+dr*nr/2
  #[   self.hts1 = RectangularCoil(r0, z0, nr=nr, nz=nz, dr=dr, dz=dz)
  #[   self.hts2 = RectangularCoil(r0, -z0, nr=nr, nz=nz, dr=dr, dz=dz)
  #[   self.hts1.current = 35140
  #[   self.hts2.current = 35140
  #[   # Set central coil parameters (w7a coils)
  #[   z0 = 0.20
  #[   dr, dz = 0.4/12, 0.183/2
  #[   nr, nz = 12, 2
  #[   r0 = 0.55 + dr*nr/2
  #[   self.w7a1 = RectangularCoil(r0, z0, nr=nr, nz=nz, dr=dr, dz=dz)
  #[   self.w7a2 = RectangularCoil(r0, -z0, nr=nr, nz=nz, dr=dr, dz=dz)
  #[   #For 0.425 T central Field (Target field for WHAM Phase-1)
  #[   self.w7a1.current = 2850
  #[   self.w7a2.current = 2850
  wham = WHAM()  #[ Create the WHAM device.
  
  xxrange, zzrange = [0, 0.75], [-2.5, 2.5]  #[ Range of Pleaides solve.
  
  mesh = RectMesh(rmin=xxrange[0], rmax=xxrange[1], zmin=zzrange[0], zmax=zzrange[1], nr=76, nz=201)
  R, Z = mesh.R, mesh.Z
  
  #[ Set the brb grid (does all greens functions calculations right here).
  wham.mesh = mesh
  
  #[ Get desired field quantities from brb object and view coilset.
  #[ These are vacuum fields before the solution.
  B = np.sqrt(wham.BR()**2 + wham.BZ()**2).reshape(R.shape)
  BR = wham.BR().reshape(R.shape)
  BZ = wham.BZ().reshape(R.shape)
  print(" max Bmag = ",np.amax(B))
  print(" min Bmag = ",np.amin(B))

  hpl1b = list()
  hpl1b.append(ax1[1].streamplot(np.transpose(Z), np.transpose(R), np.transpose(BZ), np.transpose(BR), \
                                 color=np.transpose(B), start_points=seedPoints.T, density=20, \
                                 norm=colors.Normalize(np.amin(B),np.amax(B))))
  hcb1b = plt.colorbar(hpl1b[0].lines,ax=ax1[1],cax=cbar_ax1[1])
  hcb1b.set_label('$B$ (T)', rotation=90, labelpad=+8, fontsize=colorBarLabelFontSize)
  ax1[1].set_xlim(np.amin(Z),np.amax(Z))
  ax1[1].set_ylim(-0.01,0.38)
  ax1[1].set_ylabel(r'$R$ (m)', fontsize=xyLabelFontSize, labelpad=+0)
  ax1[1].set_xlabel(r'$Z$ (m)', fontsize=xyLabelFontSize, labelpad=+0)
  plt.text(0.75, 0.85, r'Pleiades', fontsize=16, color='black', transform=ax1[1].transAxes)
  plt.text(0.05, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax1[1].transAxes)
  setTickFontSize(ax1[1],tickFontSize)

  if outFigureFile:
    plt.savefig(outDir+"gkylWHAM-1comp"+figureFileFormat)
    plt.close()
  else:
    plt.show()

  #[ Create a plot of the axial magnetic field as a function of Z.
  figProp2  = (8., 3.5)
  ax2Pos    = [0.13, 0.15, 0.815, 0.82]
  fig2      = plt.figure(figsize=figProp2)
  ax2       = fig2.add_axes(ax2Pos)
  hpl2 = list()
  hpl2.append(ax2.plot(Z[:,0], B[:,0], color=defaultBlue, linestyle='-'))
  hpl2.append(ax2.plot(Zgrid[0,:], Bmag[0,:], color=defaultOrange, linestyle='--'))
  #.Plot vertical lines where Gkeyll domain terminates.
  #hpl2.append(ax2.plot([-2.3,-2.3],[-1.,20.], linestyle='--', color=grey))
  #hpl2.append(ax2.plot([ 2.3, 2.3],[-1.,20.], linestyle='--', color=grey))

  ax2.set_xlabel(r'$Z$ (m)', fontsize=xyLabelFontSize, labelpad=+0)
  ax2.set_ylabel(r'$B(R=0)$ (T)', fontsize=xyLabelFontSize, labelpad=+2)
  ax2.legend([r'Pleiades',r'Gkeyll'], fontsize=legendFontSize, frameon=False)
  plt.text(0.05, 0.85, r'(c)', fontsize=textFontSize, color='black', transform=ax2.transAxes)
  ax2.set_xlim(Zmin,Zmax)
  ax2.set_ylim(-0.01,18.)
  setTickFontSize(ax2,tickFontSize)

  if outFigureFile:
    plt.savefig(outDir+"gkylWHAM-1comp_BatReq0p0"+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_fi_zmu:
  #[ Plot f_i(z,mu) at some vpar.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk57/'

  fileName = 'gk57-wham1x2v'    #.Root name of files to process.
  frame    = 128

  fName = dataDir+fileName+'_%s_%d.bp'    #.Complete file name.

  #[ Load the grid.
  xInt_i, _, nxInt_i, lxInt_i, dxInt_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType)
  xIntC_i, _, nxIntC_i, lxIntC_i, dxIntC_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType,location='center')

  #[ Get indices along vpar of slices we wish to plot:
  plotVpar = [1.e-4]
  plotVparIdx = [np.argmin(np.abs(xIntC_i[1]-val)) for val in plotVpar]
  
  #[ Load the distribution functions.
  fIon = np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType))

  #[ Prepare figure.
  figProp3 = (6.5,4.)
  ax3Pos   = [[0.1, 0.15, 0.72, 0.83],]
  cax3Pos  = [[0.825, 0.15, 0.02, 0.83]]
  fig3     = plt.figure(figsize=figProp3)
  ax3      = [fig3.add_axes(pos) for pos in ax3Pos]
  cbar_ax3 = [fig3.add_axes(pos) for pos in cax3Pos]
  
  #[ Create colorplot grid. Recall coordinates have to be nodal.
  Xnodal_i = [np.outer(xInt_i[0]/vti, np.ones(nxInt_i[2])),
              np.outer(np.ones(nxInt_i[0]), xInt_i[2]/(0.5*mi*(vti**2)/B_p))]

  hpl3a = list()
  ax3in = [0 for i in range(len(ax3))]
  hpl3ain = [0 for i in hpl3a]

  fIon_zmu = np.abs(fIon[:,plotVparIdx[0],:])
  hpl3a.append(ax3[0].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon_zmu, 
                                 norm=colors.LogNorm(vmin=fIon_zmu.min(), vmax=fIon_zmu.max()), cmap='inferno'))

  print(np.amin(fIon[:,plotVparIdx[0],:]),np.amax(fIon[:,plotVparIdx[0],:]))
  print(np.amin(fIon[:,plotVparIdx[-1],:]),np.amax(fIon[:,plotVparIdx[-1],:]))

  hcb3a = plt.colorbar(hpl3a[0], ax=ax3[0], cax=cbar_ax3[0])
  hcb3a.set_label('$f_i(z,v_\parallel=0,\mu,t=32\,\mu\mathrm{s})$', rotation=270, labelpad=18, fontsize=colorBarLabelFontSize)
  hcb3a.ax.tick_params(labelsize=tickFontSize)
  ax3[0].set_ylabel(r'$\mu/[m_iv_{ti0}^2/(2B_p)]$', fontsize=xyLabelFontSize)
  ax3[0].set_xlabel(r'$v_\parallel/v_{ti0}$', fontsize=xyLabelFontSize)
  setTickFontSize(ax3[0],tickFontSize) 
  hmagx = ax3[0].xaxis.get_offset_text().set_size(tickFontSize)
#  plt.text(0.05, 0.85, r'$f_i(z=0)$', fontsize=textFontSize, color='white', transform=ax3[0].transAxes)
  
  if outFigureFile:
    fig3.savefig(outDir+fileName+'_fi_zmu_log_'+str(frame)+figureFileFormat)
    plt.close()
  else:
    plt.show()
  
#................................................................................#

#if plot_chi:
#  #[ Plot chi(z,mu) at some vpar.
#
#  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk71/'
#
#  fileName = 'gk71-wham1x2v'    #.Root name of files to process.
#
#  fName = dataDir+fileName+'_ion_mirrorPenalty_0.bp'    #.Complete file name.
#
#  #[ Load the grid.
#  xInt_i, _, nxInt_i, lxInt_i, dxInt_i, _ = pgu.getGrid(fName,polyOrder,basisType)
#  xIntC_i, _, nxIntC_i, lxIntC_i, dxIntC_i, _ = pgu.getGrid(fName,polyOrder,basisType,location='center')
#
#  #[ Get indices along vpar of slices we wish to plot:
#  plotVpar = [1.e-4]
#  plotVparIdx = [np.argmin(np.abs(xIntC_i[1]-val)) for val in plotVpar]
#  
#  #[ Load the softening functions.
#  chi = np.squeeze(pgu.getInterpData(fName, polyOrder, basisType))
#
#  #[ Prepare figure.
#  figProp4 = (6.5,4.)
#  ax4Pos   = [[0.1, 0.15, 0.72, 0.83],]
#  cax4Pos  = [[0.825, 0.15, 0.02, 0.83]]
#  fig4     = plt.figure(figsize=figProp4)
#  ax4      = [fig4.add_axes(pos) for pos in ax4Pos]
#  cbar_ax4 = [fig4.add_axes(pos) for pos in cax4Pos]
#  
#  #[ Create colorplot grid. Recall coordinates have to be nodal.
#  Xnodal_i = [np.outer(xInt_i[0]/vti, np.ones(nxInt_i[2])),
#              np.outer(np.ones(nxInt_i[0]), xInt_i[2]/(0.5*mi*(vti**2)/B_p))]
#
#  hpl4a = list()
#  ax4in = [0 for i in range(len(ax4))]
#  hpl4ain = [0 for i in hpl4a]
#
#  chi_zmu = np.abs(chi[:,plotVparIdx[0],:])
#  hpl4a.append(ax4[0].pcolormesh(Xnodal_i[0], Xnodal_i[1], chi_zmu, cmap='inferno'))
#
#  hcb4a = plt.colorbar(hpl4a[0], ax=ax4[0], cax=cbar_ax4[0])
#  hcb4a.set_label('$\chi(z,\mu)$', rotation=270, labelpad=18, fontsize=colorBarLabelFontSize)
#  hcb4a.ax.tick_params(labelsize=tickFontSize)
#  ax4[0].set_ylabel(r'$\mu/[m_iv_{ti0}^2/(2B_p)]$', fontsize=xyLabelFontSize)
#  ax4[0].set_xlabel(r'$v_\parallel/v_{ti0}$', fontsize=xyLabelFontSize)
#  setTickFontSize(ax4[0],tickFontSize) 
#  hmagx = ax4[0].xaxis.get_offset_text().set_size(tickFontSize)
##    nohmagy = ax4[0+i].yaxis.get_offset_text().set_size(0)
##  hmagy = ax4[0].yaxis.get_offset_text().set_size(tickFontSize)
##  plt.text(0.05, 0.85, r'$f_i(z=0)$', fontsize=textFontSize, color='white', transform=ax4[0].transAxes)
##  plt.text(0.05, 0.85, r'$f_i(z=z_m)$', fontsize=textFontSize, color='white', transform=ax4[1].transAxes)
##  plt.text(0.05, 0.85, r'$f_i(z=L_z/2)$', fontsize=textFontSize, color='white', transform=ax4[2].transAxes)
#  
##  fileName = 'gk53-wham1x2v-nonuniformmu'    #.Root name of files to process.
#  if outFigureFile:
#    fig4.savefig(outDir+fileName+'_ion_mirrorPenalty'+figureFileFormat)
#    plt.close()
#  else:
#    plt.show()
  
#................................................................................#

if plot_fi_zmu_chi:
  #[ Plot f_i(z,mu) and chi(z,mu) at some vpar.

  #[ Load the ion distribution function and its grid.
  dataDir  = '/scratch/gpfs/manaurer/gkeyll/mirror/gk57/'
  fileName = 'gk57-wham1x2v'    #.Root name of files to process.
  frame    = 128

  fName = dataDir+fileName+'_%s_%d.bp'    #.Complete file name.
  fIon = np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType))
  xInt_i, _, nxInt_i, lxInt_i, dxInt_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType)
  xIntC_i, _, nxIntC_i, lxIntC_i, dxIntC_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType,location='center')

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk71/'
  fileName = 'gk71-wham1x2v'    #.Root name of files to process.

  #[ Load the softening function and its grid.
  fName = dataDir+fileName+'_ion_mirrorPenalty_0.bp'    #.Complete file name.
  chi = np.squeeze(pgu.getInterpData(fName, polyOrder, basisType))
  xInt_i, _, nxInt_i, lxInt_i, dxInt_i, _ = pgu.getGrid(fName,polyOrder,basisType)
  xIntC_i, _, nxIntC_i, lxIntC_i, dxIntC_i, _ = pgu.getGrid(fName,polyOrder,basisType,location='center')

  #[ Get indices along vpar of slices we wish to plot:
  plotVpar = [1.e-4]
  plotVparIdx = [np.argmin(np.abs(xIntC_i[1]-val)) for val in plotVpar]

  fIon_zmu = np.abs(fIon[:,plotVparIdx[0],:])
  chi_zmu = chi[:,plotVparIdx[0],:]

  #[ Load the magnetic field amplitude and compute its z derivative.
  fName = dataDir+fileName+'_allGeo_0.bp'    #.Complete file name.
  bmag = np.squeeze(pgu.getInterpData(fName, polyOrder, basisType, varName='bmag'))
  bmag_z = np.gradient(bmag, dxIntC_i[0])
  
  #[ Prepare figure.
  figProp4 = (6.8,7.6)
  ax4Pos   = [[0.1, 0.49, 0.72, 0.40],
              [0.1, 0.07, 0.72, 0.40],]
  cax4Pos  = [[0.1, 0.93, 0.72, 0.02],
              [0.825, 0.07, 0.02, 0.40],]
  fig4     = plt.figure(figsize=figProp4)
  ax4      = [fig4.add_axes(pos) for pos in ax4Pos]
  cbar_ax4 = [fig4.add_axes(pos) for pos in cax4Pos]
  ax40b    = ax4[0].twinx()
  
  #[ Create colorplot grid. Recall coordinates have to be nodal.
  Xnodal_i = [np.outer(xInt_i[0], np.ones(nxInt_i[2])),
              np.outer(np.ones(nxInt_i[0]), xInt_i[2]/(0.5*mi*(vti**2)/B_p))]

  hpl4a = list()
  ax4in = [0 for i in range(len(ax4))]
  hpl4ain = [0 for i in hpl4a]

  hpl4a.append(ax4[0].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon_zmu, 
#                                 norm=colors.LogNorm(vmin=fIon_zmu.min(), vmax=fIon_zmu.max()), cmap='inferno'))
                                 norm=colors.LogNorm(vmin=1.e-15*fIon_zmu.max(), vmax=fIon_zmu.max()), cmap='inferno'))
  hpl4a.append(ax4[1].pcolormesh(Xnodal_i[0], Xnodal_i[1], chi_zmu, cmap='inferno'))
  hpl4a.append(ax40b.plot(xIntC_i[0], bmag_z, color=defaultGreen))

  hcb4a = list()
  hcb4a.append(plt.colorbar(hpl4a[0], ax=ax4[0], cax=cbar_ax4[0], orientation='horizontal', extend='min'))
#  hcb4a[0].ax.yaxis.set_ticks_position('left')
  hcb4a[0].ax.xaxis.set_ticks_position('top')
#  cbar_ax4[0].yaxis.tick_right()
  hcb4a.append(plt.colorbar(hpl4a[1], ax=ax4[1], cax=cbar_ax4[1]))
  hcb4a[0].set_label('$f_i(z,v_\parallel=0,\mu,t=32\,\mu\mathrm{s})$', rotation=0, labelpad=3, fontsize=colorBarLabelFontSize)
  hcb4a[1].set_label('$\chi(z,\mu)$', rotation=90, labelpad=8, fontsize=colorBarLabelFontSize)
  plt.setp( ax4[0].get_xticklabels(), visible=False)
  ax4[1].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize, labelpad=-2)
  for i in range(2):
    ax4[i].set_ylabel(r'$\mu/[m_iv_{ti0}^2/(2B_p)]$', fontsize=xyLabelFontSize)
    setTickFontSize(ax4[i],tickFontSize) 
    hcb4a[i].ax.tick_params(labelsize=tickFontSize)
  ax40b.set_ylabel(r'$dB/dz$ (T/m)', fontsize=xyLabelFontSize, color=defaultGreen)
  ax40b.set_ylim(-100., 100.) #1.15*np.amin(bmag_z),1.15*np.amax(bmag_z))
  ax40b.tick_params(axis='y', labelcolor=defaultGreen, labelsize=tickFontSize)
  nohmagx = ax4[0].yaxis.get_offset_text().set_size(0)
  hmagx = ax4[1].xaxis.get_offset_text().set_size(tickFontSize)
#  hmagy = ax4[0].yaxis.get_offset_text().set_size(tickFontSize)
  plt.text(0.05, 0.85, r'(a)', fontsize=textFontSize, color='white', transform=ax4[0].transAxes)
  plt.text(0.05, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax4[1].transAxes)
  
#  fileName = 'gk53-wham1x2v-nonuniformmu'    #.Root name of files to process.
  figName = 'gk-wham1x2v_ion_fi_mirrorPenalty_zmu_128'
  if outFigureFile:
    fig4.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")
    h5f.create_dataset('dBdz_z',np.shape(xIntC_i[0]),  dtype='f8', data=xIntC_i[0])
    h5f.create_dataset('dBdz',  np.shape(bmag_z),      dtype='f8', data=bmag_z)
    h5f.create_dataset('z',     np.shape(Xnodal_i[0]), dtype='f8', data=Xnodal_i[0])
    h5f.create_dataset('mu',    np.shape(Xnodal_i[1]), dtype='f8', data=Xnodal_i[1])
    h5f.create_dataset('f_i',   np.shape(fIon_zmu),    dtype='f8', data=fIon_zmu)
    h5f.create_dataset('chi',   np.shape(chi_zmu),     dtype='f8', data=chi_zmu)
    h5f.close()
  
#................................................................................#

if plot_pastukhovPhi:
  #[ Plot estimate of the Pastukhov potential.

  nuFrac_e = np.array([0.05, 0.1, 0.5, 1., 2., 3., 5., 10., 20.]) #np.linspace(0.05, 20, 16)

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk57/'
  fileName = 'gk57-wham1x2v'
  figName = fileName+'_pastukhov_ePhiDTe'

  #[ Electron-electron collision freq.
  logLambdaElc = 6.6 - 0.5*np.log(n0/1e20) + 1.5*np.log(Te0/eV)
  nuElc        = logLambdaElc*(eV**4)*n0/(6*np.sqrt(2)*(np.pi**(3/2))*(eps0**2)*np.sqrt(me)*(Te0**(3/2)))
  #[ Ion-ion collision freq.
  logLambdaIon = 6.6 - 0.5*np.log(n0/1e20) + 1.5*np.log(Ti0/eV)
  nuIon        = logLambdaIon*(eV**4)*n0/(12*(np.pi**(3/2))*(eps0**2)*np.sqrt(mi)*(Ti0**(3/2)))

  def G(R):
    return ((2.*R+1)/(2.*R))*np.log(4.*R+2)
  
  def I_x(x):
    return 1.+0.5*np.sqrt(np.pi*x)*np.exp(1./x)*special.erfc(np.sqrt(1./x))
  
  def tau_pe(x,R,frac):
    #[ Electron confinement time as a function of x=e*phi/Te and mirror ratio R.
    #[ This has a 1/4 (Cohen) instead of a 1/2 (Pastukhov). However, Cohen assumes
    #[ ee and ei collisions. If we want to consider only ee collisions use this
    #[ function with frac=0.5 (essentially turning the 1/4 into 1/2).
    return (np.sqrt(np.pi)/4.)*(1./(frac*nuElc))*G(R)*x*np.exp(x)/I_x(1./x)
  
  def tau_pi(R):
    #[ Ion confinement time as a function of mirror ratio R.
    tau_i = (1./np.sqrt(2))*np.sqrt(mi/me)*((Ti0/Te0)**(3./2.))/nuElc
    return tau_i*np.log10(R)

  def rootEq(x,R,frac_e):
    return tau_pi(R)-tau_pe(x,R,frac_e)

  #[ Load the magnetic field.
  dataFile = dataDir + fileName + '_allGeo_0.bp'
  bmag = pgu.getInterpData(dataFile,polyOrder,basisType,varName='bmag')
  
  #[ Load interpolated cell centered grid.
  xIcc, dimIcc, nxIcc, lxIcc, dxIcc, gridIcc = pgu.getGrid(dataFile,polyOrder,basisType,location='center',varName='bmag')
  
  #[ Only compute the Pastukhov potential between the center of the mirror and the mirror throat.
  iLo, iUp = nxIcc[0]//2, nxIcc[0]//2+np.argmax(bmag[nxIcc[0]//2:])

  #[ Prepare figure.
  figProp5 = (6.,5.5)
  ax5Pos   = [[0.13, 0.59, 0.85, 0.40],
              [0.13, 0.09, 0.85, 0.40],]
  fig5     = plt.figure(figsize=figProp5)
  ax5      = [fig5.add_axes(pos) for pos in ax5Pos]
  hpl5a    = list()
  hpl5b    = list()

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  #[ Show the phi profile for nuFrac = 0.5, 1.
  ePhiDTe = np.zeros(iUp-iLo)
  for d, nuFracElc in np.ndenumerate(np.array([0.5, 1.])):
    c = 0
    for i in range(iLo,iUp):
      B = bmag[i]
      R_m = np.amax(bmag)/B
      ePhiDTe[c] = optimize.ridder(rootEq, 1.e-10, 10., args=(R_m, nuFracElc))
      c = c +1

    xIcc0_sym   = np.concatenate([-xIcc[0][iUp-1:iLo-1:-1],xIcc[0][iLo:iUp]])
    ePhiDTe_sym = np.concatenate([ePhiDTe[::-1],ePhiDTe])

#    hpl5a.append(ax5[0].plot(xIcc[0][iLo:iUp], ePhiDTe))
#    ax5[0].set_xlim( xIcc[0][iLo], xIcc[0][iUp-1] )
    hpl5a.append(ax5[0].plot(xIcc0_sym, ePhiDTe_sym, color=defaultColors[1-d[0]], linestyle=lineStyles[1-d[0]]))

    if saveData:
      ApStr = '0p5'
      if d[0]==1:
        ApStr = '1p0'
      h5f.create_dataset('z_Ap_'+ApStr,       (np.size(xIcc0_sym),),   dtype='f8', data=xIcc0_sym)
      h5f.create_dataset('ePhiDTe_Ap_'+ApStr, (np.size(ePhiDTe_sym),), dtype='f8', data=ePhiDTe_sym)

  plot_verticalLinesPM(z_m, ax5[0])  #[ Indicate location of max B.
  ax5[0].set_xlim( xIcc[0][0], xIcc[0][-1] )
  ax5[0].legend([hpl5a[0][0], hpl5a[1][0]],[r'$A_p=0.5$', r'$A_p=1$'], fontsize=legendFontSize, frameon=False)

  #[ Plot the potential drop as a function of nuFrac_e:
  deltaPhi = np.zeros(np.size(nuFrac_e))
  for d, nuFracElc in np.ndenumerate(nuFrac_e):
    ePhiDTe = np.zeros(iUp-iLo)
    c = 0
    for i in range(iLo,iUp):
      B = bmag[i]
      R_m = np.amax(bmag)/B
      ePhiDTe[c] = optimize.ridder(rootEq, 1.e-10, 20, args=(R_m, nuFracElc))
      c = c+1

    deltaPhi[d] = np.amax(ePhiDTe)-np.amin(ePhiDTe)

  hpl5b.append(ax5[1].semilogx(nuFrac_e,deltaPhi, marker='.'))
  ax5[1].set_xlim( nuFrac_e[0], nuFrac_e[-1] )

  for i in range(2):
    setTickFontSize(ax5[i],tickFontSize)
  ax5[0].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize, labelpad=-2)
  ax5[0].set_ylabel(r'$e\phi/T_{e0}$', fontsize=xyLabelFontSize)
  ax5[1].set_xlabel(r'$A_p$', fontsize=xyLabelFontSize, labelpad=-2)
  ax5[1].set_ylabel(r'$e\Delta\phi/T_{e0}$', fontsize=xyLabelFontSize)
  plt.text(0.05, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax5[0].transAxes)
  plt.text(0.05, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax5[1].transAxes)

  if outFigureFile:
    plt.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f.create_dataset('A_p',          (np.size(nuFrac_e),), dtype='f8', data=nuFrac_e)
    h5f.create_dataset('eDeltaPhiDTe', (np.size(deltaPhi),), dtype='f8', data=deltaPhi)
    h5f.close()
  

#................................................................................#

if plot_phiAdiabatic:
  #[ Plot phi(z) and e*Phi/Te0 for a single frame, or phi(z,t) and phi(z=0,t).

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk57/'
  
  fileName = 'gk57-wham1x2v'    #.Root name of files to process.

  #[ Prepare figure.
  figProp7 = (6.4,4.5)
  ax7Pos   = [[0.12, 0.57, 0.725, 0.42],
              [0.12, 0.13, 0.725, 0.42]]
  cax7Pos  = [[0.855, 0.57, 0.02, 0.42]]
  fig7     = plt.figure(figsize=figProp7)
  ax7      = [fig7.add_axes(pos) for pos in ax7Pos]
  cbar_ax7 = [fig7.add_axes(pos) for pos in cax7Pos]
  
  phiFile = dataDir+fileName+'_phi_'    #.Complete file name.
  nFrames = 1+pgu.findLastFrame(phiFile, 'bp')
  times = 1.e6*pgu.getTimeStamps(phiFile,0,nFrames-1)
  
  phiFile = dataDir+fileName+'_phi_%d.bp'    #.Complete file name.
  
  #[ Load the grid.
  xInt, _, nxInt, lxInt, dxInt, _ = pgu.getGrid(phiFile % 0,polyOrder,basisType)
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(phiFile % 0,polyOrder,basisType,location='center')
  
  phi = np.zeros((nFrames,nxIntC[0]))
  
  for fI in range(nFrames):
    phi[fI,:] = np.squeeze(pgu.getInterpData(phiFile % fI, polyOrder, basisType))
  
  #[ Create colorplot grid (coordinates have to be nodal).
  timesC  = 0.5*(times[1:]+times[0:-1])
  Xnodal = [np.outer(np.concatenate([[timesC[0]-(timesC[1]-timesC[0])],timesC,[timesC[-1]+(timesC[-1]-timesC[-2])]]), \
                     np.ones(np.shape(xInt[0]))), \
            np.outer(np.ones(np.size(timesC)+2),xInt[0])]
  
  hpl7a = ax7[0].pcolormesh(Xnodal[0], Xnodal[1], eV*phi/Te0, cmap='inferno')
  hcb7a = plt.colorbar(hpl7a, ax=ax7[0], cax=cbar_ax7[0], ticks=[0., 2.5, 5., 7.5, 10.])
  #[ Plots lines at mirror throat:
  ax7[0].plot([Xnodal[0][0][0], Xnodal[0][-1][0]],   [z_m, z_m], color='white', linestyle='--', alpha=0.5)
  ax7[0].plot([Xnodal[0][0][0], Xnodal[0][-1][0]], [-z_m, -z_m], color='white', linestyle='--', alpha=0.5)

#  #[ Plot central value over time:
  hpl7b = ax7[1].plot(times, (eV/Te0)*0.5*(phi[:,nxIntC[0]//2-1]+phi[:,nxIntC[0]//2]), color=defaultColors[0])
#  #[ Create a set of line segments so that we can color them individually
#  #[ This creates the points as a N x 1 x 2 array so that we can stack points
#  #[ together easily to get the segments. The segments array for line collection
#  #[ needs to be (numlines) x (points per line) x 2 (for x and y)
#  x = times;  y = (eV/Te0)*0.5*(phi[:,nxIntC[0]//2-1]+phi[:,nxIntC[0]//2])
#  #x = np.linspace(0, 3 * np.pi, 500);  y = np.sin(x)
#  points = np.array([x, y]).T.reshape(-1, 1, 2)
#  segments = np.concatenate([points[:-1], points[1:]], axis=1)
#  # Create a continuous norm to map from data points to colors
#  norm = plt.Normalize(y.min(), y.max())
#  lc = LineCollection(segments, cmap='inferno', norm=norm)
#  # Set the values used for colormapping
#  lc.set_array(y)
#  lc.set_linewidth(2)
#  line = ax7[1].add_collection(lc)

  ax7[1].set_xlim( np.amin(times),np.amax(times) )
  ax7[1].set_ylim( 0., 12. ) #np.amin(y),np.amax(y) )
  hcb7a.set_label('$e\phi/T_{e0}$', rotation=270, labelpad=18, fontsize=colorBarLabelFontSize)
  hcb7a.ax.tick_params(labelsize=tickFontSize)
  plt.setp( ax7[0].get_xticklabels(), visible=False)
  ax7[0].set_ylabel(r'$z$ (m)', fontsize=xyLabelFontSize)
  ax7[1].set_xlabel(r'Time ($\mu$s)', fontsize=xyLabelFontSize)
  ax7[1].set_ylabel(r'$e\phi(z=0)/T_{e0}$', fontsize=xyLabelFontSize)
  plt.text(0.025, 0.85, r'(a)', fontsize=textFontSize, color='white', transform=ax7[0].transAxes)
  plt.text(0.025, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax7[1].transAxes)
  for i in range(len(ax7)):
#    ax7[i].set_xlim( times[0], times[-1] )
    setTickFontSize(ax7[i],tickFontSize) 

  figName = fileName+'_phiVtime_0-'+str(nFrames-1)
  if outFigureFile:
    fig7.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")
    h5f.create_dataset('time',           np.shape(Xnodal[0]), dtype='f8', data=Xnodal[0])
    h5f.create_dataset('z',              np.shape(Xnodal[1]), dtype='f8', data=Xnodal[1])
    h5f.create_dataset('ePhi0DTe_time', (np.size(x),),        dtype='f8', data=x)
    h5f.create_dataset('ePhi0DTe',      (np.size(y),),        dtype='f8', data=y)
    h5f.close()
  
#................................................................................#

# Experiment with different computation of Tperp
#  1) First check if computing Tperp = (m/2)* (2*pi/(m*n))* int dvpar dmu (2*mu*B/m) f gives the same as Gkeyll's Tperp diagnostic.
#  2) Repeat using the softening factor: Tperp = (m/2)* (2*pi/(m*n))*int dvpar dmu Pi(z)*(2*mu*B/m) f.
#   n*2*Tperp/m = (2*pi/m)*int dvpar dmu Pi(z)*(2*mu*B/m) f.
if plot_momAdiabatic:
  #[ Plot density, temperature and u_parallel along the field line for the kinetic elc sim,
  #[ with and without force softening.

  dataDir = ['/scratch/gpfs/manaurer/gkeyll/mirror/gk57/',  #[ w/o force softening.
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk71/',] #[ w/ force softening.
  
  #.Root name of files to process.
  fileName = ['gk57-wham1x2v',  #[ w/o force softening.
              'gk71-wham1x2v',] #[ w/ force softening.
  frame    = 128

  figName = 'gk57-71-wham1x2v_M0TempUpar_'+str(frame)

  c_s = np.sqrt(Te0/mi)
  
  #[ Prepare figure.
  figProp16 = (7.,8.5)
  ax16Pos   = [[0.11, 0.750, 0.77, 0.215],
               [0.11, 0.525, 0.77, 0.215],
               [0.11, 0.300, 0.77, 0.215],
               [0.11, 0.075, 0.77, 0.215]]
  fig16     = plt.figure(figsize=figProp16)
  ax16      = [fig16.add_axes(d) for d in ax16Pos]

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  for fI in range(len(dataDir)):
    fName = dataDir[fI]+fileName[fI]+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.
  
    #[ Load the electron density, temperatures and flows.
    den   = np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='M0'))
    upar  = (1./c_s)*np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Upar'))
    temp  = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Temp'))
    tpar  = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Tpar'))
    if fI==0:
      tperp = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Tperp'))
    else:
      #[ Alternative calculation of T_perp.
      dBdzFile = './'+fileName[fI]+'-dBdz_dBdz.bp'    #.Complete file name.
      fsPi = forceSofteningPi_zvparmu(dataDir[fI]+fileName[fI],'ion',frame,dBdzFile)
      tperp = (1.e-3/eV)*calcTperp(dataDir[fI]+fileName[fI],'ion',frame,phaseFac=fsPi)

    #[ Load the grid.
    xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType,location='center',varName='M0')

    hpl16 = list()
    hpl16.append(ax16[0].semilogy(xIntC[0], den, color=defaultColors[0], linestyle=lineStyles[2*fI], marker=markers[4*fI], markevery=12, markersize=4))
    hpl16.append(ax16[1].plot(xIntC[0], upar,    color=defaultColors[0], linestyle=lineStyles[2*fI], marker=markers[4*fI], markevery=12, markersize=4))
    hpl16.append(ax16[2].plot(xIntC[0], tpar,    color=defaultColors[0], linestyle=lineStyles[2*fI], marker=markers[4*fI], markevery=12, markersize=4))
    hpl16.append(ax16[3].plot(xIntC[0], tperp/tpar,   color=defaultColors[0], linestyle=lineStyles[2*fI], marker=markers[4*fI], markevery=12, markersize=4))

    if saveData:
      fsStr = ''
      if fI == 1:
        fsStr = '_fs'
      h5f.create_dataset('den'+fsStr,   np.shape(den),      dtype='f8', data=den)
      h5f.create_dataset('upar'+fsStr,  np.shape(upar),     dtype='f8', data=upar)
      h5f.create_dataset('Tpar'+fsStr,  np.shape(tpar),     dtype='f8', data=tpar)
      h5f.create_dataset('Tperp'+fsStr, np.shape(tperp),    dtype='f8', data=tperp)

  ax16[0].set_ylim(0., n0)
  for i in range(len(ax16)):
    ax16[i].set_xlim( xIntC[0][0], xIntC[0][-1] )
    setTickFontSize(ax16[i],tickFontSize) 
    ax16[i].tick_params(axis='y', labelcolor=defaultColors[0], labelsize=tickFontSize)
    hmag = ax16[i].yaxis.get_offset_text().set_size(tickFontSize)
  for i in range(3):
    plt.setp( ax16[i].get_xticklabels(), visible=False)
  ax16[0].set_ylabel(r'$n_i$ (m$^{-3}$)', fontsize=xyLabelFontSize, color=defaultColors[0], labelpad=-4)
  ax16[1].set_ylabel(r'$u_{\parallel i}/c_{se0}$', fontsize=xyLabelFontSize, color=defaultColors[0], labelpad=-2)
  ax16[2].set_ylabel(r'$T_{\parallel i}$ (keV)', fontsize=xyLabelFontSize, color=defaultColors[0])
  ax16[3].set_ylabel(r'$T_{\perp i}$ (keV)', fontsize=xyLabelFontSize, color=defaultColors[0])
  plt.text(0.025, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax16[0].transAxes)
  plt.text(0.025, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax16[1].transAxes)
  plt.text(0.025, 0.85, r'(c)', fontsize=textFontSize, color='black', transform=ax16[2].transAxes)
  plt.text(0.025, 0.85, r'(d)', fontsize=textFontSize, color='black', transform=ax16[3].transAxes)

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

if plot_momAdiabaticOld:
  #[ Plot density, temperature and u_parallel along the field line for the adiabatic elc sim.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk57/'
  
  fileName = 'gk57-wham1x2v'    #.Root name of files to process.
  frame    = 144

  #[ Prepare figure.
  figProp8 = (6.8,7.2)
  ax8Pos   = [[0.12, 0.685, 0.77, 0.285],
              [0.12, 0.385, 0.77, 0.285],
              [0.12, 0.085, 0.77, 0.285]]
  fig8     = plt.figure(figsize=figProp8)
  ax8      = [fig8.add_axes(d) for d in ax8Pos]

  fName = dataDir+fileName+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.

  #[ Load the grid.
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType,location='center',varName='M0')
  
  #[ Load the electron density, temperatures and flows.
  den  = pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='M0')
  upar = pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Upar')
  temp = (1.e-3/eV)*pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Temp')
  tpar = (1.e-3/eV)*pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Tpar')
  tperp = (1.e-3/eV)*pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Tperp')

  c_s = np.sqrt(Te0/mi)
#  c_s = np.sqrt((Te0+3.*Ti0)/mi)
#  c_s = np.sqrt((Te0+3.*tpar/(1.e-3/eV))/mi)

  hpl8 = list()
  hpl8.append(ax8[0].plot(xIntC[0], den, color=defaultColors[0], linestyle=lineStyles[0]))
  hpl8.append(ax8[1].plot(xIntC[0], upar/c_s, color=defaultColors[0], linestyle=lineStyles[0]))
  hpl8.append(ax8[2].plot(xIntC[0], tpar, color=defaultColors[0], linestyle=lineStyles[0]))

  ax82_twin = ax8[2].twinx()
  hpl82_twin = list()
  hpl82_twin.append(ax82_twin.plot(xIntC[0], tperp, color=defaultColors[1], linestyle=lineStyles[1]))

  for i in range(len(ax8)):
    ax8[i].set_xlim( xIntC[0][0], xIntC[0][-1])
    setTickFontSize(ax8[i],tickFontSize) 
    ax8[i].tick_params(axis='y', labelcolor=defaultColors[0])
    hmag = ax8[i].yaxis.get_offset_text().set_size(tickFontSize)
    plot_verticalLinesPM(z_m, ax8[i])  #[ Indicate location of max B.
  hmag = ax82_twin.yaxis.get_offset_text().set_size(tickFontSize)
  ax82_twin.tick_params(axis='y', labelcolor=defaultColors[1], labelsize=tickFontSize)
  plt.setp( ax8[0].get_xticklabels(), visible=False)
  plt.setp( ax8[1].get_xticklabels(), visible=False)
  ax8[2].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  ax8[0].set_ylabel(r'$n_i$ (m$^{-3}$)', fontsize=xyLabelFontSize, color=defaultColors[0])
  ax8[1].set_ylabel(r'$u_{\parallel i}/c_{se0}$', fontsize=xyLabelFontSize, color=defaultColors[0])
#  ax8[1].set_ylabel(r'$u_{\parallel i}$ (m/s)', fontsize=xyLabelFontSize, color=defaultColors[0])
  ax8[2].set_ylabel(r'$T_{\parallel i}$ (keV)', fontsize=xyLabelFontSize, color=defaultColors[0])
  ax82_twin.set_ylabel(r'$T_{\perp i}$ (keV)', fontsize=xyLabelFontSize, color=defaultColors[1])
  
  if outFigureFile:
    plt.savefig(outDir+fileName+'_M0TempUpar_'+str(frame)+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_fi_vspace_Adiabatic:
  #[ Plot slices of the ion distribution functions at z=0, z=z_m and z=zMax.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk57/'

  fileName = 'gk57-wham1x2v'    #.Root name of files to process.
  frame    = 192

  fName = dataDir+fileName+'_%s_%d.bp'    #.Complete file name.

  #[ Load the grid.
  xInt_i, _, nxInt_i, lxInt_i, dxInt_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType)
  xIntC_i, _, nxIntC_i, lxIntC_i, dxIntC_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType,location='center')

  #[ Load the magnetic field and electrostatic potential.
  dataFile = dataDir + fileName + '_allGeo_0.bp'
  bmag = np.squeeze(pgu.getInterpData(dataFile,polyOrder,basisType,location='center',varName='bmag'))
  dataFile = dataDir + fileName + '_phi_'+str(frame)+'.bp'
  phi  = np.squeeze(pgu.getInterpData(dataFile,polyOrder,basisType,location='center'))

  #.Compute the trapped-passing boundary in vpar-mu space (normalized).
  B_p  = np.interp(0., xIntC_i[0], bmag)
  R_m  = np.interp(z_m, xIntC_i[0], bmag)/B_p
  Dphi = np.interp(0., xIntC_i[0], phi)-np.interp(z_m, xIntC_i[0], phi)
  muBound = np.zeros(np.shape(xIntC_i[1]))
  for i in range(nxIntC_i[1]):
    muBound[i] = ( (1./B_p)*(0.5*mi*xIntC_i[1][i]**2+qi*Dphi)/(R_m-1) )/(0.5*mi*(vti**2)/B_p)

  #[ Normalize velocity space
  xInt_i[1] = xInt_i[1]/vti
  xInt_i[2] = xInt_i[2]/(0.5*mi*(vti**2)/B_p)
  xIntC_i[1] = xIntC_i[1]/vti
  xIntC_i[2] = xIntC_i[2]/(0.5*mi*(vti**2)/B_p)

  #[ Get indices along z of slices we wish to plot:
  plotz = [0., z_m, xInt_i[0][-1]]
  plotzIdx = [np.argmin(np.abs(xIntC_i[0]-val)) for val in plotz]
  
  #[ Load the electron and ion distribution functions.
  fIon  = np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType))

  #[ Prepare figure.
  figProp9 = (10.4, 4.)
  ax9Pos   = [[0.06, 0.15, 0.305, 0.74],[0.375, 0.15, 0.305, 0.74],[0.69, 0.15, 0.305, 0.74],]
  ca9Pos   = [[0.07, 0.9, 0.285, 0.02],[0.385, 0.9, 0.285, 0.02],[0.70, 0.9, 0.285, 0.02]]
#  figProp9 = (5.,10.4)
#  ax9Pos   = [[0.13, 0.690, 0.85, 0.295],
#              [0.13, 0.375, 0.85, 0.295],
#              [0.13, 0.060, 0.85, 0.295],]
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
  ax9in[0], hpl9bin[0] = plot_inset_pcolormesh(ax9[0], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[0],:,:], \
                                               inset_size, sub_coords, "white", extreme_vals)
  #[ Plot trapped-passing boundary.
  ax9[0].plot(xIntC_i[1], muBound, ':', color='white',alpha=0.5,linewidth=3)
  ax9in[0].plot(xIntC_i[1], muBound, ':', color='white',alpha=0.5,linewidth=3)

  hcb9b.append(plt.colorbar(hpl9b[0], ax=ax9[0], cax=cbar_ax9[0], orientation='horizontal'))
  hcb9b[0].ax.xaxis.set_ticks_position('top')
  hcb9b[0].ax.tick_params(labelsize=tickFontSize)
  ax9[0].set_ylabel(r'$\mu/[m_iv_{ti0}^2/(2B_p)]$', fontsize=xyLabelFontSize)

  sub_coords = [[-0.5, 1.5], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[1],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  hpl9b.append(ax9[1].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[1],:,:],
                                 cmap='inferno', vmin=extreme_vals[0], vmax=extreme_vals[1]))
  #[ Plot trapped passing boundary
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

if plot_phi_forceSoft:
  #[ Plot phi(z) at a frame comparing adiabatic electron runs w/ and w/o force softening.

  dataDir = ['/scratch/gpfs/manaurer/gkeyll/mirror/gk57/',
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk71/']
  
  fileName = ['gk57-wham1x2v','gk71-wham1x2v']    #.Root name of files to process.
  frame    = 192
  figName  = 'gk57-71-wham1x2v_ePhiDTe_forceSoft_'+str(frame) 

  #[ Prepare figure.
  figProp10 = (6,3.)
  ax10Pos   = [[0.1, 0.18, 0.88, 0.795],]
  fig10     = plt.figure(figsize=figProp10)
  ax10      = [fig10.add_axes(pos) for pos in ax10Pos]
  
  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  hpl10 = list()
  for fI in range(len(fileName)):
    phiFile = dataDir[fI]+fileName[fI]+'_phi_%d.bp'    #.Complete file name.
    
    #[ Load the grid.
    xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(phiFile % frame,polyOrder,basisType,location='center')
    
    #[ Load phi.
    phi = np.squeeze(pgu.getInterpData(phiFile % frame, polyOrder, basisType))
    
    ePhiDTe = eV*phi/Te0
    
    hpl10.append(ax10[0].plot(xIntC[0], ePhiDTe, color=defaultColors[fI], linestyle=lineStyles[fI]))
    ax10[0].set_xlim( xIntC[0][0], xIntC[0][-1] )
    ax10[0].set_ylim( 0., 12. )

    if saveData:
      fsStr = ''
      if fI == 1:
        fsStr = '_fs'
      h5f.create_dataset('ePhiDTe'+fsStr, np.shape(ePhiDTe), dtype='f8', data=ePhiDTe)

  setTickFontSize(ax10[0],tickFontSize) 
  plot_verticalLinesPM(z_m, ax10[0])  #[ Indicate location of max B.
  ax10[0].set_ylabel(r'$e\phi/T_{e0}$', fontsize=xyLabelFontSize, labelpad=-2)
  ax10[0].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize, labelpad=-1)
  ax10[0].legend([hpl10[0][0], hpl10[1][0]],[r'no force softening', r'w/ force softening'], fontsize=legendFontSize, frameon=False)
  
  if outFigureFile:
    fig10.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()
  
  if saveData:
    h5f.create_dataset('z', np.shape(xIntC[0]), dtype='f8', data=xIntC[0])
    h5f.close()
  
#................................................................................#

if plot_fi_vspace_forceSoft:
  #[ Plot slices of the ion distribution functions at z=0, z=z_m and z=zMax
  #[ for the adiabatic electron sim with force softening.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk71/'

  fileName = 'gk71-wham1x2v'    #.Root name of files to process.
  frame    = 192

  fName = dataDir+fileName+'_%s_%d.bp'    #.Complete file name.

  #[ Load the grid.
  xInt_i, _, nxInt_i, lxInt_i, dxInt_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType)
  xIntC_i, _, nxIntC_i, lxIntC_i, dxIntC_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType,location='center')

  #[ Load the magnetic field and electrostatic potential.
  dataFile = dataDir + fileName + '_allGeo_0.bp'
  bmag = np.squeeze(pgu.getInterpData(dataFile,polyOrder,basisType,location='center',varName='bmag'))
  dataFile = dataDir + fileName + '_phi_'+str(frame)+'.bp'
  phi  = np.squeeze(pgu.getInterpData(dataFile,polyOrder,basisType,location='center'))

  #.Compute the trapped-passing boundary in vpar-mu space (normalized).
  B_p  = np.interp(0., xIntC_i[0], bmag)
  R_m  = np.interp(z_m, xIntC_i[0], bmag)/B_p
  Dphi = np.interp(0., xIntC_i[0], phi)-np.interp(z_m, xIntC_i[0], phi)
  muBound = np.zeros(np.shape(xIntC_i[1]))
  for i in range(nxIntC_i[1]):
    muBound[i] = ( (1./B_p)*(0.5*mi*xIntC_i[1][i]**2+qi*Dphi)/(R_m-1) )/(0.5*mi*(vti**2)/B_p)

  #[ Normalize velocity space
  xInt_i[1] = xInt_i[1]/vti
  xInt_i[2] = xInt_i[2]/(0.5*mi*(vti**2)/B_p)
  xIntC_i[1] = xIntC_i[1]/vti
  xIntC_i[2] = xIntC_i[2]/(0.5*mi*(vti**2)/B_p)

  #[ Get indices along z of slices we wish to plot:
  plotz = [0., z_m, xInt_i[0][-1]]
  plotzIdx = [np.argmin(np.abs(xIntC_i[0]-val)) for val in plotz]
  
  #[ Load the electron and ion distribution functions.
  fIon  = np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType))

  #[ Prepare figure.
  figProp11 = (10.4, 4.)
  ax11Pos   = [[0.06, 0.15, 0.305, 0.74],[0.375, 0.15, 0.305, 0.74],[0.69, 0.15, 0.305, 0.74],]
  ca11Pos   = [[0.07, 0.9, 0.285, 0.02],[0.385, 0.9, 0.285, 0.02],[0.70, 0.9, 0.285, 0.02]]
  fig11     = plt.figure(figsize=figProp11)
  ax11      = [fig11.add_axes(pos) for pos in ax11Pos]
  cbar_ax11 = [fig11.add_axes(pos) for pos in ca11Pos]
  
  #[ Create colorplot grid. Recall coordinates have to be nodal.
  Xnodal_i = [np.outer(xInt_i[1], np.ones(nxInt_i[2])),
              np.outer(np.ones(nxInt_i[1]), xInt_i[2])]

  hpl11b = list()
  hcb11b = list()
  #[ For insets:
  inset_size = [0.5, 0.4, 0.48, 0.57]
  ax11in = [0 for i in range(len(ax11))]
  hpl11bin = [0 for i in range(len(ax11))]
  hpl11bin = [0 for i in range(len(ax11))]

  sub_coords = [[0., 2.], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[0],:,:])]
  hpl11b.append(ax11[0].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[0],:,:],
                                 cmap='inferno', vmin=extreme_vals[0], vmax=extreme_vals[1]))
  ax11in[0], hpl11bin[0] = plot_inset_pcolormesh(ax11[0], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[0],:,:], \
                                               inset_size, sub_coords, "white", extreme_vals)
  #[ Plot trapped-passing boundary.
  ax11[0].plot(xIntC_i[1], muBound, ':', color='white',alpha=0.5,linewidth=3)
  ax11in[0].plot(xIntC_i[1], muBound, ':', color='white',alpha=0.5,linewidth=3)

  hcb11b.append(plt.colorbar(hpl11b[0], ax=ax11[0], cax=cbar_ax11[0], orientation='horizontal'))
  hcb11b[0].ax.xaxis.set_ticks_position('top')
  hcb11b[0].ax.tick_params(labelsize=tickFontSize)

  sub_coords = [[-0.5, 1.5], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[1],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  hpl11b.append(ax11[1].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[1],:,:],
                                 cmap='inferno', vmin=extreme_vals[0], vmax=extreme_vals[1]))
  #[ Plot trapped passing boundary
  ax11in[1], hpl11bin[1] = plot_inset_pcolormesh(ax11[1], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[1],:,:], \
                                               inset_size, sub_coords, "white", extreme_vals)

  hcb11b.append(plt.colorbar(hpl11b[1], ax=ax11[1], cax=cbar_ax11[1], orientation='horizontal'))
  hcb11b[1].ax.xaxis.set_ticks_position('top')
  hcb11b[1].ax.tick_params(labelsize=tickFontSize)

  sub_coords = [[.75, 3.75], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[2],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  hpl11b.append(ax11[2].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[2],:,:],
                                 cmap='inferno', vmin=0., vmax=2.e-1*np.amax(fIon[plotzIdx[2],:,:])))
  ax11in[2], hpl11bin[2] = plot_inset_pcolormesh(ax11[2], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[2],:,:], \
                                               inset_size, sub_coords, "white", extreme_vals)

  hcb11b.append(plt.colorbar(hpl11b[2], ax=ax11[2], cax=cbar_ax11[2], orientation='horizontal'))
  hcb11b[2].ax.xaxis.set_ticks_position('top')
  hcb11b[2].ax.tick_params(labelsize=tickFontSize)

  ax11[0].set_ylabel(r'$\mu/[m_iv_{ti0}^2/(2B_p)]$', fontsize=xyLabelFontSize)
  for i in range(3):
    ax11[i].set_xlabel(r'$v_\parallel/v_{ti0}$', fontsize=xyLabelFontSize, labelpad=-2)
    setTickFontSize(ax11[0+i],tickFontSize) 
  for i in range(1,3):
    plt.setp( ax11[i].get_yticklabels(), visible=False)

  plt.text(0.03, 0.89, r'(a) $f_i(z=0)$', fontsize=textFontSize, color='white', transform=ax11[0].transAxes)
  plt.text(0.03, 0.89, r'(b) $f_i(z=z_m)$', fontsize=textFontSize, color='white', transform=ax11[1].transAxes)
  plt.text(0.03, 0.89, r'(c) $f_i(z=L_z/2)$', fontsize=textFontSize, color='white', transform=ax11[2].transAxes)

  plt.text(0.85, 0.85, r'', fontsize=textFontSize, color='white', transform=ax11[0].transAxes)
  plt.text(0.85, 0.85, r'', fontsize=textFontSize, color='white', transform=ax11[1].transAxes)
  plt.text(0.85, 0.85, r'', fontsize=textFontSize, color='white', transform=ax11[2].transAxes)
  
  figName = fileName+'_fiCenterThroatWall_'+str(frame)
  if outFigureFile:
    fig11.savefig(outDir+figName+figureFileFormat)
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

if plot_phi_forceSoftChiComp:
  #[ Plot phi(z) at a frame comparing adiabatic electron runs w/ various chi factors.

  dataDir = ['/scratch/gpfs/manaurer/gkeyll/mirror/gk68/',
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk66/',
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk71/',]
  
  fileName = ['gk68-wham1x2v','gk66-wham1x2v','gk71-wham1x2v']    #.Root name of files to process.
  frame    = 400
  figName  = 'gk66-68-71-wham1x2v_ePhiDTe_chiComp_'+str(frame)

  #[ Prepare figure.
  figProp12 = (6,3.)
  ax12Pos   = [[0.1, 0.18, 0.88, 0.795],]
  fig12     = plt.figure(figsize=figProp12)
  ax12      = [fig12.add_axes(pos) for pos in ax12Pos]
  
  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  hpl12 = list()
  for fI in range(len(fileName)):
    phiFile = dataDir[fI]+fileName[fI]+'_phi_%d.bp'    #.Complete file name.
    
    #[ Load the grid.
    xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(phiFile % frame,polyOrder,basisType,location='center')
    
    #[ Load phi.
    phi = np.squeeze(pgu.getInterpData(phiFile % frame, polyOrder, basisType))
    
    ePhiDTe = eV*phi/Te0
    
    hpl12.append(ax12[0].plot(xIntC[0], ePhiDTe, color=defaultColors[fI], linestyle=lineStyles[fI]))
    ax12[0].set_xlim( xIntC[0][0], xIntC[0][-1] )
    ax12[0].set_ylim( 0., 12. )

    if saveData:
      chiStr = ''
      if fI < 2:
        chiStr = '_'+str(2-fI)
      h5f.create_dataset('ePhiDTe_chi'+chiStr, np.shape(ePhiDTe), dtype='f8', data=ePhiDTe)

  setTickFontSize(ax12[0],tickFontSize) 
  plot_verticalLinesPM(z_m, ax12[0])  #[ Indicate location of max B.
  ax12[0].set_ylabel(r'$e\phi/T_{e0}$', fontsize=xyLabelFontSize, labelpad=-2)
  ax12[0].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize, labelpad=-1)
  ax12[0].legend([hpl12[0][0], hpl12[1][0], hpl12[2][0]],[r'$\chi_1$', r'$\chi_2$', r'$\chi$'], fontsize=legendFontSize, frameon=False)
  
  if outFigureFile:
    fig12.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()
  
  if saveData:
    h5f.create_dataset('z', np.shape(xIntC[0]), dtype='f8', data=xIntC[0])
    h5f.close()
  
#................................................................................#

if plot_kperpScan:
  #[ Plot e*phi/Te for all kperp*rhos, and e*Delta phi/Te0 vs. kperp*rhos.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/'

  tests = ['gk32','gk43','gk44','gk45','gk41','gk46','gk48','gk52']
  kperp = [0.3, 0.25,0.2,0.15,0.1,0.05,0.01,0.005]
  tests = tests[::-1]
  kperp = kperp[::-1]
  
  fileName = '%s-wham1x2v'    #.Root name of files to process.
  frame    = 32
  frame_gk32 = 152
  figName = 'gk-wham1x2v_kperpScan_phi_eDeltaPhiDTe_'+str(frame)

  #[ Prepare figure.
  figProp13 = (7,5.5)
  ax13Pos   = [[0.12, 0.59, 0.72, 0.39],
              [0.12, 0.1, 0.72, 0.39]] 
  cax13Pos  = [[0.845, 0.59, 0.02, 0.39]]
  fig13     = plt.figure(figsize=figProp13)
  ax13      = [fig13.add_axes(pos) for pos in ax13Pos]
  cbar_ax13 = [fig13.add_axes(cax13Pos[0])]

  hpl13a, hpl13b = list(), list()
  legends = list()

  ax13[0].set_prop_cycle(plt.cycler('color', plt.cm.viridis(np.linspace(0, 1, len(kperp)))))

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  eDeltaPhiDTe = np.zeros(len(tests))
  for tI in range(len(tests)):

    fFrame = frame
    phiFile = dataDir+tests[tI]+'/'+fileName%tests[tI]+'_phi_%d.bp'    #.Complete file name.
    if tests[tI]=='gk32':
      fFrame = frame_gk32
  
    #[ Load the grid.
    x, _, nx, lx, dx, _ = pgu.getGrid(phiFile % fFrame,polyOrder,basisType,location='center')
    
    #[ Load phi.
    phi = np.squeeze(pgu.getInterpData(phiFile % fFrame, polyOrder, basisType))

    ePhiDTe = eV*phi/Te0
    eDeltaPhiDTe[tI] = eV*(np.interp(0., x[0], phi)-np.interp(z_m, x[0], phi))/Te0
  
    hpl13a.append(ax13[0].plot(x[0], ePhiDTe))
    legends.append(r'$k_\perp\rho_s$='+str(kperp[tI]))

    if saveData:
      kperpStr = str(kperp[tI]).replace('0.','0p')
      h5f.create_dataset('ePhiDTe_kperpRhos_'+kperpStr, np.shape(ePhiDTe), dtype='f8', data=ePhiDTe)
      if tI==0:
        h5f.create_dataset('z', np.shape(x[0]), dtype='f8', data=x[0])
      elif tI==len(tests)-1:
        h5f.create_dataset('z_kperpRhos_0p3', np.shape(x[0]), dtype='f8', data=x[0])

  hpl13b.append(ax13[1].plot(kperp, eDeltaPhiDTe, color=defaultColors[0], linestyle=lineStyles[0], marker='.'))
  plot_verticalLinesPM(z_m, ax13[0])  #[ Indicate location of max B.

  #[ Create scalar mappable for colorbar indicating kperp.
  cmap, bounds = mpl.cm.viridis, kperp
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  hcb13 = fig13.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cbar_ax13[0])
  hcb13.set_label(r'$k_\perp\rho_s$', rotation=270, labelpad=+16, fontsize=colorBarLabelFontSize)
  hcb13.ax.tick_params(labelsize=tickFontSize)

  ax13[0].set_xlim( x[0][0], x[0][-1] )
  ax13[0].set_ylabel(r'$e\phi/T_{e0}$', fontsize=xyLabelFontSize)
  ax13[0].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize, labelpad=-1)
  ax13[1].set_xlim( kperp[0], kperp[-1] )
  ax13[1].set_ylabel(r'$e\Delta\phi/T_{e0}$', fontsize=xyLabelFontSize)
  ax13[1].set_xlabel(r'$k_\perp\rho_s$', fontsize=xyLabelFontSize, labelpad=-1)
  plt.text(0.92, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax13[0].transAxes)
  plt.text(0.92, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax13[1].transAxes)
  for i in range(2):
    setTickFontSize(ax13[i],tickFontSize) 
  
  if outFigureFile:
    fig13.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()
  
  if saveData:
    h5f.create_dataset('kperpRhos',    np.shape(kperp),        dtype='f8', data=kperp)
    h5f.create_dataset('eDeltaPhiDTe', np.shape(eDeltaPhiDTe), dtype='f8', data=eDeltaPhiDTe)
    h5f.close()
  
#................................................................................#

if plot_resScan:
  #[ Plot moments and e*phi/Te for various resolutions of kinetic and adiabatic
  #[ electron simulations.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/'

  tests = [
           ['gk54','gk49','gk48','gk50','gk51','gk53','gk55'],  #[ Kinetic elc.
           ['gk57','gk60','gk63','gk61'],                       #[ Adiabatic elc. 
          ]
  res   = [
           ['64x32','96x32','192x32','384x32','192x64','192x128','64x192'],  #[ Kinetic elc.
           ['64x192','128x192','256x192','64x320'],                          #[ Adiabatic elc. 
          ]
  frame = [
           24,  #[ Kinetic elc.
           42,  #[ Adiabatic elc. 
          ]
  model = ['kinElc','adiElc']
  
  fileName = '%s-wham1x2v'    #.Root name of files to process.

  for mI in range(2):
    #[ Plot e*phi/Te0.
    figProp14 = (6.5,3.5)
    ax14Pos   = [[0.11, 0.18, 0.88, 0.795],]
    fig14     = plt.figure(figsize=figProp14)
    ax14      = [fig14.add_axes(pos) for pos in ax14Pos]
  
    hpl14 = list()
    legends = list()
    for tI in range(len(tests[mI])):
  
      phiFile = dataDir+tests[mI][tI]+'/'+fileName%tests[mI][tI]+'_phi_%d.bp'    #.Complete file name.
    
      #[ Load the grid.
      x, _, nx, lx, dx, _ = pgu.getGrid(phiFile % frame[mI],polyOrder,basisType,location='center')
      
      #[ Load phi and Te.
      phi = pgu.getInterpData(phiFile % frame[mI], polyOrder, basisType)
    
      ePhiDTe = eV*phi/Te0
    
      hpl14.append(ax14[0].plot(x[0], ePhiDTe, color=defaultColors[tI], linestyle=lineStyles[tI], marker=markers[tI], markevery=10, markersize=4))
      legends.append(res[mI][tI])
  
    plot_verticalLinesPM(z_m, ax14[0])  #[ Indicate location of max B.
    ax14[0].set_xlim( x[0][0], x[0][-1] )
    if mI==0:
      ax14[0].set_ylim(0., 3.5)
    else:
      ax14[0].set_ylim(0., 12.0)
    ax14[0].set_ylabel(r'$e\phi/T_{e0}$', fontsize=xyLabelFontSize)
    ax14[0].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
    ax14[0].legend(legends, fontsize=legendFontSize, frameon=False, loc='upper right')
    setTickFontSize(ax14[0],tickFontSize) 
    
    if outFigureFile:
      fig14.savefig(outDir+'gk-wham1x2v_resScan_'+model[mI]+'_ePhiDTe_'+str(frame[mI])+figureFileFormat)
      plt.close()
    else:
      plt.show()
  
    #[ Plot the moments
    species = ["ion","elc"]
    if mI==1:
      species = ["ion"]
    for spec in species:
      figProp15 = (6.8,8.5)
      ax15Pos   = [[0.13, 0.750, 0.85, 0.215],
                   [0.13, 0.525, 0.85, 0.215],
                   [0.13, 0.300, 0.85, 0.215],
                   [0.13, 0.075, 0.85, 0.215]]
      fig15     = plt.figure(figsize=figProp15)
      ax15      = [fig15.add_axes(d) for d in ax15Pos]
    
      hpl15 = list()
      for tI in range(len(tests[mI])):
    
        fName = dataDir+tests[mI][tI]+'/'+fileName%tests[mI][tI]+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.
    
        #[ Load the grid.
        xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % (spec,frame[mI]),polyOrder,basisType,location='center',varName='M0')
        
        #[ Load the electron density, temperatures and flows.
        den  = pgu.getInterpData(fName % (spec,frame[mI]), polyOrder, basisType, varName='M0')
        upar = pgu.getInterpData(fName % (spec,frame[mI]), polyOrder, basisType, varName='Upar')
        temp = (1.e-3/eV)*pgu.getInterpData(fName % (spec,frame[mI]), polyOrder, basisType, varName='Temp')
        tpar = (1.e-3/eV)*pgu.getInterpData(fName % (spec,frame[mI]), polyOrder, basisType, varName='Tpar')
        tperp = (1.e-3/eV)*pgu.getInterpData(fName % (spec,frame[mI]), polyOrder, basisType, varName='Tperp')
    
        c_s = np.sqrt(Te0/mi)
    
        hpl15.append(ax15[0].plot(xIntC[0], den,      color=defaultColors[tI], linestyle=lineStyles[tI], marker=markers[tI], markevery=10, markersize=4))
        hpl15.append(ax15[1].plot(xIntC[0], upar/c_s, color=defaultColors[tI], linestyle=lineStyles[tI], marker=markers[tI], markevery=10, markersize=4))
        hpl15.append(ax15[2].plot(xIntC[0], tpar,     color=defaultColors[tI], linestyle=lineStyles[tI], marker=markers[tI], markevery=10, markersize=4))
        hpl15.append(ax15[3].plot(xIntC[0], tperp,    color=defaultColors[tI], linestyle=lineStyles[tI], marker=markers[tI], markevery=10, markersize=4))
    
      for i in range(len(ax15)):
        ax15[i].set_xlim( xIntC[0][0], xIntC[0][-1])
        setTickFontSize(ax15[i],tickFontSize) 
        hmag = ax15[i].yaxis.get_offset_text().set_size(tickFontSize)
        plot_verticalLinesPM(z_m, ax15[i])  #[ Indicate location of max B.
      for i in range(3):
        plt.setp( ax15[i].get_xticklabels(), visible=False)
      ax15[0].set_ylabel(r'$n_%s$ (m$^{-3}$)' % (spec[0]), fontsize=xyLabelFontSize)
      ax15[1].set_ylabel(r'$u_{\parallel %s}/c_{se0}$' % (spec[0]), fontsize=xyLabelFontSize)
      ax15[2].set_ylabel(r'$T_{\parallel %s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize)
      ax15[3].set_ylabel(r'$T_{\perp %s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize)
      ax15[3].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
      if mI==0:
        ax15[0].legend([hpl15[i*4][0] for i in range(4)], legends[0:4], fontsize=legendFontSize, frameon=False)
        ax15[1].legend([hpl15[i*4][0] for i in range(4,7)], legends[4:7], fontsize=legendFontSize, frameon=False, loc='lower right')
      else:
        ax15[0].legend([hpl15[i*4][0] for i in range(len(res[mI]))], legends[0:len(res[mI])], fontsize=legendFontSize, frameon=False)
      
      if outFigureFile:
        plt.savefig(outDir+'gk-wham1x2v_resScan_'+model[mI]+'_'+spec+'_M0TempUpar_'+str(frame[mI])+figureFileFormat)
        plt.close()
      else:
        plt.show()

#................................................................................#

if print_noNegCells:
  #[ Print the number of negative cells in a file.

  dataFile = '/scratch/gpfs/manaurer/gkeyll/mirror/gk55/gk55-wham1x2v_elc_128.bp'

  #[ Load data.
  dataIn = np.squeeze(pgu.getInterpData(dataFile, polyOrder, basisType))

  cellsTot = np.size(dataIn)
  print(np.shape(dataIn), cellsTot)

  negCells = np.sum(dataIn < 0.)

  normData = np.zeros(np.shape(dataIn))
  for i in range(np.size(dataIn,0)):
    normData[i,:,:] = dataIn[i,:,:]/np.amax(dataIn[i,:,:])
  negCellsNorm = np.sum(dataIn < -1.e-8)

  print("  Number of negative cells:",negCells)
  print("  Number of cells with a relative amplitude <-1e-8:",negCellsNorm)
  print("  Total number of cells:   ",cellsTot)

#................................................................................#

if plot_mom_ic:
  #[ Plot initial density, temperature and u_parallel.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk55/'
  
  fileName = 'gk55-wham1x2v'    #.Root name of files to process.
  frame    = 0
  figName  = fileName+'_M0TempUpar_'+str(frame)

  #[ Prepare figure.
  figProp16 = (7.,6.4)
  ax16Pos   = [[0.11, 0.69 , 0.77, 0.285],
               [0.11, 0.395, 0.77, 0.285],
               [0.11, 0.1  , 0.77, 0.285]]
  fig16     = plt.figure(figsize=figProp16)
  ax16      = [[],[]]
  ax16[0]   = [fig16.add_axes(d) for d in ax16Pos]
  ax16[1]   = [ax.twinx() for ax in ax16[0]]

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  fName = dataDir+fileName+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.

  c_s = np.sqrt(Te0/mi)

  sI = 0
  for spec in ["ion","elc"]:
    #[ Load the grid.
    xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % (spec,frame),polyOrder,basisType,location='center',varName='M0')
    
    #[ Load the electron density, temperatures and flows.
    den   =            np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='M0'))
    upar  = (1./c_s)*  np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Upar'))
    temp  = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Temp'))


    hpl16 = list()
    hpl16.append(ax16[sI][0].semilogy(xIntC[0], den, color=defaultColors[sI], linestyle=lineStyles[sI]))
    hpl16.append(ax16[sI][1].plot(xIntC[0], upar,    color=defaultColors[sI], linestyle=lineStyles[sI]))
    hpl16.append(ax16[sI][2].plot(xIntC[0], temp,    color=defaultColors[sI], linestyle=lineStyles[sI]))

    if saveData:
      sStr = '_i'
      if spec == 'elc':
        sStr = '_e'
        h5f.create_dataset('z', np.shape(xIntC[0]), dtype='f8', data=xIntC[0])
      h5f.create_dataset('den'+sStr,   np.shape(den),      dtype='f8', data=den)
      h5f.create_dataset('upar'+sStr,  np.shape(upar),     dtype='f8', data=upar)
      h5f.create_dataset('Temp'+sStr,  np.shape(temp),     dtype='f8', data=temp)

    ax16[sI][0].set_ylim(0., n0*4./3.)
    for i in range(len(ax16[sI])):
      ax16[sI][i].set_xlim( xIntC[0][0], xIntC[0][-1] )
      setTickFontSize(ax16[sI][i],tickFontSize) 
      ax16[sI][i].tick_params(axis='y', labelcolor=defaultColors[sI], labelsize=tickFontSize)
      hmag = ax16[sI][i].yaxis.get_offset_text().set_size(tickFontSize)
    for i in range(2):
      plt.setp( ax16[sI][i].get_xticklabels(), visible=False)
    ax16[sI][0].set_ylabel(r'$n_%s$ (m$^{-3}$)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI], labelpad=-4+4*sI)
    ax16[sI][1].set_ylabel(r'$u_{\parallel %s}/c_{se0}$' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI], labelpad=-2)
    ax16[sI][2].set_ylabel(r'$T_{%s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI])
    sI = sI+1

  for i in range(len(ax16[0])):
    plot_verticalLinesPM(z_m, ax16[0][i])  #[ Indicate location of max B.
  ax16[0][2].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  plt.text(0.025, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax16[0][0].transAxes)
  plt.text(0.025, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax16[0][1].transAxes)
  plt.text(0.025, 0.85, r'(c)', fontsize=textFontSize, color='black', transform=ax16[0][2].transAxes)
  
  if outFigureFile:
    plt.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f.close()

#................................................................................#

if plot_mom_ic_final:
  #[ Plot density, temperature and u_parallel along the field
  #[ line for the kinetic elc sim with a nonlinear polarization
  #[ and the initial conditions.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk75wExBenergy_vparRes2x/'
  
  #.Root name of files to process.
  fileName = 'gk75-wham1x2v'
  frame    = [0,400]
  species  = [["ion"],["ion","elc"]]
  mycolors = ['black',defaultColors[0],defaultColors[1]]
  myliness = [':',lineStyles[0],lineStyles[1]]
  figName  = 'gk75-wham1x2v_M0TempUpar_'+str(frame[0])+'_'+str(frame[1])

  #[ Prepare figure.
  figProp21 = (7.,8.5)
  ax21Pos   = [[0.11, 0.750, 0.77, 0.215],
               [0.11, 0.525, 0.77, 0.215],
               [0.11, 0.300, 0.77, 0.215],
               [0.11, 0.075, 0.77, 0.215]]
  fig21     = plt.figure(figsize=figProp21)
  ax21      = [[],[]]
  ax21[0]   = [fig21.add_axes(d) for d in ax21Pos]
  ax21[1]   = [ax.twinx() for ax in ax21[0]]

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  c_s = np.sqrt(Te0/mi)

  tI = 0
  for fI in range(len(frame)):

    fName = dataDir+fileName+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.

    sI = 0
    for spec in species[fI]:
      #[ Load the grid.
      xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % (spec,frame[fI]),polyOrder,basisType,location='center',varName='M0')
      
      #[ Load the electron density, temperatures and flows.
      den   =            np.squeeze(pgu.getInterpData(fName % (spec,frame[fI]), polyOrder, basisType, varName='M0'))
      upar  = (1./c_s)*  np.squeeze(pgu.getInterpData(fName % (spec,frame[fI]), polyOrder, basisType, varName='Upar'))
      tpar  = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % (spec,frame[fI]), polyOrder, basisType, varName='Tpar'))
      #tperp = (1.e-3/eV)*pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Tperp')
      #[ Alternative calculation of T_perp.
      dBdzFile = './'+fileName+'-dBdz_dBdz.bp'    #.Complete file name.
      fsPi = forceSofteningPi_zvparmu(dataDir+fileName,spec,frame[fI],dBdzFile)
      tperp = (1.e-3/eV)*calcTperp(dataDir+fileName,spec,frame[fI],phaseFac=fsPi)
  
  
      hpl21 = list()
      hpl21.append(ax21[sI][0].semilogy(xIntC[0], den, color=mycolors[tI], linestyle=myliness[tI]))
      hpl21.append(ax21[sI][1].plot(xIntC[0], upar,    color=mycolors[tI], linestyle=myliness[tI]))
      hpl21.append(ax21[sI][2].plot(xIntC[0], tpar,    color=mycolors[tI], linestyle=myliness[tI]))
      hpl21.append(ax21[sI][3].plot(xIntC[0], tperp,   color=mycolors[tI], linestyle=myliness[tI]))

      if saveData:
        sStr = '_i'
        if fI == 0:
          sStr = '_i_0'
        if spec == 'elc':
          sStr = '_e'
          h5f.create_dataset('z', np.shape(xIntC[0]), dtype='f8', data=xIntC[0])
        h5f.create_dataset('den'+sStr,   np.shape(den),      dtype='f8', data=den)
        h5f.create_dataset('upar'+sStr,  np.shape(upar),     dtype='f8', data=upar)
        h5f.create_dataset('Tpar'+sStr,  np.shape(tpar),     dtype='f8', data=tpar)
        h5f.create_dataset('Tperp'+sStr, np.shape(tperp),    dtype='f8', data=tperp)

      sI = sI+1
      tI = tI+1
  
  sI = 0
  for spec in ["ion","elc"]:
#    ax21[sI][0].set_ylim(0., n0*1.075)
    for i in range(len(ax21[sI])):
      ax21[sI][i].set_xlim( xIntC[0][0], xIntC[0][-1] )
      setTickFontSize(ax21[sI][i],tickFontSize) 
      ax21[sI][i].tick_params(axis='y', labelcolor=defaultColors[sI], labelsize=tickFontSize)
      hmag = ax21[sI][i].yaxis.get_offset_text().set_size(tickFontSize)
    for i in range(3):
      plt.setp( ax21[sI][i].get_xticklabels(), visible=False)
    ax21[sI][0].set_ylabel(r'$n_%s$ (m$^{-3}$)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI], labelpad=-4+4*sI)
    ax21[sI][1].set_ylabel(r'$u_{\parallel %s}/c_{se0}$' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI], labelpad=-2)
    ax21[sI][2].set_ylabel(r'$T_{\parallel %s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI])
    ax21[sI][3].set_ylabel(r'$T_{\perp %s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI])
    sI = sI+1

  for i in range(len(ax21[1])):
    plot_verticalLinesPM(z_m, ax21[1][i])  #[ Indicate location of max B.
  ax21[0][3].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  plt.text(0.025, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax21[0][0].transAxes)
  plt.text(0.025, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax21[0][1].transAxes)
  plt.text(0.025, 0.85, r'(c)', fontsize=textFontSize, color='black', transform=ax21[0][2].transAxes)
  plt.text(0.025, 0.85, r'(d)', fontsize=textFontSize, color='black', transform=ax21[0][3].transAxes)
  
  if outFigureFile:
    plt.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f.close()

#................................................................................#

if plot_momKinElc:
  #[ Plot density, temperature and u_parallel along the field line for the kinetic elc sim.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk55/'
  
  fileName = 'gk55-wham1x2v'    #.Root name of files to process.
  frame    = 128
  figName  = fileName+'_M0TempUpar_'+str(frame)

  #[ Prepare figure.
  figProp16 = (7.,8.5)
  ax16Pos   = [[0.11, 0.750, 0.77, 0.215],
               [0.11, 0.525, 0.77, 0.215],
               [0.11, 0.300, 0.77, 0.215],
               [0.11, 0.075, 0.77, 0.215]]
  fig16     = plt.figure(figsize=figProp16)
  ax16      = [[],[]]
  ax16[0]   = [fig16.add_axes(d) for d in ax16Pos]
  ax16[1]   = [ax.twinx() for ax in ax16[0]]

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  fName = dataDir+fileName+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.

  c_s = np.sqrt(Te0/mi)

  sI = 0
  for spec in ["ion","elc"]:
    #[ Load the grid.
    xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % (spec,frame),polyOrder,basisType,location='center',varName='M0')
    
    #[ Load the electron density, temperatures and flows.
    den   =            np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='M0'))
    upar  = (1./c_s)*  np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Upar'))
    tpar  = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Tpar'))
    tperp = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Tperp'))


    hpl16 = list()
    hpl16.append(ax16[sI][0].semilogy(xIntC[0], den, color=defaultColors[sI], linestyle=lineStyles[sI]))
    hpl16.append(ax16[sI][1].plot(xIntC[0], upar,    color=defaultColors[sI], linestyle=lineStyles[sI]))
    hpl16.append(ax16[sI][2].plot(xIntC[0], tpar,    color=defaultColors[sI], linestyle=lineStyles[sI]))
    hpl16.append(ax16[sI][3].plot(xIntC[0], tperp,   color=defaultColors[sI], linestyle=lineStyles[sI]))

    if saveData:
      sStr = '_i'
      if spec == 'elc':
        sStr = '_e'
        h5f.create_dataset('z', np.shape(xIntC[0]), dtype='f8', data=xIntC[0])
      h5f.create_dataset('den'+sStr,   np.shape(den),      dtype='f8', data=den)
      h5f.create_dataset('upar'+sStr,  np.shape(upar),     dtype='f8', data=upar)
      h5f.create_dataset('Tpar'+sStr,  np.shape(tpar),     dtype='f8', data=tpar)
      h5f.create_dataset('Tperp'+sStr, np.shape(tperp),    dtype='f8', data=tperp)

    ax16[sI][0].set_ylim(0., n0)
    for i in range(len(ax16[sI])):
      ax16[sI][i].set_xlim( xIntC[0][0], xIntC[0][-1] )
      setTickFontSize(ax16[sI][i],tickFontSize) 
      ax16[sI][i].tick_params(axis='y', labelcolor=defaultColors[sI], labelsize=tickFontSize)
      hmag = ax16[sI][i].yaxis.get_offset_text().set_size(tickFontSize)
    for i in range(3):
      plt.setp( ax16[sI][i].get_xticklabels(), visible=False)
    ax16[sI][0].set_ylabel(r'$n_%s$ (m$^{-3}$)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI], labelpad=-4+4*sI)
    ax16[sI][1].set_ylabel(r'$u_{\parallel %s}/c_{se0}$' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI], labelpad=-2)
    ax16[sI][2].set_ylabel(r'$T_{\parallel %s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI])
    ax16[sI][3].set_ylabel(r'$T_{\perp %s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI])
    sI = sI+1

  for i in range(len(ax16[0])):
    plot_verticalLinesPM(z_m, ax16[0][i])  #[ Indicate location of max B.
  ax16[0][3].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  plt.text(0.025, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax16[0][0].transAxes)
  plt.text(0.025, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax16[0][1].transAxes)
  plt.text(0.025, 0.85, r'(c)', fontsize=textFontSize, color='black', transform=ax16[0][2].transAxes)
  plt.text(0.025, 0.85, r'(d)', fontsize=textFontSize, color='black', transform=ax16[0][3].transAxes)
  
  if outFigureFile:
    plt.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f.close()

#................................................................................#

if plot_f_vspace_kinetic:
  #[ Plot slices of the distribution functions at z=0, z=z_m and z=zMax.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk55/'
  fileName = 'gk55-wham1x2v'    #.Root name of files to process.
  frame    = 128

  fName = dataDir+fileName+'_%s_%d.bp'    #.Complete file name.

  #[ Load the grid.
  xInt_e, _, nxInt_e, lxInt_e, dxInt_e, _ = pgu.getGrid(fName % ('elc',frame),polyOrder,basisType)
  xIntC_e, _, nxIntC_e, lxIntC_e, dxIntC_e, _ = pgu.getGrid(fName % ('elc',frame),polyOrder,basisType,location='center')
  xInt_i, _, nxInt_i, lxInt_i, dxInt_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType)
  xIntC_i, _, nxIntC_i, lxIntC_i, dxIntC_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType,location='center')

  #[ Get indices along z of slices we wish to plot:
  plotz = [0., z_m, xInt_e[0][-1]]
  plotzIdx = [np.argmin(np.abs(xIntC_e[0]-val)) for val in plotz]
  
  #[ Load the magnetic field and electrostatic potential.
  dataFile = dataDir + fileName + '_allGeo_0.bp'
  bmag = np.squeeze(pgu.getInterpData(dataFile,polyOrder,basisType,location='center',varName='bmag'))
  dataFile = dataDir + fileName + '_phi_'+str(frame)+'.bp'
  phi  = np.squeeze(pgu.getInterpData(dataFile,polyOrder,basisType,location='center'))

  #.Compute the trapped-passing boundary in vpar-mu space (normalized).
  B_p  = np.interp(0., xIntC_i[0], bmag)
  R_m  = np.interp(z_m, xIntC_i[0], bmag)/B_p
  Dphi = np.interp(0., xIntC_i[0], phi)-np.interp(z_m, xIntC_i[0], phi)
  muBound_i = np.zeros(np.shape(xIntC_i[1]))
  muBound_e = np.zeros(np.shape(xIntC_i[1]))
  for i in range(nxIntC_i[1]):
    muBound_i[i] = ( (1./B_p)*(0.5*mi*xIntC_i[1][i]**2+qi*Dphi)/(R_m-1) )/(0.5*mi*(vti**2)/B_p)
    muBound_e[i] = ( (1./B_p)*(0.5*me*xIntC_e[1][i]**2+qe*Dphi)/(R_m-1) )/(0.5*me*(vte**2)/B_p)

  #[ Normalize velocity space
  xInt_i[1]  = xInt_i[1]/vti;   xInt_i[2]  = xInt_i[2]/(0.5*mi*(vti**2)/B_p)
  xIntC_i[1] = xIntC_i[1]/vti;  xIntC_i[2] = xIntC_i[2]/(0.5*mi*(vti**2)/B_p)
  xInt_e[1]  = xInt_e[1]/vte;   xInt_e[2]  = xInt_e[2]/(0.5*me*(vte**2)/B_p)
  xIntC_e[1] = xIntC_e[1]/vte;  xIntC_e[2] = xIntC_e[2]/(0.5*me*(vte**2)/B_p)

  #[ Load the electron and ion distribution functions.
  fElc  = np.squeeze(pgu.getInterpData(fName % ('elc',frame), polyOrder, basisType))
  fIon  = np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType))

  #[ Prepare figure.
  figProp17 = (10.4,6.8)
  ax17Pos   = [[0.06-0.012, 0.545, 0.305, 0.384],[0.375-0.012, 0.545, 0.305, 0.384],[0.69-0.012, 0.545, 0.305, 0.384],
               [0.06-0.012, 0.08 , 0.305, 0.384],[0.375-0.012, 0.08 , 0.305, 0.384],[0.69-0.012, 0.08 , 0.305, 0.384],]
  ca17Pos   = [[0.07-0.022, 0.935, 0.285, 0.02 ],[0.385-0.012, 0.935, 0.285, 0.02 ],[0.70-0.02, 0.935, 0.285, 0.02 ],
               [0.07-0.022, 0.471, 0.285, 0.02 ],[0.385-0.012, 0.471, 0.285, 0.02 ],[0.70-0.02, 0.471, 0.285, 0.02 ],]
  fig17     = plt.figure(figsize=figProp17)
  ax17      = [fig17.add_axes(pos) for pos in ax17Pos]
  cbar_ax17 = [fig17.add_axes(pos) for pos in ca17Pos]
  
  #[ Create colorplot grid. Recall coordinates have to be nodal.
  Xnodal_e = [np.outer(xInt_e[1], np.ones(nxInt_e[2])),
              np.outer(np.ones(nxInt_e[1]), xInt_e[2])]
  Xnodal_i = [np.outer(xInt_i[1], np.ones(nxInt_i[2])),
              np.outer(np.ones(nxInt_i[1]), xInt_i[2])]

  cb17a_ticklabels = [['0','1.99','3.99','5.98','7.97'],
                      ['0','0.39','0.78','1.17','1.56'],
                      ['0','0.18','0.36','0.53','0.71'],  ]
  cb17b_ticklabels = [['0','3.24e-4','6.48e-4', '9.72e-4'],
                      ['0','1.44e-5','2.88e-5', '4.32e-5'],
                      ['0','1.61e-6','3.21e-6', '4.8e-6'],  ]
  hpl17a, hpl17b = list(), list()
  hcb17a, hcb17b = list(), list()
  c = 0
  for i in plotzIdx:
    hpl17a.append(ax17[0+c].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon[i,:,:], cmap='inferno'))
    hpl17b.append(ax17[3+c].pcolormesh(Xnodal_e[0], Xnodal_e[1], fElc[i,:,:], cmap='inferno'))

    extreme_vals = [0., np.amax(fIon[i,:,:])]
    hcb17a.append(plt.colorbar(hpl17a[c], ax=ax17[0+c], cax=cbar_ax17[0+c], orientation='horizontal', format='%0.2f'))
    hcb17a[c].set_ticks(np.linspace(extreme_vals[0], extreme_vals[1], 5))
    hcb17a[c].set_ticklabels(cb17a_ticklabels[c])
    hcb17a[c].ax.xaxis.set_ticks_position('top')
    hcb17a[c].ax.tick_params(labelsize=tickFontSize)

    extreme_vals = [0., np.amax(fElc[i,:,:])]
    hcb17b.append(plt.colorbar(hpl17b[c], ax=ax17[3+c], cax=cbar_ax17[3+c], orientation='horizontal', format='%3.2e'))
    hcb17b[c].set_ticks(np.linspace(extreme_vals[0], extreme_vals[1], 4))
    hcb17b[c].set_ticklabels(cb17b_ticklabels[c])
    hcb17b[c].ax.xaxis.set_ticks_position('top')
    hcb17b[c].ax.tick_params(labelsize=tickFontSize)
    c = c+1

  #[ Plot insets:
  inset_size = [0.5, 0.4, 0.48, 0.57]
  ax17in = [0 for i in range(len(ax17))]
  hpl17in = [0 for i in range(len(ax17))]

  sub_coords = [[0., 2.], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[0],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax17in[0], hpl17in[0] = plot_inset_pcolormesh(ax17[0], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[0],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)
  #[ Plot trapped-passing boundary.
  ax17[0].plot(xIntC_i[1], muBound_i, ':', color='white',alpha=0.5,linewidth=3)
  ax17in[0].plot(xIntC_i[1], muBound_i, ':', color='white',alpha=0.5,linewidth=3)
  ax17[0].set_ylim(xInt_i[2][0], xInt_i[2][-1])

#  sub_coords = [[-1., 1.], [0., 1.]]
  sub_coords = [[-0.5, 1.5], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[1],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax17in[1], hpl17in[1] = plot_inset_pcolormesh(ax17[1], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[1],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)
#  sub_coords = [[.5, 2.5], [0., 1.]]
  sub_coords = [[.75, 3.75], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[2],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax17in[2], hpl17in[2] = plot_inset_pcolormesh(ax17[2], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[2],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)

  sub_coords = [[0., 2.], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_e[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_e[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fElc[plotzIdx[0],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax17in[3], hpl17in[3] = plot_inset_pcolormesh(ax17[3], Xnodal_e[0], Xnodal_e[1], fElc[plotzIdx[0],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)
  #[ Plot trapped-passing boundary.
  ax17[3].plot(xIntC_e[1], muBound_e, ':', color='white',alpha=0.5,linewidth=3)
  ax17in[3].plot(xIntC_e[1], muBound_e, ':', color='white',alpha=0.5,linewidth=3)
  ax17[3].set_ylim(xInt_e[2][0], xInt_e[2][-1])

  sub_coords = [[-1.5, 1.5], [0., 0.5]]
  insetIdx = [[np.argmin(np.abs(xIntC_e[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_e[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fElc[plotzIdx[1],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax17in[4], hpl17in[4] = plot_inset_pcolormesh(ax17[4], Xnodal_e[0], Xnodal_e[1], fElc[plotzIdx[1],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)
  sub_coords = [[-.65, 1.65], [0., 1.2]]
  insetIdx = [[np.argmin(np.abs(xIntC_e[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_e[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fElc[plotzIdx[2],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax17in[5], hpl17in[5] = plot_inset_pcolormesh(ax17[5], Xnodal_e[0], Xnodal_e[1], fElc[plotzIdx[2],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)

  for i in range(3):
    plt.setp( ax17[i].get_xticklabels(), visible=False)
    nohmagx = ax17[i].xaxis.get_offset_text().set_size(0)
    ax17[3+i].set_xlabel(r'$v_\parallel/v_{ts0}$', fontsize=xyLabelFontSize, labelpad=-2)
    setTickFontSize(ax17[0+i],tickFontSize) 
    setTickFontSize(ax17[3+i],tickFontSize) 
    hmagx = ax17[3+i].xaxis.get_offset_text().set_size(tickFontSize)
  for i in range(1,3):
    plt.setp( ax17[0+i].get_yticklabels(), visible=False)
    plt.setp( ax17[3+i].get_yticklabels(), visible=False)
    nohmagy = ax17[0+i].yaxis.get_offset_text().set_size(0)
    nohmagy = ax17[3+i].yaxis.get_offset_text().set_size(0)
    hmagy = ax17[(i-1)*3].yaxis.get_offset_text().set_size(tickFontSize)
  ax17[0].set_ylabel(r'$\mu/[m_iv_{ti0}^2/(2B_p)]$', fontsize=xyLabelFontSize, labelpad=-4)
  ax17[3].set_ylabel(r'$\mu/[m_ev_{te0}^2/(2B_p)]$', fontsize=xyLabelFontSize, labelpad=-4)
  plt.text(0.03, 0.85, r'(a) $f_i(z=0)$',     fontsize=textFontSize, color='white', transform=ax17[0].transAxes)
  plt.text(0.03, 0.85, r'(b) $f_i(z=z_m)$',   fontsize=textFontSize, color='white', transform=ax17[1].transAxes)
  plt.text(0.03, 0.85, r'(c) $f_i(z=L_z/2)$', fontsize=textFontSize, color='white', transform=ax17[2].transAxes)
  plt.text(0.03, 0.85, r'(d) $f_e(z=0)$',     fontsize=textFontSize, color='white', transform=ax17[3].transAxes)
  plt.text(0.03, 0.85, r'(e) $f_e(z=z_m)$',   fontsize=textFontSize, color='white', transform=ax17[4].transAxes)
  plt.text(0.03, 0.85, r'(f) $f_e(z=L_z/2)$', fontsize=textFontSize, color='white', transform=ax17[5].transAxes)
  
  figName = fileName+'_fCenterThroatWall_'+str(frame)
  if outFigureFile:
    fig17.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")
    h5f.create_dataset('vpar_i',     np.shape(Xnodal_i[0]),           dtype='f8', data=Xnodal_i[0])
    h5f.create_dataset('mu_i',       np.shape(Xnodal_i[1]),           dtype='f8', data=Xnodal_i[1])
    h5f.create_dataset('fi_zeq0',    np.shape(fIon[plotzIdx[0],:,:]), dtype='f8', data=fIon[plotzIdx[0],:,:])
    h5f.create_dataset('fi_zeqzm',   np.shape(fIon[plotzIdx[1],:,:]), dtype='f8', data=fIon[plotzIdx[1],:,:])
    h5f.create_dataset('fi_zeqLzD2', np.shape(fIon[plotzIdx[2],:,:]), dtype='f8', data=fIon[plotzIdx[2],:,:])
    h5f.create_dataset('vpar_e',     np.shape(Xnodal_e[0]),           dtype='f8', data=Xnodal_e[0])
    h5f.create_dataset('mu_e',       np.shape(Xnodal_e[1]),           dtype='f8', data=Xnodal_e[1])
    h5f.create_dataset('fe_zeq0',    np.shape(fElc[plotzIdx[0],:,:]), dtype='f8', data=fElc[plotzIdx[0],:,:])
    h5f.create_dataset('fe_zeqzm',   np.shape(fElc[plotzIdx[1],:,:]), dtype='f8', data=fElc[plotzIdx[1],:,:])
    h5f.create_dataset('fe_zeqLzD2', np.shape(fElc[plotzIdx[2],:,:]), dtype='f8', data=fElc[plotzIdx[2],:,:])
    h5f.close()
  
#................................................................................#

if plot_phiKinetic:
  #[ Plot phi(z) and e*Phi/Te0 for a single frame, or phi(z,t) and phi(z=0,t).

  dataDir = ['/scratch/gpfs/manaurer/gkeyll/mirror/gk55/',  #[ Kinetic electrons.
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk57/',] #[ Adiaabtic electrons.
  
  #.Root name of files to process.
  fileName = ['gk55-wham1x2v',  #[ Kinetic electrons.
              'gk57-wham1x2v',] #[ Adiaabtic electrons. 
  frame     = 132
  frameProf = 128

  #[ Prepare figure.
  figProp18 = (6.4,5.)
  ax18Pos   = [[0.12, 0.605, 0.725, 0.39],
              [0.12, 0.105, 0.725, 0.39]]
  cax18Pos  = [[0.855, 0.605, 0.02, 0.39]]
  fig18     = plt.figure(figsize=figProp18)
  ax18      = [fig18.add_axes(pos) for pos in ax18Pos]
  cbar_ax18 = [fig18.add_axes(pos) for pos in cax18Pos]
  
  phiFile = dataDir[0]+fileName[0]+'_phi_'    #.Complete file name.
  nFrames = 1+frame
  times = 1.e6*pgu.getTimeStamps(phiFile,0,frame)
  
  phiFile = dataDir[0]+fileName[0]+'_phi_%d.bp'    #.Complete file name.
  
  #[ Load the grid.
  xInt, _, nxInt, lxInt, dxInt, _ = pgu.getGrid(phiFile % 0,polyOrder,basisType)
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(phiFile % 0,polyOrder,basisType,location='center')
  
  phi = np.zeros((nFrames,nxIntC[0]))
  
  for fI in range(nFrames):
    phi[fI,:] = np.squeeze(pgu.getInterpData(phiFile % fI, polyOrder, basisType))
  
  #[ Create colorplot grid (coordinates have to be nodal).
  timesC  = 0.5*(times[1:]+times[0:-1])
  Xnodal = [np.outer(np.concatenate([[timesC[0]-(timesC[1]-timesC[0])],timesC,[timesC[-1]+(timesC[-1]-timesC[-2])]]), \
                     np.ones(np.shape(xInt[0]))), \
            np.outer(np.ones(np.size(timesC)+2),xInt[0])]
  
  ePhiDTe = eV*phi/Te0
  hpl18a = ax18[0].pcolormesh(Xnodal[0], Xnodal[1], ePhiDTe, cmap='inferno')
  hcb18a = plt.colorbar(hpl18a, ax=ax18[0], cax=cbar_ax18[0], ticks=[0., 1.2, 2.4, 3.6])
  #[ Plots lines at mirror throat:
  ax18[0].plot([Xnodal[0][0][0], Xnodal[0][-1][0]],   [z_m, z_m], color='white', linestyle='--', alpha=0.5)
  ax18[0].plot([Xnodal[0][0][0], Xnodal[0][-1][0]], [-z_m, -z_m], color='white', linestyle='--', alpha=0.5)

  #[ Plot the profile in frame frameProf.
  ax18[0].plot([times[frameProf],times[frameProf]], [xInt[0][0], xInt[0][-1]], color='white', linestyle=':', alpha=0.5)
  hpl18b = [ax18[1].plot(xIntC[0], ePhiDTe[frameProf,:], color=defaultColors[0], linestyle=lineStyles[0])]
  #[ Load and plot phi from adiabatic electron run.
  phiFile = dataDir[1]+fileName[1]+'_phi_%d.bp'    #.Complete file name.
  ePhiDTeAdi = (eV/Te0)*np.squeeze(pgu.getInterpData(phiFile % frameProf, polyOrder, basisType))
  hpl18b.append(ax18[1].plot(xIntC[0], ePhiDTeAdi, color=defaultColors[1], linestyle=lineStyles[1]))

  plot_verticalLinesPM(z_m, ax18[1])  #[ Indicate location of max B.
  hcb18a.set_label('$e\phi/T_{e0}$', rotation=270, labelpad=18, fontsize=colorBarLabelFontSize)
  hcb18a.ax.tick_params(labelsize=tickFontSize)
  ax18[0].set_ylabel(r'$z$ (m)', fontsize=xyLabelFontSize)
  ax18[0].set_xlabel(r'Time ($\mu$s)', fontsize=xyLabelFontSize, labelpad=-2)
  ax18[1].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize, labelpad=-2)
  ax18[1].set_ylabel(r'$e\phi/T_{e0}$', fontsize=xyLabelFontSize)
  plt.text(0.025, 0.85, r'(a)', fontsize=textFontSize, color='white', transform=ax18[0].transAxes)
  plt.text(0.025, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax18[1].transAxes)
  ax18[1].legend([r'kinetic e$^-$',r'Boltzmann e$^-$'], fontsize=legendFontSize, frameon=False, loc='center')
  con = ConnectionPatch(xyA=(times[frameProf], xInt[0][0]*1.05), coordsA=ax18[0].transData,
                        xyB=(xIntC[0][-1]*0.8, 11.), coordsB=ax18[1].transData, arrowstyle='->',connectionstyle='angle3')
  fig18.add_artist(con)
  ax18[0].set_xlim( times[0], times[-1] )
  ax18[1].set_xlim( xIntC[0][0], xIntC[0][-1] )
  ax18[1].set_ylim( 0., 12. )
  for i in range(len(ax18)):
    setTickFontSize(ax18[i],tickFontSize) 
  
  figName = fileName[0]+'_phiVtime_0-'+str(frame)
  if outFigureFile:
    fig18.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")
    h5f.create_dataset('ePhiDTe', np.shape(ePhiDTe), dtype='f8', data=ePhiDTe)
    h5f.create_dataset('ePhiDTe_z', np.shape(Xnodal[0]), dtype='f8', data=Xnodal[0])
    h5f.create_dataset('ePhiDTe_time', np.shape(Xnodal[1]), dtype='f8', data=Xnodal[1])
    h5f.create_dataset('z', np.shape(xIntC[0]), dtype='f8', data=xIntC[0])
    h5f.create_dataset('ePhiDTe_teq32mus', np.shape(ePhiDTe[frameProf,:]), dtype='f8', data=ePhiDTe[frameProf,:])
    h5f.create_dataset('ePhiDTe_boltz_teq32mus', np.shape(ePhiDTeAdi), dtype='f8', data=ePhiDTeAdi)
    h5f.close()
  
#................................................................................#

if plot_f_vspace_nonlinPol:
  #[ Plot slices of the distribution functions at z=0, z=z_m and z=zMax
  #[ in kinetic electron simulation with nonlinear polarization.

  dataDir  = '/scratch/gpfs/manaurer/gkeyll/mirror/gk75wExBenergy_vparRes2x/'
  fileName = 'gk75-wham1x2v'    #.Root name of files to process.
  frame    = 128

  fName = dataDir+fileName+'_%s_%d.bp'    #.Complete file name.

  #[ Load the grid.
  xInt_e, _, nxInt_e, lxInt_e, dxInt_e, _ = pgu.getGrid(fName % ('elc',frame),polyOrder,basisType)
  xIntC_e, _, nxIntC_e, lxIntC_e, dxIntC_e, _ = pgu.getGrid(fName % ('elc',frame),polyOrder,basisType,location='center')
  xInt_i, _, nxInt_i, lxInt_i, dxInt_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType)
  xIntC_i, _, nxIntC_i, lxIntC_i, dxIntC_i, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType,location='center')

  #[ Get indices along z of slices we wish to plot:
  plotz = [0., z_m, xInt_e[0][-1]]
  plotzIdx = [np.argmin(np.abs(xIntC_e[0]-val)) for val in plotz]
  
  #[ Load the magnetic field and electrostatic potential.
  dataFile = dataDir + fileName + '_allGeo_0.bp'
  bmag = np.squeeze(pgu.getInterpData(dataFile,polyOrder,basisType,location='center',varName='bmag'))
  dataFile = dataDir + fileName + '_phi_'+str(frame)+'.bp'
  phi  = np.squeeze(pgu.getInterpData(dataFile,polyOrder,basisType,location='center'))

  #.Compute the trapped-passing boundary in vpar-mu space (normalized).
  B_p  = np.interp(0., xIntC_i[0], bmag)
  R_m  = np.interp(z_m, xIntC_i[0], bmag)/B_p
  Dphi = np.interp(0., xIntC_i[0], phi)-np.interp(z_m, xIntC_i[0], phi)
  muBound_i = np.zeros(np.shape(xIntC_i[1]))
  muBound_e = np.zeros(np.shape(xIntC_i[1]))
  for i in range(nxIntC_i[1]):
    muBound_i[i] = ( (1./B_p)*(0.5*mi*xIntC_i[1][i]**2+qi*Dphi)/(R_m-1) )/(0.5*mi*(vti**2)/B_p)
    muBound_e[i] = ( (1./B_p)*(0.5*me*xIntC_e[1][i]**2+qe*Dphi)/(R_m-1) )/(0.5*me*(vte**2)/B_p)

  #[ Normalize velocity space
  xInt_i[1]  = xInt_i[1]/vti;   xInt_i[2]  = xInt_i[2]/(0.5*mi*(vti**2)/B_p)
  xIntC_i[1] = xIntC_i[1]/vti;  xIntC_i[2] = xIntC_i[2]/(0.5*mi*(vti**2)/B_p)
  xInt_e[1]  = xInt_e[1]/vte;   xInt_e[2]  = xInt_e[2]/(0.5*me*(vte**2)/B_p)
  xIntC_e[1] = xIntC_e[1]/vte;  xIntC_e[2] = xIntC_e[2]/(0.5*me*(vte**2)/B_p)

  #[ Load the electron and ion distribution functions.
  fElc  = np.squeeze(pgu.getInterpData(fName % ('elc',frame), polyOrder, basisType))
  fIon  = np.squeeze(pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType))

  #[ Prepare figure.
  figProp19 = (10.4,6.8)
  ax19Pos   = [[0.06-0.012, 0.545, 0.305, 0.384],[0.375-0.012, 0.545, 0.305, 0.384],[0.69-0.012, 0.545, 0.305, 0.384],
               [0.06-0.012, 0.08 , 0.305, 0.384],[0.375-0.012, 0.08 , 0.305, 0.384],[0.69-0.012, 0.08 , 0.305, 0.384],]
  ca19Pos   = [[0.07-0.022, 0.935, 0.285, 0.02 ],[0.385-0.012, 0.935, 0.282, 0.02 ],[0.70-0.02, 0.935, 0.285, 0.02 ],
               [0.07-0.022, 0.471, 0.285, 0.02 ],[0.385-0.012, 0.471, 0.282, 0.02 ],[0.70-0.02, 0.471, 0.285, 0.02 ],]
  fig19     = plt.figure(figsize=figProp19)
  ax19      = [fig19.add_axes(pos) for pos in ax19Pos]
  cbar_ax19 = [fig19.add_axes(pos) for pos in ca19Pos]
  
  #[ Create colorplot grid. Recall coordinates have to be nodal.
  Xnodal_e = [np.outer(xInt_e[1], np.ones(nxInt_e[2])),
              np.outer(np.ones(nxInt_e[1]), xInt_e[2])]
  Xnodal_i = [np.outer(xInt_i[1], np.ones(nxInt_i[2])),
              np.outer(np.ones(nxInt_i[1]), xInt_i[2])]

  cb19a_ticklabels = [['0','2','4','6','8'],
                      ['0','0.64','1.28','1.92','2.56'],
                      ['0','0.43','0.85','1.28','1.71'],  ]
  cb19b_ticklabels = [['0','2.72e-4','5.44e-4','8.16e-4'],
                      ['0','9.45e-6','1.89e-5','2.8e-5'],
                      ['0','1.62e-6','3.23e-6','4.85e-6'],  ]

  hpl19a, hpl19b = list(), list()
  hcb19a, hcb19b = list(), list()
  c = 0
  for i in plotzIdx:
    hpl19a.append(ax19[0+c].pcolormesh(Xnodal_i[0], Xnodal_i[1], fIon[i,:,:], cmap='inferno'))
    hpl19b.append(ax19[3+c].pcolormesh(Xnodal_e[0], Xnodal_e[1], fElc[i,:,:], cmap='inferno'))

    extreme_vals = [0., np.amax(fIon[i,:,:])]
    hcb19a.append(plt.colorbar(hpl19a[c], ax=ax19[0+c], cax=cbar_ax19[0+c], orientation='horizontal', format='%0.2f'))
    hcb19a[c].set_ticks(np.linspace(extreme_vals[0], extreme_vals[1], 5))
    hcb19a[c].set_ticklabels(cb19a_ticklabels[c])
    hcb19a[c].ax.xaxis.set_ticks_position('top')
    hcb19a[c].ax.tick_params(labelsize=tickFontSize)

    extreme_vals = [0., np.amax(fElc[i,:,:])]
    hcb19b.append(plt.colorbar(hpl19b[c], ax=ax19[3+c], cax=cbar_ax19[3+c], orientation='horizontal', format='%3.2e'))
    hcb19b[c].set_ticks(np.linspace(extreme_vals[0], extreme_vals[1], 4))
    hcb19b[c].set_ticklabels(cb19b_ticklabels[c])
    hcb19b[c].ax.xaxis.set_ticks_position('top')
    hcb19b[c].ax.tick_params(labelsize=tickFontSize)
    c = c+1

  #[ Plot insets:
  inset_size = [0.5, 0.4, 0.48, 0.57]
  ax19in = [0 for i in range(len(ax19))]
  hpl19in = [0 for i in range(len(ax19))]

  sub_coords = [[0., 2.], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[0],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax19in[0], hpl19in[0] = plot_inset_pcolormesh(ax19[0], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[0],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)
  #[ Plot trapped-passing boundary.
  ax19[0].plot(xIntC_i[1], muBound_i, ':', color='white',alpha=0.5,linewidth=3)
  ax19in[0].plot(xIntC_i[1], muBound_i, ':', color='white',alpha=0.5,linewidth=3)
  ax19[0].set_ylim(xInt_i[2][0], xInt_i[2][-1])

  sub_coords = [[-0.5, 1.5], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[1],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax19in[1], hpl19in[1] = plot_inset_pcolormesh(ax19[1], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[1],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)
  sub_coords = [[.75, 3.75], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_i[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_i[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fIon[plotzIdx[2],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax19in[2], hpl19in[2] = plot_inset_pcolormesh(ax19[2], Xnodal_i[0], Xnodal_i[1], fIon[plotzIdx[2],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)

  sub_coords = [[0., 2.], [0., 1.]]
  insetIdx = [[np.argmin(np.abs(xIntC_e[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_e[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fElc[plotzIdx[0],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax19in[3], hpl19in[3] = plot_inset_pcolormesh(ax19[3], Xnodal_e[0], Xnodal_e[1], fElc[plotzIdx[0],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)
  #[ Plot trapped-passing boundary.
  ax19[3].plot(xIntC_e[1], muBound_e, ':', color='white',alpha=0.5,linewidth=3)
  ax19in[3].plot(xIntC_e[1], muBound_e, ':', color='white',alpha=0.5,linewidth=3)
  ax19[3].set_ylim(xInt_e[2][0], xInt_e[2][-1])

  sub_coords = [[-1.5, 1.5], [0., 0.5]]
  insetIdx = [[np.argmin(np.abs(xIntC_e[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_e[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fElc[plotzIdx[1],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax19in[4], hpl19in[4] = plot_inset_pcolormesh(ax19[4], Xnodal_e[0], Xnodal_e[1], fElc[plotzIdx[1],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)
  sub_coords = [[-.65, 1.65], [0., 1.2]]
  insetIdx = [[np.argmin(np.abs(xIntC_e[1]-val)) for val in sub_coords[0]],
              [np.argmin(np.abs(xIntC_e[2]-val)) for val in sub_coords[1]]]
  extreme_vals = [0., np.amax(fElc[plotzIdx[2],insetIdx[0][0]:insetIdx[0][1],insetIdx[1][0]:insetIdx[1][1]])]
  ax19in[5], hpl19in[5] = plot_inset_pcolormesh(ax19[5], Xnodal_e[0], Xnodal_e[1], fElc[plotzIdx[2],:,:], \
                                              inset_size, sub_coords, "white", extreme_vals)

  for i in range(3):
    plt.setp( ax19[i].get_xticklabels(), visible=False)
    nohmagx = ax19[i].xaxis.get_offset_text().set_size(0)
    ax19[3+i].set_xlabel(r'$v_\parallel/v_{ts0}$', fontsize=xyLabelFontSize, labelpad=-2)
    setTickFontSize(ax19[0+i],tickFontSize) 
    setTickFontSize(ax19[3+i],tickFontSize) 
    hmagx = ax19[3+i].xaxis.get_offset_text().set_size(tickFontSize)
  for i in range(1,3):
    plt.setp( ax19[0+i].get_yticklabels(), visible=False)
    plt.setp( ax19[3+i].get_yticklabels(), visible=False)
    nohmagy = ax19[0+i].yaxis.get_offset_text().set_size(0)
    nohmagy = ax19[3+i].yaxis.get_offset_text().set_size(0)
    hmagy = ax19[(i-1)*3].yaxis.get_offset_text().set_size(tickFontSize)
  ax19[0].set_ylabel(r'$\mu/[m_iv_{ti0}^2/(2B_p)]$', fontsize=xyLabelFontSize, labelpad=-4)
  ax19[3].set_ylabel(r'$\mu/[m_ev_{te0}^2/(2B_p)]$', fontsize=xyLabelFontSize, labelpad=-4)
  plt.text(0.03, 0.85, r'(a) $f_i(z=0)$',     fontsize=textFontSize, color='white', transform=ax19[0].transAxes)
  plt.text(0.03, 0.85, r'(b) $f_i(z=z_m)$',   fontsize=textFontSize, color='white', transform=ax19[1].transAxes)
  plt.text(0.03, 0.85, r'(c) $f_i(z=L_z/2)$', fontsize=textFontSize, color='white', transform=ax19[2].transAxes)
  plt.text(0.03, 0.85, r'(d) $f_e(z=0)$',     fontsize=textFontSize, color='white', transform=ax19[3].transAxes)
  plt.text(0.03, 0.85, r'(e) $f_e(z=z_m)$',   fontsize=textFontSize, color='white', transform=ax19[4].transAxes)
  plt.text(0.03, 0.85, r'(f) $f_e(z=L_z/2)$', fontsize=textFontSize, color='white', transform=ax19[5].transAxes)
  
  figName = fileName+'_fCenterThroatWall_'+str(frame)
  if outFigureFile:
    fig19.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")
    h5f.create_dataset('vpar_i',     np.shape(Xnodal_i[0]),           dtype='f8', data=Xnodal_i[0])
    h5f.create_dataset('mu_i',       np.shape(Xnodal_i[1]),           dtype='f8', data=Xnodal_i[1])
    h5f.create_dataset('fi_zeq0',    np.shape(fIon[plotzIdx[0],:,:]), dtype='f8', data=fIon[plotzIdx[0],:,:])
    h5f.create_dataset('fi_zeqzm',   np.shape(fIon[plotzIdx[1],:,:]), dtype='f8', data=fIon[plotzIdx[1],:,:])
    h5f.create_dataset('fi_zeqLzD2', np.shape(fIon[plotzIdx[2],:,:]), dtype='f8', data=fIon[plotzIdx[2],:,:])
    h5f.create_dataset('vpar_e',     np.shape(Xnodal_e[0]),           dtype='f8', data=Xnodal_e[0])
    h5f.create_dataset('mu_e',       np.shape(Xnodal_e[1]),           dtype='f8', data=Xnodal_e[1])
    h5f.create_dataset('fe_zeq0',    np.shape(fElc[plotzIdx[0],:,:]), dtype='f8', data=fElc[plotzIdx[0],:,:])
    h5f.create_dataset('fe_zeqzm',   np.shape(fElc[plotzIdx[1],:,:]), dtype='f8', data=fElc[plotzIdx[1],:,:])
    h5f.create_dataset('fe_zeqLzD2', np.shape(fElc[plotzIdx[2],:,:]), dtype='f8', data=fElc[plotzIdx[2],:,:])
    h5f.close()

#................................................................................#

if plot_phiNonlinPol:
  #[ Plot phi(z) and e*Phi/Te0 for a single frame, or phi(z,t) and phi(z=0,t) for the kinetic electron
  #[ simulation with nonlinear polarization and force softening.

  dataDir = ['/scratch/gpfs/manaurer/gkeyll/mirror/gk55/',  #[ Kinetic electrons.
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk57/', #[ Adiaabtic electrons.
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk75wExBenergy_vparRes2x/',] #[ Adiaabtic electrons.
  
  #.Root name of files to process.
  fileName = ['gk55-wham1x2v',  #[ Linear pol.
              'gk57-wham1x2v',  #[ Adiabatic electrons. 
              'gk75-wham1x2v',] #[ Nonlinear pol.
  frame     = 132
  frameProf = 128

  #[ Prepare figure.
  figProp20 = (6.4,5.)
  ax20Pos   = [[0.12, 0.605, 0.725, 0.39],
              [0.12, 0.105, 0.725, 0.39]]
  cax20Pos  = [[0.855, 0.605, 0.02, 0.39]]
  fig20     = plt.figure(figsize=figProp20)
  ax20      = [fig20.add_axes(pos) for pos in ax20Pos]
  cbar_ax20 = [fig20.add_axes(pos) for pos in cax20Pos]
  
  phiFile = dataDir[2]+fileName[2]+'_phi_'    #.Complete file name.
  nFrames = 1+frame
  times = 1.e6*pgu.getTimeStamps(phiFile,0,frame)
  
  phiFile = dataDir[2]+fileName[2]+'_phi_%d.bp'    #.Complete file name.
  
  #[ Load the grid.
  xInt, _, nxInt, lxInt, dxInt, _ = pgu.getGrid(phiFile % 0,polyOrder,basisType)
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(phiFile % 0,polyOrder,basisType,location='center')
  
  phi = np.zeros((nFrames,nxIntC[0]))
  
  for fI in range(nFrames):
    phi[fI,:] = np.squeeze(pgu.getInterpData(phiFile % fI, polyOrder, basisType))
  
  #[ Create colorplot grid (coordinates have to be nodal).
  timesC  = 0.5*(times[1:]+times[0:-1])
  Xnodal = [np.outer(np.concatenate([[timesC[0]-(timesC[1]-timesC[0])],timesC,[timesC[-1]+(timesC[-1]-timesC[-2])]]), \
                     np.ones(np.shape(xInt[0]))), \
            np.outer(np.ones(np.size(timesC)+2),xInt[0])]
  
  ePhiDTe = eV*phi/Te0
  hpl20a = ax20[0].pcolormesh(Xnodal[0], Xnodal[1], ePhiDTe, cmap='inferno')
  hcb20a = plt.colorbar(hpl20a, ax=ax20[0], cax=cbar_ax20[0], ticks=[0., 1.2, 2.4, 3.6])
  #[ Plots lines at mirror throat:
  ax20[0].plot([Xnodal[0][0][0], Xnodal[0][-1][0]],   [z_m, z_m], color='white', linestyle='--', alpha=0.5)
  ax20[0].plot([Xnodal[0][0][0], Xnodal[0][-1][0]], [-z_m, -z_m], color='white', linestyle='--', alpha=0.5)

  ax20[0].plot([times[frameProf],times[frameProf]], [xInt[0][0], xInt[0][-1]], color='white', linestyle=':', alpha=0.5)
  hpl20b = list()
  #[ Load and plot phi from linear pol run.
  phiFile = dataDir[0]+fileName[0]+'_phi_%d.bp'    #.Complete file name.
  ePhiDTeLinPol  = (eV/Te0)*np.squeeze(pgu.getInterpData(phiFile % frameProf, polyOrder, basisType))
  hpl20b.append(ax20[1].plot(xIntC[0], ePhiDTeLinPol, color=defaultColors[0], linestyle=lineStyles[0]))
  #[ Load and plot phi from adiabatic electron run.
  phiFile = dataDir[1]+fileName[1]+'_phi_%d.bp'    #.Complete file name.
  ePhiDTeAdi = (eV/Te0)*np.squeeze(pgu.getInterpData(phiFile % frameProf, polyOrder, basisType))
  hpl20b.append(ax20[1].plot(xIntC[0], ePhiDTeAdi, color=defaultColors[1], linestyle=lineStyles[1]))
  #[ Plot the profile in frame frameProf.
  hpl20b.append(ax20[1].plot(xIntC[0], ePhiDTe[frameProf,:], color=defaultColors[2], linestyle=lineStyles[2]))

  plot_verticalLinesPM(z_m, ax20[1])  #[ Indicate location of max B.
  hcb20a.set_label('$e\phi/T_{e0}$', rotation=270, labelpad=18, fontsize=colorBarLabelFontSize)
  hcb20a.ax.tick_params(labelsize=tickFontSize)
  ax20[0].set_ylabel(r'$z$ (m)', fontsize=xyLabelFontSize)
  ax20[0].set_xlabel(r'Time ($\mu$s)', fontsize=xyLabelFontSize, labelpad=-2)
  ax20[1].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize, labelpad=-2)
  ax20[1].set_ylabel(r'$e\phi/T_{e0}$', fontsize=xyLabelFontSize)
  plt.text(0.025, 0.85, r'(a)', fontsize=textFontSize, color='white', transform=ax20[0].transAxes)
  plt.text(0.025, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax20[1].transAxes)
  ax20[1].legend([hpl20b[1][0],hpl20b[0][0],hpl20b[2][0],],[r'Boltzmann e$^-$',r'linear pol.',r'nonlinear pol.'], fontsize=legendFontSize, frameon=True, loc=(0.75,0.4), framealpha=1.)
  con = ConnectionPatch(xyA=(times[frameProf], xInt[0][0]*1.05), coordsA=ax20[0].transData,
                        xyB=(xIntC[0][-1]*0.8, 11.), coordsB=ax20[1].transData, arrowstyle='->',connectionstyle='angle3')
  fig20.add_artist(con)
  ax20[0].set_xlim( times[0], times[-1] )
  ax20[1].set_xlim( xIntC[0][0], xIntC[0][-1] )
  ax20[1].set_ylim( 0., 12. )
  for i in range(len(ax20)):
    setTickFontSize(ax20[i],tickFontSize) 
  
  figName = fileName[2]+'_phiVtime_0-'+str(frame)
  if outFigureFile:
    fig20.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")
    h5f.create_dataset('ePhiDTe', np.shape(ePhiDTe), dtype='f8', data=ePhiDTe)
    h5f.create_dataset('ePhiDTe_z', np.shape(Xnodal[0]), dtype='f8', data=Xnodal[0])
    h5f.create_dataset('ePhiDTe_time', np.shape(Xnodal[1]), dtype='f8', data=Xnodal[1])
    h5f.create_dataset('z', np.shape(xIntC[0]), dtype='f8', data=xIntC[0])
    h5f.create_dataset('ePhiDTe_linpol_teq32mus', np.shape(ePhiDTeLinPol), dtype='f8', data=ePhiDTeLinPol)
    h5f.create_dataset('ePhiDTe_nonlinpol_teq32mus', np.shape(ePhiDTe[frameProf,:]), dtype='f8', data=ePhiDTe[frameProf,:])
    h5f.create_dataset('ePhiDTe_boltz_teq32mus', np.shape(ePhiDTeAdi), dtype='f8', data=ePhiDTeAdi)
    h5f.close()
  
#................................................................................#

if plot_momKinElc_nonlinPol:
  #[ Plot density, temperature and u_parallel along the field
  #[ line for the kinetic elc sim with a nonlinear polarization.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk75wExBenergy_vparRes2x/'
  
  fileName = 'gk75-wham1x2v'    #.Root name of files to process.
  frame    = 128
  figName = fileName+'_M0TempUpar_'+str(frame)

  #[ Prepare figure.
  figProp21 = (7.,8.5)
  ax21Pos   = [[0.11, 0.750, 0.77, 0.215],
               [0.11, 0.525, 0.77, 0.215],
               [0.11, 0.300, 0.77, 0.215],
               [0.11, 0.075, 0.77, 0.215]]
  fig21     = plt.figure(figsize=figProp21)
  ax21      = [[],[]]
  ax21[0]   = [fig21.add_axes(d) for d in ax21Pos]
  ax21[1]   = [ax.twinx() for ax in ax21[0]]

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  fName = dataDir+fileName+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.

  c_s = np.sqrt(Te0/mi)

  sI = 0
  for spec in ["ion","elc"]:
    #[ Load the grid.
    xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % (spec,frame),polyOrder,basisType,location='center',varName='M0')
    
    #[ Load the electron density, temperatures and flows.
    den   =            np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='M0'))
    upar  = (1./c_s)*  np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Upar'))
    temp  = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Temp'))
    tpar  = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Tpar'))
    #tperp = (1.e-3/eV)*pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Tperp')
    #[ Alternative calculation of T_perp.
    dBdzFile = './'+fileName+'-dBdz_dBdz.bp'    #.Complete file name.
    fsPi  = forceSofteningPi_zvparmu(dataDir+fileName,spec,frame,dBdzFile)
    tperp = (1.e-3/eV)*calcTperp(dataDir+fileName,spec,frame,phaseFac=fsPi)

    hpl21 = list()
    hpl21.append(ax21[sI][0].semilogy(xIntC[0], den,      color=defaultColors[sI], linestyle=lineStyles[sI]))
    hpl21.append(ax21[sI][1].plot(xIntC[0], upar, color=defaultColors[sI], linestyle=lineStyles[sI]))
    hpl21.append(ax21[sI][2].plot(xIntC[0], tpar,     color=defaultColors[sI], linestyle=lineStyles[sI]))
    hpl21.append(ax21[sI][3].plot(xIntC[0], tperp,    color=defaultColors[sI], linestyle=lineStyles[sI]))

    if saveData:
      sStr = '_i'
      if spec == 'elc':
        sStr = '_e'
        h5f.create_dataset('z', np.shape(xIntC[0]), dtype='f8', data=xIntC[0])
      h5f.create_dataset('den'+sStr,   np.shape(den),      dtype='f8', data=den)
      h5f.create_dataset('upar'+sStr,  np.shape(upar),     dtype='f8', data=upar)
      h5f.create_dataset('Tpar'+sStr,  np.shape(tpar),     dtype='f8', data=tpar)
      h5f.create_dataset('Tperp'+sStr, np.shape(tperp),    dtype='f8', data=tperp)

#    ax21[sI][0].set_ylim(0., n0)
    for i in range(len(ax21[sI])):
      ax21[sI][i].set_xlim( xIntC[0][0], xIntC[0][-1] )
      setTickFontSize(ax21[sI][i],tickFontSize) 
      ax21[sI][i].tick_params(axis='y', labelcolor=defaultColors[sI], labelsize=tickFontSize)
      hmag = ax21[sI][i].yaxis.get_offset_text().set_size(tickFontSize)
    for i in range(3):
      plt.setp( ax21[sI][i].get_xticklabels(), visible=False)
    ax21[sI][0].set_ylabel(r'$n_%s$ (m$^{-3}$)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI], labelpad=-4+2*sI)
    ax21[sI][1].set_ylabel(r'$u_{\parallel %s}/c_{se0}$' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI], labelpad=-2)
    ax21[sI][2].set_ylabel(r'$T_{\parallel %s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI])
    ax21[sI][3].set_ylabel(r'$T_{\perp %s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI])
    sI = sI+1

  for i in range(len(ax21[0])):
    plot_verticalLinesPM(z_m, ax21[0][i])  #[ Indicate location of max B.
  ax21[0][3].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  plt.text(0.025, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax21[0][0].transAxes)
  plt.text(0.025, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax21[0][1].transAxes)
  plt.text(0.025, 0.85, r'(c)', fontsize=textFontSize, color='black', transform=ax21[0][2].transAxes)
  plt.text(0.025, 0.85, r'(d)', fontsize=textFontSize, color='black', transform=ax21[0][3].transAxes)
  
  if outFigureFile:
    plt.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f.close()

#................................................................................#

if plot_phi_t_comp:
  #[ Plot phi(z,t) and e*Delta phi/Te vs. t for all three models.

  #.Linear polarization, nonlinear polarization, and adiabatic electrons.
  dataDir = ['/scratch/gpfs/manaurer/gkeyll/mirror/gk77/',
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk75wExBenergy_vparRes2x/',
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk55/',]
  fileName = ['gk77-wham1x2v','gk75-wham1x2v','gk55-wham1x2v']    #.Root name of files to process.
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

  #[ Place a line at the analytic Pastukhov expectation (A_p=0.5 and A_p=1):
  #[ 4.5494485-0.82591375 = 3.72353475
  timeLims = [0., 100.]
  ax12[-1].plot([timeLims[0],timeLims[-1]],[3.72,3.72],color='grey',linestyle=lineStyles[3])
  plt.text(0.03, 0.7, r'$A_p=0.5$', fontsize=14, color='grey', transform=ax12[-1].transAxes)
  #[ 5.115795-1.13181857 = 3.98397643
  ax12[-1].plot([timeLims[0],timeLims[-1]],[3.98,3.98],color='grey',linestyle=lineStyles[3])
  plt.text(0.03, 0.88, r'$A_p=1$', fontsize=14, color='grey', transform=ax12[-1].transAxes)

  hpl12 = list()
  hcb12 = list()
  hpl12b = list()
  timeLims = [1e10, -1e10]
  for sI in range(len(fileName)):
  
    phiFile = dataDir[sI]+fileName[sI]+'_phi_'    #.Complete file name.
    if sI == 2:
      nFrames = 131
    else:
      nFrames = 1+pgu.findLastFrame(phiFile, 'bp')
    times = 1.e6*pgu.getTimeStamps(phiFile,0,nFrames-1)
  
    phiFile = dataDir[sI]+fileName[sI]+'_phi_%d.bp'    #.Complete file name.
    TeFile = dataDir[sI]+fileName[sI]+'_elc_gridDiagnostics_%d.bp'    #.Complete file name.
  
    #[ Load the grid.
    xInt, _, nxInt, lxInt, dxInt, _ = pgu.getGrid(phiFile % 0,polyOrder,basisType)
    xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(phiFile % 0,polyOrder,basisType,location='center')
  
    phi = np.zeros((nFrames,nxIntC[0]))
    Te = np.zeros((nFrames,nxIntC[0]))
    if sI == 0:
      Te = Te0*np.ones((nFrames,nxIntC[0]))
    elif sI == 1:
      dBdzFile = './'+fileName[sI]+'-dBdz_dBdz.bp'    #.Complete file name.
      fsPi = forceSofteningPi_zvparmu(dataDir[sI]+fileName[sI],'elc',0,dBdzFile)
      geoFile = dataDir[sI]+fileName[sI]+'_allGeo_0.bp'
      Bmag    = np.squeeze(pgu.getInterpData(geoFile,polyOrder,basisType,location='center',varName='bmag'))
      jacInv  = np.squeeze(pgu.getInterpData(geoFile,polyOrder,basisType,location='center',varName='jacobGeoInv'))
      distfFile = dataDir[sI]+fileName[sI]+'_elc_0.bp'
      pxIntC, _, pnxIntC, _, _, _ = pgu.getGrid(distfFile,polyOrder,basisType,location='center')
      gridParams = {"xIntC" : pxIntC, "nxIntC" : pnxIntC}
  
    eDeltaPhiDTe = np.zeros(nFrames)
    for fI in range(nFrames):
      if sI != 0:
        Te[fI,:] = np.squeeze(pgu.getInterpData(TeFile % fI, polyOrder, basisType,varName='Temp'))
#      if sI == 1:
#        #[ Alternative calculation of T_e.
##        dBdzFile = './'+fileName[sI]+'-dBdz_dBdz.bp'    #.Complete file name.
##        fsPi = forceSofteningPi_zvparmu(dataDir[sI]+fileName[sI],'elc',fI,dBdzFile)
#        tpar = np.squeeze(pgu.getInterpData(TeFile % fI, polyOrder, basisType,varName='Tpar'))
#        Te[fI,:] = (tpar+2.*calcTperp(dataDir[sI]+fileName[sI],'elc',fI,phaseFac=fsPi,bmag=Bmag,jacobGeoInv=jacInv,gridPars=gridParams))/3.
      TeMid = 0.5*(Te[fI,nxIntC[0]//2-1]+Te[fI,nxIntC[0]//2])
      phi[fI,:] = (eV/TeMid)*np.squeeze(pgu.getInterpData(phiFile % fI, polyOrder, basisType))
      eDeltaPhiDTe[fI] = np.interp(0., xIntC[0], phi[fI,:])-np.interp(z_m, xIntC[0], phi[fI,:])
  
    #[ Create colorplot grid. Recall coordinates have to be nodal.
    timesC  = 0.5*(times[1:]+times[0:-1])
    Xnodal = [np.outer(np.append(np.append([timesC[0]-(timesC[1]-timesC[0])],timesC),timesC[-1]+(timesC[-1]-timesC[-2])), \
                       np.ones(np.shape(xInt[0]))), \
              np.outer(np.ones(np.size(timesC)+2),xInt[0])]

    timeLims = [min(timeLims[0],np.amin(Xnodal[0])), max(timeLims[-1],np.amax(Xnodal[0]))]
  
    if sI<2:
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
    hcb12[sI].set_label('$e\phi/T_e(z=0)$', rotation=270, labelpad=18, fontsize=colorBarLabelFontSize)
    hcb12[sI].ax.tick_params(labelsize=tickFontSize)
    plt.setp( ax12[sI].get_xticklabels(), visible=False)
    ax12[sI].set_ylabel(r'$z$ (m)', fontsize=xyLabelFontSize)
  ax12[-1].set_xlabel(r'Time ($\mu$s)', fontsize=xyLabelFontSize)
  ax12[-1].set_ylabel(r'$e\Delta \phi/T_e(z=0)$', fontsize=xyLabelFontSize)
  ax12[-1].legend([hpl12b[0][0], hpl12b[1][0], hpl12b[2][0]],['Boltzmann e$^-$',r'nonlinear pol.',r'linear pol.'], fontsize=legendFontSize, frameon=False)
  plt.text(0.7, 0.88, r'Boltzmann e$^-$', fontsize=14, color='white', fontweight='regular', transform=ax12[0].transAxes)
  plt.text(0.7, 0.88, r'nonlinear pol.' , fontsize=14, color='white', fontweight='regular', transform=ax12[1].transAxes)
  plt.text(0.025, 0.05, r'(a)', fontsize=textFontSize, color='white', transform=ax12[0].transAxes)
  plt.text(0.025, 0.05, r'(b)', fontsize=textFontSize, color='white', transform=ax12[1].transAxes)
  plt.text(0.025, 0.05, r'(c)', fontsize=textFontSize, color='black', transform=ax12[2].transAxes)
  for i in range(len(ax12)):
    ax12[i].set_xlim( timeLims[0], timeLims[-1] )
    setTickFontSize(ax12[i],tickFontSize) 
  
  if outFigureFile:
    fig12.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f.close()

#................................................................................#

if plot_mom_comp:
  #[ Plot density, temperature and u_parallel along the field
  #[ line for the kinetic elc sim with a nonlinear polarization
  #[ and the adiabatic elec sim.

  dataDir = ['/scratch/gpfs/manaurer/gkeyll/mirror/gk75wExBenergy_vparRes2x/',
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk77/',]
  
  #.Root name of files to process.
  fileName = ['gk75-wham1x2v',
              'gk77-wham1x2v',]
  frame    = 400
  figName  = 'gk75-77-wham1x2v_M0TempUpar_'+str(frame)

  #[ Prepare figure.
  figProp21 = (7.,8.5)
  ax21Pos   = [[0.11, 0.750, 0.77, 0.215],
               [0.11, 0.525, 0.77, 0.215],
               [0.11, 0.300, 0.77, 0.215],
               [0.11, 0.075, 0.77, 0.215]]
  fig21     = plt.figure(figsize=figProp21)
  ax21      = [[],[]]
  ax21[0]   = [fig21.add_axes(d) for d in ax21Pos]
  ax21[1]   = [ax.twinx() for ax in ax21[0]]

  if saveData:
    h5f = h5py.File(outDir+figName+'.h5', "w")

  c_s = np.sqrt(Te0/mi)

  for fI in range(len(dataDir)):

    fName = dataDir[fI]+fileName[fI]+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.
    species = ["ion","elc"]
    if fI == 1:
      species = ["ion"]

    sI = 0
    for spec in species:
      #[ Load the grid.
      xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % (spec,frame),polyOrder,basisType,location='center',varName='M0')
      
      #[ Load the electron density, temperatures and flows.
      den   =            np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='M0'))
      upar  = (1./c_s)*  np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Upar'))
      tpar  = (1.e-3/eV)*np.squeeze(pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Tpar'))
      #tperp = (1.e-3/eV)*pgu.getInterpData(fName % (spec,frame), polyOrder, basisType, varName='Tperp')
      #[ Alternative calculation of T_perp.
      dBdzFile = './'+fileName[fI]+'-dBdz_dBdz.bp'    #.Complete file name.
      fsPi = forceSofteningPi_zvparmu(dataDir[fI]+fileName[fI],spec,frame,dBdzFile)
      tperp = (1.e-3/eV)*calcTperp(dataDir[fI]+fileName[fI],spec,frame,phaseFac=fsPi)
  
  
      hpl21 = list()
      hpl21.append(ax21[sI][0].semilogy(xIntC[0], den, color=defaultColors[sI], linestyle=lineStyles[fI*2+sI], marker=markers[4*fI], markersize=4, markevery=12))
      hpl21.append(ax21[sI][1].plot(xIntC[0], upar,    color=defaultColors[sI], linestyle=lineStyles[fI*2+sI], marker=markers[4*fI], markersize=4, markevery=12))
      hpl21.append(ax21[sI][2].plot(xIntC[0], tpar,    color=defaultColors[sI], linestyle=lineStyles[fI*2+sI], marker=markers[4*fI], markersize=4, markevery=12))
      hpl21.append(ax21[sI][3].plot(xIntC[0], tperp,   color=defaultColors[sI], linestyle=lineStyles[fI*2+sI], marker=markers[4*fI], markersize=4, markevery=12))

      if saveData:
        sStr = '_i'
        if fI == 1:
          sStr = '_i_boltz'
        if spec == 'elc':
          sStr = '_e'
          h5f.create_dataset('z', np.shape(xIntC[0]), dtype='f8', data=xIntC[0])
        h5f.create_dataset('den'+sStr,   np.shape(den),      dtype='f8', data=den)
        h5f.create_dataset('upar'+sStr,  np.shape(upar),     dtype='f8', data=upar)
        h5f.create_dataset('Tpar'+sStr,  np.shape(tpar),     dtype='f8', data=tpar)
        h5f.create_dataset('Tperp'+sStr, np.shape(tperp),    dtype='f8', data=tperp)

      sI = sI+1
  
  sI = 0
  for spec in ["ion","elc"]:
#    ax21[sI][0].set_ylim(0., n0*1.075)
    for i in range(len(ax21[sI])):
      ax21[sI][i].set_xlim( xIntC[0][0], xIntC[0][-1] )
      setTickFontSize(ax21[sI][i],tickFontSize) 
      ax21[sI][i].tick_params(axis='y', labelcolor=defaultColors[sI], labelsize=tickFontSize)
      hmag = ax21[sI][i].yaxis.get_offset_text().set_size(tickFontSize)
    for i in range(3):
      plt.setp( ax21[sI][i].get_xticklabels(), visible=False)
    ax21[sI][0].set_ylabel(r'$n_%s$ (m$^{-3}$)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI], labelpad=-4+4*sI)
    ax21[sI][1].set_ylabel(r'$u_{\parallel %s}/c_{se0}$' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI], labelpad=-2)
    ax21[sI][2].set_ylabel(r'$T_{\parallel %s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI])
    ax21[sI][3].set_ylabel(r'$T_{\perp %s}$ (keV)' % (spec[0]), fontsize=xyLabelFontSize, color=defaultColors[sI])
    sI = sI+1

  for i in range(len(ax21[0])):
    plot_verticalLinesPM(z_m, ax21[0][i])  #[ Indicate location of max B.
  ax21[0][3].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  plt.text(0.025, 0.85, r'(a)', fontsize=textFontSize, color='black', transform=ax21[0][0].transAxes)
  plt.text(0.025, 0.85, r'(b)', fontsize=textFontSize, color='black', transform=ax21[0][1].transAxes)
  plt.text(0.025, 0.85, r'(c)', fontsize=textFontSize, color='black', transform=ax21[0][2].transAxes)
  plt.text(0.025, 0.85, r'(d)', fontsize=textFontSize, color='black', transform=ax21[0][3].transAxes)
  
  if outFigureFile:
    plt.savefig(outDir+figName+figureFileFormat)
    plt.close()
  else:
    plt.show()

  if saveData:
    h5f.close()

#................................................................................#

if plot_mom_forceSoftComp:
  #[ Plot density, temperature and u_parallel along the field line for the adiabatic elc sims
  #[ with and without force softening.

  dataDir = ['/scratch/gpfs/manaurer/gkeyll/mirror/gk57/',
             '/scratch/gpfs/manaurer/gkeyll/mirror/gk71/']
  
  fileName = ['gk57-wham1x2v','gk71-wham1x2v']    #.Root name of files to process.
  frame    = 144

  #[ Prepare figure.
  figProp8 = (6.8,7.2)
  ax8Pos   = [[0.12, 0.685, 0.77, 0.285],
              [0.12, 0.385, 0.77, 0.285],
              [0.12, 0.085, 0.77, 0.285]]
  fig8     = plt.figure(figsize=figProp8)
  ax8      = [fig8.add_axes(d) for d in ax8Pos]
  ax82_twin = ax8[2].twinx()

  hpl8 = list()
  hpl82_twin = list()
  for fI in range(len(fileName)):

    fName = dataDir[fI]+fileName[fI]+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.

    #[ Load the grid.
    xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % ('ion',frame),polyOrder,basisType,location='center',varName='M0')
    
    #[ Load the electron density, temperatures and flows.
    den  = pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='M0')
    temp = (1.e-3/eV)*pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Temp')
    tpar = (1.e-3/eV)*pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Tpar')
    tperp = (1.e-3/eV)*pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Tperp')
    upar = pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Upar')

    c_s = np.sqrt(Te0/mi)
#    c_s = np.sqrt((Te0+3.*Ti0)/mi)
#    c_s = np.sqrt((Te0+3.*tpar/(1.e-3/eV))/mi)

    hpl8.append(ax8[0].plot(xIntC[0], den, color=defaultColors[fI], linestyle=lineStyles[fI]))
    hpl8.append(ax8[1].plot(xIntC[0], upar/c_s, color=defaultColors[fI], linestyle=lineStyles[fI]))
    hpl8.append(ax8[2].plot(xIntC[0], tpar, color=defaultColors[fI], linestyle=lineStyles[fI]))

    hpl82_twin.append(ax82_twin.plot(xIntC[0], tperp, color=defaultColors[fI], linestyle=lineStyles[fI+2]))

  for i in range(len(ax8)):
    ax8[i].set_xlim( xIntC[0][0], xIntC[0][-1])
    setTickFontSize(ax8[i],tickFontSize) 
    ax8[i].tick_params(axis='y', labelcolor=defaultColors[0])
    hmag = ax8[i].yaxis.get_offset_text().set_size(tickFontSize)
    plot_verticalLinesPM(z_m, ax8[i])  #[ Indicate location of max B.
  hmag = ax82_twin.yaxis.get_offset_text().set_size(tickFontSize)
  ax82_twin.tick_params(axis='y', labelcolor=defaultColors[1], labelsize=tickFontSize)
  plt.setp( ax8[0].get_xticklabels(), visible=False)
  plt.setp( ax8[1].get_xticklabels(), visible=False)
  ax8[2].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  ax8[0].set_ylabel(r'$n_i$ (m$^{-3}$)', fontsize=xyLabelFontSize, color=defaultColors[0])
  ax8[1].set_ylabel(r'$u_{\parallel i}/c_{se0}$', fontsize=xyLabelFontSize, color=defaultColors[0])
#  ax8[1].set_ylabel(r'$u_{\parallel i}$ (m/s)', fontsize=xyLabelFontSize, color=defaultColors[0])
  ax8[2].set_ylabel(r'$T_{\parallel i}$ (keV)', fontsize=xyLabelFontSize, color=defaultColors[0])
  ax82_twin.set_ylabel(r'$T_{\perp i}$ (keV)', fontsize=xyLabelFontSize, color=defaultColors[1])
  
  if outFigureFile:
    plt.savefig(outDir+fileName+'_M0TempUpar_'+str(frame)+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_M0:
  #[ Plot density along the field line.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk43/'
  
  fileName = 'gk43-wham1x2v'    #.Root name of files to process.
  frame    = 32

  #[ Prepare figure.
  figProp3 = (6,3.2)
  ax3Pos   = [0.1, 0.19, 0.88, 0.78]
  fig3     = plt.figure(figsize=figProp3)
  ax3      = fig3.add_axes(ax3Pos)

  fName = dataDir+fileName+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.

  #[ Load the grid.
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % ('elc',frame),polyOrder,basisType,location='center',varName='M0')
  
  #[ Load the electrostatic potential phi.
  denElc = pgu.getInterpData(fName % ('elc',frame), polyOrder, basisType, varName='M0')
  denIon = pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='M0')

  hpl3 = list()
  hpl3.append(ax3.semilogy(xIntC[0], denElc, color=defaultColors[0], linestyle=lineStyles[0]))
  hpl3.append(ax3.semilogy(xIntC[0], denIon, color=defaultColors[1], linestyle=lineStyles[1]))
  ax3.set_xlim( xIntC[0][0], xIntC[0][-1])
  ax3.set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  ax3.legend([hpl3[0][0], hpl3[1][0]],[r'$n_e$ (m$^{-3}$)', r'$n_i$ (m$^{-3}$)'], fontsize=legendFontSize, frameon=False)
  setTickFontSize(ax3,tickFontSize) 
  
  if outFigureFile:
    plt.savefig(outDir+fileName+'_M0_'+str(frame)+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_Temp:
  #[ Plot temperature along the field line.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk43/'
  
  fileName = 'gk43-wham1x2v'    #.Root name of files to process.
  frame    = 32

  #[ Prepare figure.
  figProp4 = (6,3.2)
  ax4Pos   = [0.1, 0.19, 0.88, 0.78]
  fig4     = plt.figure(figsize=figProp4)
  ax4      = fig4.add_axes(ax4Pos)

  fName = dataDir+fileName+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.

  #[ Load the grid.
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % ('elc',frame),polyOrder,basisType,location='center',varName='Temp')
  
  #[ Load the electrostatic potential phi.
  TempElc = (1./1.602e-19)*pgu.getInterpData(fName % ('elc',frame), polyOrder, basisType, varName='Temp')
  TempIon = (1./1.602e-19)*pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Temp')

  hpl4 = list()
  hpl4.append(ax4.semilogy(xIntC[0], TempElc, color=defaultColors[0], linestyle=lineStyles[0]))
  hpl4.append(ax4.semilogy(xIntC[0], TempIon, color=defaultColors[1], linestyle=lineStyles[1]))
  ax4.set_xlim( xIntC[0][0], xIntC[0][-1])
  ax4.set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  ax4.legend([hpl4[0][0], hpl4[1][0]],[r'$T_e$ (eV)', r'$T_i$ (eV)'], fontsize=legendFontSize, frameon=False)
  setTickFontSize(ax4,tickFontSize) 
  
  if outFigureFile:
    plt.savefig(outDir+fileName+'_Temp_'+str(frame)+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_intM0:
  #[ Plot integrated density in time:

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk43/'
  
  fileName = 'gk43-wham1x2v'    #.Root name of files to process.

  #[ Prepare figure.
  figProp5 = (6,3.2)
  ax5Pos   = [[0.11, 0.19, 0.87, 0.75]]
  fig5     = plt.figure(figsize=figProp5)
  ax5      = [fig5.add_axes(pos) for pos in ax5Pos]

  fName = dataDir+fileName+'_%s_intM0.bp'    #.Complete file name.

  #[ Load the electrostatic potential phi.
  timeElc, intM0Elc = pgu.readDynVector(fName % 'elc')
  timeIon, intM0Ion = pgu.readDynVector(fName % 'ion')
  #[ Scale so we plot in microsec:
  timeElc, timeIon = 1.e6*timeElc, 1.e6*timeIon

  hpl5 = list()
  hpl5.append(ax5[0].plot(timeElc, intM0Elc, color=defaultColors[0], linestyle=lineStyles[0]))
  hpl5.append(ax5[0].plot(timeIon, intM0Ion, color=defaultColors[1], linestyle=lineStyles[1]))
  ax5[0].set_xlim( min(timeElc[0], timeIon[0]), max(timeElc[-1],timeIon[-1]))
  ax5[0].set_xlabel(r'Time ($\mu$s)', fontsize=xyLabelFontSize)
#  ax5[0].set_ylabel(r'$N_e = \int dz\,Jn_s$ ', fontsize=xyLabelFontSize)
  ax5[0].legend([hpl5[0][0], hpl5[1][0]],[r'$\left\langle n_e\right\rangle$', r'$\left\langle n_{Gi}\right\rangle$'], fontsize=legendFontSize, frameon=False)
  setTickFontSize(ax5[0],tickFontSize) 
  hmag = ax5[0].yaxis.get_offset_text().set_size(tickFontSize)
  
  if outFigureFile:
    plt.savefig(outDir+fileName+'_intM0'+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_M0TempUpar:
  #[ Plot density, temperature and u_parallel along the field line.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk55/'
  
  fileName = 'gk55-wham1x2v'    #.Root name of files to process.
  frame    = 131

  #[ Prepare figure.
  figProp3 = (6.8,7.2)
  ax3Pos   = [[0.13, 0.69, 0.74, 0.28],
              [0.13, 0.395, 0.74, 0.28],
              [0.13, 0.085, 0.74, 0.28]]
  fig3     = plt.figure(figsize=figProp3)
  ax3      = [fig3.add_axes(d) for d in ax3Pos]

  fName = dataDir+fileName+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.

  #[ Load the grid.
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % ('elc',frame),polyOrder,basisType,location='center',varName='M0')
  
  #[ Load the electron density, temperatures and flows.
  denElc  = pgu.getInterpData(fName % ('elc',frame), polyOrder, basisType, varName='M0')
  tempElc = (1.e-3/eV)*pgu.getInterpData(fName % ('elc',frame), polyOrder, basisType, varName='Temp')
  tempIon = (1.e-3/eV)*pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Temp')
  uparElc = pgu.getInterpData(fName % ('elc',frame), polyOrder, basisType, varName='Upar')
  uparIon = pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Upar')

  hpl3 = list()
  hpl3.append(ax3[0].plot(xIntC[0], denElc, color=defaultColors[0], linestyle=lineStyles[0]))
  hpl3.append(ax3[1].plot(xIntC[0], tempElc, color=defaultColors[0], linestyle=lineStyles[0]))
  hpl3.append(ax3[2].plot(xIntC[0], uparElc, color=defaultColors[0], linestyle=lineStyles[0]))
#  hpl3.append(ax3[2].plot(xIntC[0], 1.e-3*uparElc, color=defaultColors[0], linestyle=lineStyles[0]))

  ax3_twin = [ax.twinx() for ax in ax3]
  hpl3_twin = list()
  hpl3_twin.append(ax3_twin[1].plot(xIntC[0], tempIon, color=defaultColors[1], linestyle=lineStyles[1]))
  hpl3_twin.append(ax3_twin[2].plot(xIntC[0], uparIon, color=defaultColors[1], linestyle=lineStyles[1]))
#  hpl3_twin.append(ax3_twin[2].plot(xIntC[0], 1.e-3*uparIon, color=defaultColors[1], linestyle=lineStyles[1]))

  for i in range(len(ax3)):
    ax3[i].set_xlim( xIntC[0][0], xIntC[0][-1])
    setTickFontSize(ax3[i],tickFontSize) 
    ax3[i].tick_params(axis='y', labelcolor=defaultColors[0])
    hmag = ax3[i].yaxis.get_offset_text().set_size(tickFontSize)
    plot_verticalLinesPM(z_m, ax3[i])  #[ Indicate location of max B.
  for i in range(1,3):
    hmag = ax3_twin[i].yaxis.get_offset_text().set_size(tickFontSize)
    ax3_twin[i].tick_params(axis='y', labelcolor=defaultColors[1], labelsize=tickFontSize)
  plt.setp( ax3[0].get_xticklabels(), visible=False)
  plt.setp( ax3[1].get_xticklabels(), visible=False)
  plt.setp( ax3_twin[0].get_yticklabels(), visible=False)
  ax3[2].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  ax3[0].set_ylabel(r'$n_e$ (m$^{-3}$)', fontsize=xyLabelFontSize, color=defaultColors[0])
  ax3[1].set_ylabel(r'$T_e$ (keV)', fontsize=xyLabelFontSize, color=defaultColors[0])
  ax3[2].set_ylabel(r'$u_{\parallel e}$ (m/s)', fontsize=xyLabelFontSize, color=defaultColors[0])
#  ax3[2].set_ylabel(r'$u_{\parallel e}$ (km/s)', fontsize=xyLabelFontSize, color=defaultColors[0], labelpad=-9)
  ax3_twin[1].set_ylabel(r'$T_i$ (keV)', fontsize=xyLabelFontSize, color=defaultColors[1])
  ax3_twin[2].set_ylabel(r'$u_{\parallel i}$ (m/s)', fontsize=xyLabelFontSize, color=defaultColors[1])
#  ax3_twin[2].set_ylabel(r'$u_{\parallel i}$ (km/s)', fontsize=xyLabelFontSize, color=defaultColors[1], labelpad=-2)
#  ax3[1].legend([hpl3[1][0], hpl3[2][0]],[r'electrons', r'ions'], fontsize=legendFontSize, frameon=False)
  
  if outFigureFile:
    plt.savefig(outDir+fileName+'_M0TempUpar_'+str(frame)+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_PparPperp[0] or plot_PparPperp[1]:
  #[ Plot density, temperature and u_parallel along the field line.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/gk55/'
  
  fileName = 'gk55-wham1x2v'    #.Root name of files to process.
  frame    = 131

  fName = dataDir+fileName+'_%s_gridDiagnostics_%d.bp'    #.Complete file name.

  #[ Load the grid.
  xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fName % ('elc',frame),polyOrder,basisType,location='center',varName='M0')
  
  #[ Load the electron density, temperatures and flows.
  denElc   = pgu.getInterpData(fName % ('elc',frame), polyOrder, basisType, varName='M0')
  denIon   = pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='M0')
  tparElc  = pgu.getInterpData(fName % ('elc',frame), polyOrder, basisType, varName='Tpar')
  tperpElc = pgu.getInterpData(fName % ('elc',frame), polyOrder, basisType, varName='Tperp')
  tparIon  = pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Tpar')
  tperpIon = pgu.getInterpData(fName % ('ion',frame), polyOrder, basisType, varName='Tperp')

  pparElc  = denElc*tparElc
  pperpElc = denElc*tperpElc
  pparIon  = denIon*tparIon
  pperpIon = denIon*tperpIon

  if plot_PparPperp[0]:
    #[ Prepare figure.
    figProp3 = (6.2,5.2)
    ax3Pos   = [[0.11, 0.55, 0.87, 0.4],
                [0.11, 0.12, 0.87, 0.4]]
    fig3     = plt.figure(figsize=figProp3)
    ax3      = [fig3.add_axes(d) for d in ax3Pos]

    hpl3 = list()
    hpl3.append(ax3[0].plot(xIntC[0], 1.e-3*pparElc, color=defaultColors[0], linestyle=lineStyles[0]))
    hpl3.append(ax3[0].plot(xIntC[0], 1.e-3*pperpElc, color=defaultColors[0], linestyle=lineStyles[2]))
    hpl3.append(ax3[1].plot(xIntC[0], 1.e-3*pparIon, color=defaultColors[1], linestyle=lineStyles[1]))
    hpl3.append(ax3[1].plot(xIntC[0], 1.e-3*pperpIon, color=defaultColors[1], linestyle=lineStyles[3]))

    for i in range(len(ax3)):
      ax3[i].set_xlim( xIntC[0][0], xIntC[0][-1])
      setTickFontSize(ax3[i],tickFontSize) 
      hmag = ax3[i].yaxis.get_offset_text().set_size(tickFontSize)
      plot_verticalLinesPM(z_m, ax3[i])  #[ Indicate location of max B.
      ax3[i].set_ylabel(r'(kPa)', fontsize=xyLabelFontSize)
    plt.setp( ax3[0].get_xticklabels(), visible=False)
    ax3[1].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
    ax3[0].legend([r'$p_{\parallel e}$', r'$p_{\perp e}$'], fontsize=legendFontSize, frameon=False)
    ax3[1].legend([r'$p_{\parallel i}$', r'$p_{\perp i}$'], fontsize=legendFontSize, frameon=False)
    
    if outFigureFile:
      plt.savefig(outDir+fileName+'_PparPperp_'+str(frame)+figureFileFormat)
      plt.close()
    else:
      plt.show()

  if plot_PparPperp[1]:
    #[ Prepare figure.
    figProp3 = (6.2,3.2)
    ax3Pos   = [[0.13, 0.19, 0.85, 0.78]]
    fig3     = plt.figure(figsize=figProp3)
    ax3      = [fig3.add_axes(d) for d in ax3Pos]

    pRatElc = pperpElc/pparElc
    pRatIon = pperpIon/pparIon

    hpl3 = list()
    hpl3.append(ax3[0].plot(xIntC[0], pRatElc, color=defaultColors[0], linestyle=lineStyles[0]))
    hpl3.append(ax3[0].plot(xIntC[0], pRatIon, color=defaultColors[1], linestyle=lineStyles[1]))

    for i in range(len(ax3)):
      ax3[i].set_xlim( xIntC[0][0], xIntC[0][-1])
      setTickFontSize(ax3[i],tickFontSize) 
      hmag = ax3[i].yaxis.get_offset_text().set_size(tickFontSize)
      plot_verticalLinesPM(z_m, ax3[i])  #[ Indicate location of max B.
    ax3[0].set_ylim(0., 1.05*max(np.amax(pRatElc),np.amax(pRatIon)))
    ax3[0].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
    ax3[0].set_ylabel(r'$p_\perp/p_\parallel$', fontsize=xyLabelFontSize)
    ax3[0].legend([r'electron', r'ion'], fontsize=legendFontSize, frameon=False)
    
    if outFigureFile:
      plt.savefig(outDir+fileName+'_PparPperpRat_'+str(frame)+figureFileFormat)
      plt.close()
    else:
      plt.show()

#................................................................................#

if plot_adiabaticElc:
  #[ Plot phi and e*phi/Te for all kperp*rhos.

  dataDir = '/scratch/gpfs/manaurer/gkeyll/mirror/'

  tests = ['gk57','gk55']
  legends = ['adiabatic e$^-$','kinetic e$^-$']

  fileName = '%s-wham1x2v'    #.Root name of files to process.
  frame    = 131

  #[ Prepare figure.
  figProp11 = (6,5.)
  ax11Pos   = [[0.14, 0.56, 0.84, 0.42],
               [0.14, 0.12, 0.84, 0.42]] 
  fig11     = plt.figure(figsize=figProp11)
  ax11      = [fig11.add_axes(pos) for pos in ax11Pos]

  hpl11a, hpl11b = list(), list()

  for tI in range(len(tests)):

    phiFile = dataDir+tests[tI]+'/'+fileName%tests[tI]+'_phi_%d.bp'    #.Complete file name.
  
    #[ Load the grid.
    x, _, nx, lx, dx, _ = pgu.getGrid(phiFile % frame,polyOrder,basisType,location='center')
    
    #[ Load phi and Te.
    phi = pgu.getInterpData(phiFile % frame, polyOrder, basisType)
    ePhiDTe = eV*phi/Te0
  
    hpl11a.append(ax11[0].semilogy(x[0], phi, color=defaultColors[2-tI], linestyle=lineStyles[tI], marker=markers[-1+tI], markevery=6))
    hpl11b.append(ax11[1].plot(x[0], ePhiDTe, color=defaultColors[2-tI], linestyle=lineStyles[tI], marker=markers[-1+tI], markevery=6))

  plt.setp( ax11[0].get_xticklabels(), visible=False)
  for i in range(len(ax11)):
    ax11[i].set_xlim( x[0][0], x[0][-1])
    setTickFontSize(ax11[i],tickFontSize) 
  ax11[0].set_ylabel(r'$\phi$ (V)', fontsize=xyLabelFontSize)
  ax11[1].set_ylabel(r'$e\phi/T_{e0}$', fontsize=xyLabelFontSize)
  ax11[1].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  ax11[0].legend(legends, fontsize=legendFontSize, frameon=False)
  
  if outFigureFile:
    fig11.savefig(outDir+'gk-wham1x2v_adiabaticElc_phi_ePhiDTe0_'+str(frame)+figureFileFormat)
    plt.close()
  else:
    plt.show()

#................................................................................#

if plot_nonuniformz:
  #[ Plot something from a nonuniform z setup.

  #[ Load the ion distribution function and its grid.
  dataDir  = '/scratch/gpfs/manaurer/gkeyll/mirror/gk55gnew_nonuniformz/'
  fileName = 'gk55-wham1x2v'    #.Root name of files to process.
#  dataDir  = '/scratch/gpfs/manaurer/gkeyll/mirror/gk57/'
#  fileName = 'gk57-wham1x2v'    #.Root name of files to process.

  fName = dataDir+fileName+'_ion_0.bp'    #.Complete file name.
  x_i, _, nx_i, lx_i, dx_i, _ = pgu.getRawGrid(fName)
  xInt_i, _, nxInt_i, lxInt_i, dxInt_i, _ = pgu.getGrid(fName,polyOrder,basisType)
  xIntC_i, _, nxIntC_i, lxIntC_i, dxIntC_i, _ = pgu.getGrid(fName,polyOrder,basisType,location='center')

  def z_xi(xi):
    #[ Computational to physical mapping z(xi):
    b = 3.
    c = 0.98
    zMax = 2.515312
    f = (1.-1./(1.+(zMax/c)**b))**(-1.)
    return np.where(xi < 0.,
                    -c*np.power( 1./(1.+xi/f)-1., 1./b),
                     c*np.power( 1./(1.-xi/f)-1., 1./b))
#    return xi

  print("min(diff(z)) = ",np.amin(np.diff(z_xi(x_i[0])))," | max(diff(z)) = ",np.amax(np.diff(z_xi(x_i[0]))))
  #[ Load the magnetic field amplitude.
  fName = dataDir+fileName+'_allGeo_0.bp'    #.Complete file name.
  bmag = np.squeeze(pgu.getInterpData(fName, polyOrder, basisType, varName='bmag'))
  
  #[ Prepare figure.
  figProp4 = (6.8,4.6)
  ax4Pos   = [[0.14, 0.14, 0.82, 0.80],]
  fig4     = plt.figure(figsize=figProp4)
  ax4      = [fig4.add_axes(pos) for pos in ax4Pos]
  
  hpl4a = list()
  hpl4a.append(ax4[0].plot(z_xi(xIntC_i[0]), bmag, color=defaultBlue, linestyle='-', marker='.'))

  ax4[0].set_xlabel(r'Length along field line, $z$ (m)', fontsize=xyLabelFontSize)
  ax4[0].set_ylabel(r'$B$ (T)', fontsize=xyLabelFontSize)
  setTickFontSize(ax4[0],tickFontSize) 
  
#  fileName = 'gk53-wham1x2v-nonuniformmu'    #.Root name of files to process.
  if outFigureFile:
    fig4.savefig(outDir+'gk-wham1x2v_ion_fi_mirrorPenalty_zmu_128'+figureFileFormat)
    plt.close()
  else:
    plt.show()
  
#................................................................................#
