import postgkyl as pg
import matplotlib.pyplot as plt
import numpy as np
from random import seed
from random import random
from time import sleep
import os
import pylab
pylab.rcParams['font.size'] = 14
params = {'legend.fontsize': 12}

seed(99999999)

nbins = 20

#========== Physical Constants and Simulation Parameters ==========
B0 = 2.57             # [T]
q0 = 4.67
r0 = 0.5              # [m]

eV = 1.602176487e-19  # [C]
me = 9.1093837015e-31 # [kg]
mp = 1.672621637e-27  # [kg]
mi = 2.014*mp         # [kg]
Te = 40*eV
Ti = 72*eV
cs = np.sqrt(Te/mi)
omega_ci = eV*B0/mi
rhos = cs / omega_ci

R = 1.65              # [m]
Lpar_p = 66           # [m]
a0_p = (4*Lpar_p**2/(rhos*R))**(1/5)
v0_p = (2*Lpar_p*rhos**2/R**3)**(1/5)

#========== Calculate Mapped Lx ==========
dataDir = '/scratch/gpfs/dingyunl/gk-g0-app-gpu-hack2024-positivity/gkylzero/'
simName = 'gk_bgk_im_asdex_extend_2xIC_3x2v_p1'
nodalData = pg.GData(dataDir+simName+'-nodes_interp.gkyl') # This is a node file generated with the same input except that (Nx, Ny, Nz) = (64, 64, 64)
nodalVals = nodalData.get_values()
theta_idx = 17
R = nodalVals[:, :, theta_idx, 0]
Z = nodalVals[:, :, theta_idx, 1]
Phi = nodalVals[:, :, theta_idx, 2]

dR = R[-1] - R[0]
dZ = Z[-1] - Z[0]
Lx_mapped = np.average(np.sqrt(np.power(dR, 2.0), np.power(dZ, 2.0))) # 0.0514360453977716 [m]
Lx = 0.172 - 0.15     # magnetic flux
print("Lx=%.8f" % (Lx_mapped/rhos))

#========== Load Data ==========
# raw data col: frame, blobID, blobFrameNo, ...
dir = '/scratch/gpfs/dingyunl/gk-g0-app-gpu-hack2024-positivity/gkylzero/blobData2/'
rawDataP = np.genfromtxt(dir+'blob_size.txt')
maxFrame = len(rawDataP[:,0])
allBlobID = rawDataP[:,1]
maxBlobID = int(np.amax(allBlobID))
allBlobFrameCount = rawDataP[:,2]
blobDataArray = np.zeros(maxBlobID+1)
print(np.shape(blobDataArray))
for i in range(maxFrame):
    bi = int(allBlobID[i])
    blobDataArray[bi] = int(allBlobFrameCount[i])
blobDataArray = blobDataArray[blobDataArray != 0]
print('total blobs', len(blobDataArray))
print('ave blob lifetime', np.mean(blobDataArray))

blobData = np.genfromtxt(dir+'blob_details_x.dat')
blobXwidth = blobData[:,2] * Lx_mapped/Lx / rhos
blobXvel = blobData[:,3] * Lx_mapped/Lx / cs
blobXc = blobData[:,4]

blobData = np.genfromtxt(dir+'blob_details_y.dat')
blobYwidth = blobData[:,2]
blobYvel = blobData[:,3]
blobYc = blobData[:,4]
"""
#========== Flux ==========
print(blobXvel*cs)
print(blobXwidth*rhos)
Rarray = np.linspace(2.1,0.15,100)
blobFlux = np.zeros(np.shape(Rarray))
for i in range(1): #len(blobXc)):
    blobFlux += 1e18*blobXvel[i]*cs*np.exp(-(Rarray - blobXc[i])**2/(2*(blobXwidth[i]*rhos/2)**2))
print('Flux:',blobFlux)

plt.figure()
plt.plot(Rarray,blobFlux)
plt.show()
plt.close()
exit()
"""

lblSize        = 14
ttlSize        = 16
tickSize       = 12
lgdSize        = 12
txtSize        = 14

#========== Velocity ==========
xmin = np.amin(blobXvel)
xmax = np.amax(blobXvel)
ymin = np.amin(blobYvel)
ymax = np.amax(blobXvel)

# Plt 1D histogram
xvelAve = np.mean(blobXvel)
stdDev = np.sqrt(np.mean((blobXvel-xvelAve)**2))
print('xvelAve=%.8f, stdDev=%.8f\n' % (xvelAve, stdDev))

weights = np.ones_like(blobXvel) / len(blobXvel)
fig = plt.figure(figsize=(6.0, 4.5))
plt.hist(blobXvel,bins=nbins,range=[xmin,xmax],color='C1',weights=weights)
plt.axvline(x=xvelAve,color='orangered',linestyle='--',label=r'mean')
plt.title(r'Radial blob velocity', fontsize=ttlSize)
plt.xlabel(r'$v_r$ ($c_s$)',fontsize=lblSize)
#plt.text(0.05,0.95,r'a)',horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#plt.ylim(0,.3)
plt.legend(loc='upper right', fontsize=lgdSize)
plt.tight_layout()
plt.savefig(dir+'blob-xvel-hist.pdf')
#plt.savefig(dir+'blob-xvel-hist.png')
plt.show()
plt.close()

#========== Size ==========
xmax = np.amax(blobXwidth)
xmin = np.amin(blobXwidth)
weights = np.ones_like(blobXwidth) / len(blobXwidth)

# Plt 1D histogram
sizeAve = np.mean(blobXwidth)
print('sizeAve=%.8f\n', sizeAve)

fig = plt.figure(figsize=(6.0, 4.5))
plt.hist(blobXwidth,bins=nbins,range=[xmin,xmax],color='C1',weights=weights)
plt.axvline(x=sizeAve,color='orangered',linestyle='--',label=r'mean')
plt.title(r'Radial blob width', fontsize=ttlSize)
plt.xlabel(r'$a_b$ ($\rho_s$)',fontsize=lblSize)
#plt.text(0.05,0.93,r'a)',horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#plt.ylim(0,.4)
plt.legend(loc='upper right', fontsize=lgdSize)
plt.tight_layout()
plt.savefig(dir+'blob-size-hist.pdf')
#plt.savefig(dir+'blob-size-hist.png')
plt.show()
plt.close()

"""
#========== Size vs Velocity ==========
ahat1 = np.linspace(0.,1,100)
ahat2 = np.linspace(.8,2.5,100)
vhat1 = (2*ahat1)**0.5
vhat2 = 1/ahat2**2

# NSTX params
# mi = 	3.368659976918e-27
# dp_p = 0.721
# c_s0 = 	43617.079640066
# rho_s = 0.00062037219714627
# R = 1.35
# L_z = 8.000000
# a0b = 0.026734
# c_s3 = 2*c_s0
# sig1 = 0.5
# anorm = np.linspace(0,2.5,1000)
# ablob = anorm*a0b

#vbFuncNstx = c_s0*np.sqrt(2*ablob/R)/(1 + sig1*ablob**(5/2)*np.sqrt(2*R/(1 + Ti0/Te0))/(L_z*rho_{st}**2))*dp_p
#vbFuncNstx = np.sqrt(2*ablob/R)/(1 + ablob**(5/2)*np.sqrt(R/2)/(L_z*rhos_b**2))*dp_p

pylab.figure() #figsize=(4.5,6))
pylab.plot(ahat1,vhat1,color='gray',alpha=0.6)
pylab.text(0.1,1.0,r'$\hat{v} \sim \sqrt{2\hat{a}}$')
pylab.text(1.5,.5,r'$\hat{v} \sim 1/\hat{a}^{2}$')
pylab.plot(ahat2,vhat2,color='gray',alpha=0.6)
#pylab.plot(anorm,vbFuncNstx/v0_base,label='theory', linestyle='--',linewidth=2,color='black',alpha=0.6)
pylab.scatter(blobXwidthN/a0_n,blobXvelN/v0_n, label=r'NT',alpha=0.7,marker='D')
pylab.scatter(blobXwidthP/a0_p,blobXvelP/v0_n, label=r'PT',alpha=0.7)
pylab.title('Blob radial velocity vs. size',fontsize=14)
pylab.xlabel(r'$\hat{a}$',fontsize=14)
pylab.ylabel(r'$\hat{v}$',fontsize=14)
pylab.legend()
pylab.tight_layout()
pylab.xlim(0,None)
pylab.savefig('blob-size-vs-vel.pdf')
pylab.show()
pylab.close()
"""
