import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
import sys
#import h5py
#import os

#...........................................................................#
#.
#.Post process cross-BGK validation test.
#.
#.Dingyun Liu.
#.
#...........................................................................#

dataDir = '/scratch/gpfs/dingyunl/g0-main/gkylzero/output_cross/' 

simName = '_cross_relax_1x2v_p1-'    #.Root name of files to process.
nFrames = 9   #.gkyl is not a regular file, can't be detected by os.path.isfile.
polyOrder = 1

tests = ['gk_bgk_im', 'gk_bgk', 'gk_lbo']   
#tests = ['gk_lbo']   
particles = ['elc_', 'ion_']
moments = ['M0_', 'M1_', 'M2_', 'M2par_', 'M2perp_']
parts = ['part1/', 'part2/', 'part3/', 'part4/', 'part5/'] 
#parts = ['part1/', 'part2/', 'part3/', 'part4/', 'part5/', 'part6/'] 
basisType = 'ms'

#.Load the cross-BGK and cross-LBO data.
moms = np.zeros([len(tests), len(particles), len(moments), nFrames*(len(parts)-1)+50]) #.[operator, particle, moment, frame]
time = np.zeros(nFrames*(len(parts)-1)+50)  #.Load times.

for p in range(len(parts)):
  if (p==4):
    for fI in range(50):
      for i in range(len(tests)):
        for j in range(len(particles)):
          for k in range(len(moments)):
            filename = dataDir+parts[p]+tests[i]+simName+particles[j]+moments[k]+str(fI+1)+'.gkyl'
            data = pg.data.GData(filename)
            interp = pg.data.GInterpModal(data, polyOrder, basisType)
            iGrid, iValues = interp.interpolate()
            moms[i,j,k,p*nFrames+fI] = iValues[0][0]
      time[p*nFrames+fI] = data.ctx['time']
  else:
    for fI in range(nFrames):
      for i in range(len(tests)):
        for j in range(len(particles)):
          for k in range(len(moments)):
            filename = dataDir+parts[p]+tests[i]+simName+particles[j]+moments[k]+str(fI+1)+'.gkyl'
            data = pg.data.GData(filename)
            interp = pg.data.GInterpModal(data, polyOrder, basisType)
            iGrid, iValues = interp.interpolate()
            moms[i,j,k,p*nFrames+fI] = iValues[0][0]
      time[p*nFrames+fI] = data.ctx['time']

np.save(dataDir+'moms', moms)
np.save(dataDir+'time', time)
