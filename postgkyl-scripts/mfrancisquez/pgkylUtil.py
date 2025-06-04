#....................................................#
#.
#.pgkylUtil.py
#.Manaure Francisquez.
#.September 2018.
#.
#.This file contains functions and operators used by 
#.other scripts for post-processing Gkeyll data.
#.
#.Here's a list of those functions:
#.  checkFile
#.  checkDir
#.  checkMkdir
#.  findNearestIndex
#.  findLastFrame
#.  getAttribute
#.  getNumCells
#.  getPolyOrder
#.  getBasisType
#.  getLowerBounds
#.  getUpperBounds
#.  getGridType
#.  getGridFileName
#.  getTime
#.  getRawGrid
#.  getGrid
#.  getRawData
#.  readDynVector
#.  evalF1x_e
#.  evalF2x_e
#.  getInterpData
#.  getTimeStamps
#.  plotLocalPoly
#.  plotLocalPoints
#.  getRawDataFEM
#.  getNodeCoordsFEM
#.  getNodeCoordsFEMfromFile
#.  evDGatNodes
#.  commonVarName
#.  getTimeAv
#.  setMinMax
#.  derX 
#.  derY
#.  avAllDim
#.
#....................................................#

#.Import libraries.
import postgkyl as pg
import numpy as np
#.These are used for creating directories.
import os
from os import path
import errno
import shutil
import itertools as itt

sqrt2    = np.sqrt(2.0)
rsqrt2   = np.sqrt(2.0)/2.0
rsqrt2Cu = 1.0/np.sqrt(2.0**3)
sqrt3    = np.sqrt(3.0)
sqrt3d2  = np.sqrt(3.0/2.0)
sqrt5    = np.sqrt(5.0)
sqrt7    = np.sqrt(7.0)
sqrt15   = np.sqrt(15.0)

#.Function to check existence of file.......#
def checkFile(fileIn):
  if os.path.exists(fileIn):
     return True
  else:
     return False

#.Function to check existence of file/directory.......#
def checkDir(dirIn):
  if os.path.exists(os.path.dirname(dirIn)):
     return True
  else:
     return False

#.Function to check existence and/or make directory.......#
def checkMkdir(dirIn):
  if not os.path.exists(os.path.dirname(dirIn)):
    try:
      os.makedirs(os.path.dirname(dirIn))
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise

#.This function finds the index of the grid point nearest to a given fix value.
def findNearestIndex(array,value):
  return (np.abs(array-value)).argmin()
#...end of findNearestIndex function...#

#.Establish last frame outputted (useful for unfinished runs):
def findLastFrame(absFileName,fileExt):
  #.Input is the file name with its absolute address attached to it.
  #.Indicate file type: fileExt='bp' for ADIOS files, ='h5' for HDF5.
  cF         = 0
  moreFrames = os.path.isfile(absFileName+str(cF+1)+'.'+fileExt)
  while moreFrames:
    cF = cF+1
    moreFrames = os.path.isfile(absFileName+str(cF+1)+'.'+fileExt)
  return cF

#.Extract an attribute.
def getAttribute(dataFile, attrName):
  hF      = ad.open(dataFile, 'rra')
  attrOut = hF.attr[attrName].value
  hF.close()
  return attrOut

#.Extract the number of cells of the grid this data is defined on.
def getNumCells(dataFile):
  hF     = ad.open(dataFile, 'rra')
  nCells = hF.read_attribute('numCells')
  hF.close()
  if len(np.shape(nCells))==0:
    nCells = [nCells]
  return nCells

#.Extract the polynomial order of the basis. Does not work with older
#.Gkeyll data files, or g1-style data files that do not save this info. 
def getPolyOrder(dataFile):
  hF     = ad.open(dataFile, 'rra')
  pOrder = hF.read_attribute('polyOrder')
  hF.close()
  return pOrder

#.Extract the basis type. Does not work with older Gkeyll data 
#.files, or g1-style data files that do not save this info. 
def getBasisType(dataFile):
  hF        = ad.open(dataFile, 'rra')
  basisName = hF.read_attribute_string('basisType')[0]
  hF.close()
  if basisName == 'serendipity':
    basisType = 'Ser'
  elif basisName == 'tensor':
    basisType = 'Tensor'
  return basisType

#.Extract the lower bounds of the grid the data in a Gkeyll file is defined on.
def getLowerBounds(dataFile):
  hF    = ad.open(dataFile, 'rra')
  lower = hF.read_attribute('lowerBounds')
  hF.close()
  if len(np.shape(lower))==0:
    lower = [lower]
  return lower

#.Extract the upper bounds of the grid the data in a Gkeyll file is defined on.
def getUpperBounds(dataFile):
  hF    = ad.open(dataFile, 'rra')
  upper = hF.read_attribute('upperBounds')
  hF.close()
  if len(np.shape(upper))==0:
    upper = [upper]
  return upper

#.Determine the type of grid this file is associated with.
def getGridType(dataFile):
  hF       = ad.open(dataFile, 'rra')
  gridType = hF.read_attribute_string('type')[0]
  hF.close()
  return gridType

#.Read the name of the file that contains the grid.
def getGridFileName(dataFile):
  hF       = ad.open(dataFile, 'rra')
  gridFile = hF.read_attribute_string('grid')[0]
  hF.close()
  #.Append the rest of the file address if necessary.
  filePath = dataFile.rsplit('/',maxsplit=1)
  if len(filePath) > 1: 
    gridFile = filePath[0]+'/'+gridFile
  return gridFile 

#.Read the time variable in file..........................#
def getTime(dataFile):
  hF      = ad.open(dataFile, 'rra')
  timeOut = hF.read('time')
  hF.close()
  return timeOut

#.Obtain the true grid (not for interpolated data)..........#
def getRawGrid(dataFile,**opKey):
  gridFile = 'none' #getGridFileName(dataFile)
  gridType = 'uniform' #getGridType(dataFile)
  if (not os.path.exists(gridFile)) or (gridType=='uniform'):
#    print(" Grid file not found. Assuming uniform Cartesian grid.")
    #.Read grid parameters from file, and build grid with pgkyl.
    pgData = pg.GData(dataFile)     #.Read data with pgkyl.
    dimOut = pgData.get_num_dims()
    xNodal = pgData.get_grid()
  
    #.If desired, output cell center values of grid coordinates instead of nodal coordinates.
    if 'location' in opKey:
      if opKey['location']=='center':
        xOut = [[] for i in range(dimOut)]
        for i in range(dimOut):
          nNodes  = np.shape(xNodal[i])[0]
          xOut[i] = np.zeros(nNodes-1)
          xOut[i] = np.multiply(0.5,xNodal[i][0:nNodes-1]+xNodal[i][1:nNodes])
      else:
        xOut = xNodal
    else:
      xOut = xNodal
  
    nxOut = np.zeros(dimOut,dtype='int')
    lxOut = np.zeros(dimOut,dtype='double')
    dxOut = np.zeros(dimOut,dtype='double')
    for i in range(dimOut):
      nxOut[i] = np.size(xOut[i])
      lxOut[i] = xOut[i][-1]-xOut[i][0]
      if nxOut[i] > 1:
        dxOut[i] = xOut[i][ 1]-xOut[i][0]
      else:
        dxOut[i] = xNodal[i][ 1]-xNodal[i][0]

    gridOut = None

  else:
    #.Read grid from grid file.
    pgData  = pg.GData(gridFile)     #.Read data with pgkyl.
    gridOut = pgData.get_values()     #.This contains the nodal coordinates.

    nxOut  = np.shape(gridOut)[0:-1]
    dimOut = np.size(nxOut)
    lxOut  = np.zeros(dimOut,dtype='double')
    dxOut  = [[] for i in range(dimOut)]
    xOut   = [[] for i in range(dimOut)]

    def getNodalGrid():
      for d in range(dimOut):
        slices      = [np.s_[0:1]]*(dimOut+1)
        slices[d]   = np.s_[0:nxOut[d]]
        slices[-1]  = d
        xOut[d]     = np.zeros(nxOut[d], dtype='double')
        xOut[d][0:] = gridOut[slices]
        dxOut[d]   = xOut[d][1:]-xOut[d][0:-1]
        lxOut[d]   = xOut[d][-1]-xOut[d][0]

    #.If desired, output cell center values of grid coordinates instead of nodal coordinates.
    if 'location' in opKey:
      if opKey['location']=='center':
        nxOut = np.subtract(nxOut,1)
        for d in range(dimOut):
          slices        = [[np.s_[0:1]]*(dimOut+1), [np.s_[0:1]]*(dimOut+1)]
          slices[0][d]  = np.s_[0:nxOut[d]]
          slices[1][d]  = np.s_[1:nxOut[d]+1]
          slices[0][-1] = d
          slices[1][-1] = d
          xOut[d]       = np.zeros(nxOut[d], dtype='double')
          dxOut[d]      = np.zeros(nxOut[d], dtype='double')
          xOut[d][0:]   = np.squeeze(np.multiply(0.5,gridOut[tuple(slices[0])]+gridOut[tuple(slices[1])]))
          dxOut[d][0:]  = np.squeeze(gridOut[tuple(slices[1])]-gridOut[tuple(slices[0])])
          lxOut[d]      = xOut[d][-1]-xOut[d][0]
      else:
        getNodalGrid()
    else:
      getNodalGrid()

  return xOut, dimOut, nxOut, lxOut, dxOut, gridOut

#.Establish the interpolated grid......................................#
def getGrid(dataFile,polyOrder,basisType,**opKey):
  gridFile = 'none' #getGridFileName(dataFile)
  gridType = 'uniform' #getGridType(dataFile)

  vName = 'CartGridField'
  if 'varName' in opKey:
    vName = opKey['varName']

  if (not os.path.exists(gridFile)) or (gridType=='uniform'):
#    print(" Grid file not found. Assuming uniform Cartesian grid.")
    #.Read grid parameters from file, and build grid with pgkyl.
    pgData = pg.GData(dataFile,var_name=vName)                       #.Read data with pgkyl.
    if 'numInterp' in opKey:
      if basisType=='ms':
        pgInterp = pg.GInterpModal(pgData, polyOrder, basisType, opKey['numInterp'])    #.Interpolate data.
      elif basisType == 'ns':
        pgInterp = pg.GInterpNodal(pgData, polyOrder, basisType, opKey['numInterp'])    #.Interpolate data.
    else:
      if basisType=='ms':
        pgInterp = pg.GInterpModal(pgData, polyOrder, basisType)    #.Interpolate data.
      elif basisType == 'ns':
        pgInterp = pg.GInterpNodal(pgData, polyOrder, basisType)    #.Interpolate data.
    xNodal, dataInterp = pgInterp.interpolate()
    dimOut             = len(xNodal)			    #.Number of dimensions in data.

    #.If desired, output cell center values of grid coordinates instead of nodal coordinates.
    if 'location' in opKey:
      if opKey['location']=='center':
        xOut = [[] for i in range(dimOut)]
        for i in range(dimOut): 
          nNodes  = np.shape(xNodal[i])[0]
          xOut[i] = np.zeros(nNodes-1)
          xOut[i] = np.multiply(0.5,xNodal[i][0:nNodes-1]+xNodal[i][1:nNodes])
      else:
        xOut = xNodal
    else:
      xOut = xNodal

    nxOut = np.zeros(dimOut,dtype='int')
    lxOut = np.zeros(dimOut,dtype='double')
    dxOut = np.zeros(dimOut,dtype='double')
    for i in range(dimOut):
      nxOut[i] = np.size(xOut[i])
      lxOut[i] = xOut[i][-1]-xOut[i][0]
      dxOut[i] = xOut[i][ 1]-xOut[i][0]

    gridOut = None

  else:
    #.Read grid from grid file.
    pgData = pg.GData(gridFile,var_name=vName)     #.Read data with pgkyl.
    gridIn = pgData.get_values()     #.This contains the nodal coordinates.

    numNodes = np.shape(gridIn)[0:-1]
    numCells = np.subtract(numNodes,1)
    nxOut    = np.add(np.multiply(numCells,(polyOrder+1)),1)
    dimOut   = np.size(numCells)
    lxOut    = np.zeros(dimOut,dtype='double')
    dxOut    = [[] for i in range(dimOut)]
    xNodal   = [[] for i in range(dimOut)]

    for d in range(dimOut):
      slices     = [np.s_[0:1]]*(dimOut+1)
      slices[d]  = np.s_[0:numNodes[d]]
      slices[-1] = d
      xNodal1D   = np.squeeze(gridIn[tuple(slices)])
      #.Create a grid array with each cell evenly subdivided.
      xNodal[d]  = np.zeros(numCells[d]*(polyOrder+1)+1, dtype='double')
      dxOut[d]   = np.zeros(numCells[d]*(polyOrder+1), dtype='double')
      for i in range(numCells[d]):
        dx = (xNodal1D[i+1]-xNodal1D[i])/(polyOrder+1)
        for j in range(polyOrder+1):
          dxOut[d][i*(polyOrder+1)+j]  = dx
          xNodal[d][i*(polyOrder+1)+j] = xNodal1D[i]+j*dx
      #.Add the last node.
      dx = (xNodal1D[-1]-xNodal1D[-2])/(polyOrder+1)
      xNodal[d][numCells[d]*(polyOrder+1)] = xNodal1D[numCells[d]-1]+(polyOrder+1)*dx

      lxOut[d] = xNodal[d][-1]-xNodal[d][0]

    #.If desired, output cell center values of grid coordinates instead of nodal coordinates.
    if 'location' in opKey:
      if opKey['location']=='center':
        xOut = [[] for i in range(dimOut)]
        for d in range(dimOut): 
          nNodes  = np.shape(xNodal[d])[0]
          xOut[d] = np.zeros(nNodes-1)
          xOut[d] = np.multiply(0.5,xNodal[d][0:nNodes-1]+xNodal[d][1:nNodes])
        nxOut = np.subtract(nxOut, 1)
      else:
        xOut = xNodal
    else:
      xOut = xNodal

    #.Create an interpolated grid.
    gIdx = 'xy'
    if dimOut==1:
      gridOut = np.transpose(xOut)
    elif dimOut==2:
      gridOut = np.stack(np.meshgrid(xOut[0],xOut[1],indexing=gIdx),-1)
    elif dimOut==3:
      gridOut = np.stack(np.meshgrid(xOut[0],xOut[1],xOut[2],indexing=gIdx),-1)
    elif dimOut==4:
      gridOut = np.stack(np.meshgrid(xOut[0],xOut[1],xOut[2],xOut[3],indexing=gIdx),-1)
    elif dimOut==5:
      gridOut = np.stack(np.meshgrid(xOut[0],xOut[1],xOut[2],xOut[3],xOut[4],indexing=gIdx),-1)
    elif dimOut==6:
      gridOut = np.stack(np.meshgrid(xOut[0],xOut[1],xOut[2],xOut[3],xOut[4],xOut[6],indexing=gIdx),-1)

  return xOut, dimOut, nxOut, lxOut, dxOut, gridOut

#.Obtain raw DG data.....................................#
def getRawData(dataFile):
  pgData  = pg.GData(dataFile)     #.Read data with pgkyl.
  dataOut = pgData.get_values()
  return dataOut

#.........................................................#

#.Read data and time stamps from a DynVector..............#
def readDynVector(dataFile):
  pgData = pg.GData(dataFile)  #.Read data with pgkyl.
  time   = pgData.get_grid()    #.Time of the simulation.
  val    = pgData.get_values()
  return np.squeeze(time), np.squeeze(val)
#.........................................................#

#.Evaluate function expanded in 1x basis at certain points.
#.Limited to 0<p<4.
def evalF1x_e(fIn,xE,xcIn,dxIn,pOrderIn,**opKey):
  NxE = np.size(xE)
  fEs = np.zeros(NxE)
  if 'basis' in opKey:
    if opKey['basis'] == 'ns':
      if pOrderIn == 1:
        cellAv = 0.5*(fIn[0]+fIn[1])
        for i in range(NxE):
          fEs[i] = (fIn[1]-fIn[0])*(xE[i]-xcIn)/dxIn+cellAv
    elif opKey['basis'] == 'ms':
      if pOrderIn == 1:
        for i in range(NxE):
          fEs[i] = rsqrt2*fIn[0] + sqrt3d2*fIn[1]*(xE[i]-xcIn)/(0.5*dxIn)
      elif pOrderIn == 2:
        for i in range(NxE):
          xi = (xE[i]-xcIn)/(0.5*dxIn)
          fEs[i] = rsqrt2*fIn[0] + sqrt3d2*xi*fIn[1] + 3.0*sqrt5*rsqrt2Cu*(xi**2-1.0/3.0)*fIn[2]
      elif pOrderIn == 3:
        for i in range(NxE):
          fEs[i] = rsqrt2*fIn[0] + sqrt3d2*xi*fIn[1] + 3.0*sqrt5*rsqrt2Cu*(xi**2-1.0/3.0)*fIn[2] + 5.0*sqrt7*rsqrt2Cu*(xi**3-3.0*xi/5.0)*fIn[3]
  else:
    #.Assume modal serendipity:
    if pOrderIn == 1:
      for i in range(NxE):
        fEs[i] = rsqrt2*fIn[0] + sqrt3d2*fIn[1]*(xE[i]-xcIn)/(0.5*dxIn)
    elif pOrderIn == 2:
      for i in range(NxE):
        xi = (xE[i]-xcIn)/(0.5*dxIn)
        fEs[i] = rsqrt2*fIn[0] + sqrt3d2*xi*fIn[1] + 3.0*sqrt5*rsqrt2Cu*(xi**2-1.0/3.0)*fIn[2]
    elif pOrderIn == 3:
      for i in range(NxE):
        fEs[i] = rsqrt2*fIn[0] + sqrt3d2*xi*fIn[1] + 3.0*sqrt5*rsqrt2Cu*(xi**2-1.0/3.0)*fIn[2] + 5.0*sqrt7*rsqrt2Cu*(xi**3-3.0*xi/5.0)*fIn[3]
  return fEs

#.Evaluate function expanded in 2x basis at certain points.
#.Limited to 0<p<4.
def evalF2x_e(fIn,xE,xcIn,dxIn,pOrderIn,**opKey):
  def evalOnGrid():
    if 'basis' in opKey:
      if opKey['basis'] == 'ns':
        print(" ***** not available yet ***** ")
      elif opKey['basis'] == 'ms':
        if pOrderIn == 1:
          for i in range(NxE[0]):
            for j in range(NxE[1]):
              xi = (xE[0][i]-xcIn[0])/(0.5*dxIn[0])
              xj = (xE[1][j]-xcIn[1])/(0.5*dxIn[1])
              fEs[i,j] = ((3.0*fIn[3]*xi+sqrt3*fIn[2])*xj+sqrt3*fIn[1]*xi+fIn[0])/2.0
        elif pOrderIn == 2:
          for i in range(NxE[0]):
            for j in range(NxE[1]):
              xi = (xE[0][i]-xcIn[0])/(0.5*dxIn[0])
              xj = (xE[1][j]-xcIn[1])/(0.5*dxIn[1])
              fEs[i,j] = ((3.0*sqrt15*fIn[7]*xi+3.0*sqrt5*fIn[5])*xj**2+(3.0*sqrt15*fIn[6]*xi**2+6.0*fIn[3]*xi-sqrt15*fIn[6]+2.0*sqrt3*fIn[2])*xj+3.0*sqrt5*fIn[4]*xi**2+(2.0*sqrt3*fIn[1]-sqrt15*fIn[7])*xi-sqrt5*fIn[5]-sqrt5*fIn[4]+2.0*fIn[0])/4.0
    else:
      #.Assume modal serendipity:
      if pOrderIn == 1:
        for i in range(NxE[0]):
          for j in range(NxE[1]):
            xi = (xE[0][i]-xcIn[0])/(0.5*dxIn[0])
            xj = (xE[1][j]-xcIn[1])/(0.5*dxIn[1])
            fEs[i,j] = ((3.0*fIn[3]*xi+sqrt3*fIn[2])*xj+sqrt3*fIn[1]*xi+fIn[0])/2.0
      elif pOrderIn == 2:
        for i in range(NxE[0]):
          for j in range(NxE[1]):
            xi = (xE[0][i]-xcIn[0])/(0.5*dxIn[0])
            xj = (xE[1][j]-xcIn[1])/(0.5*dxIn[1])
            fEs[i,j] = ((3.0*sqrt15*fIn[7]*xi+3.0*sqrt5*fIn[5])*xj**2+(3.0*sqrt15*fIn[6]*xi**2+6.0*fIn[3]*xi-sqrt15*fIn[6]+2.0*sqrt3*fIn[2])*xj+3.0*sqrt5*fIn[4]*xi**2+(2.0*sqrt3*fIn[1]-sqrt15*fIn[7])*xi-sqrt5*fIn[5]-sqrt5*fIn[4]+2.0*fIn[0])/4.0

  if 'points' in opKey:
    if opKey['points']:
      #.A set of points is being passed in.
      nPoints = np.shape(xE)[0]
      fEs     = np.zeros(nPoints)
      if 'basis' in opKey:
        if opKey['basis'] == 'ns':
          print(" ***** not available yet ***** ")
        elif opKey['basis'] == 'ms':
          if pOrderIn == 1:
            for idx in range(nPoints):
              xi = (xE[idx][0]-xcIn[0])/(0.5*dxIn[0])
              xj = (xE[idx][1]-xcIn[1])/(0.5*dxIn[1])
              fEs[idx] = ((3.0*fIn[3]*xi+sqrt3*fIn[2])*xj+sqrt3*fIn[1]*xi+fIn[0])/2.0
          elif pOrderIn == 2:
            for idx in range(nPoints):
              xi = (xE[idx][0]-xcIn[0])/(0.5*dxIn[0])
              xj = (xE[idx][1]-xcIn[1])/(0.5*dxIn[1])
              fEs[idx] = ((3.0*sqrt15*fIn[7]*xi+3.0*sqrt5*fIn[5])*xj**2+(3.0*sqrt15*fIn[6]*xi**2+6.0*fIn[3]*xi-sqrt15*fIn[6]+2.0*sqrt3*fIn[2])*xj+3.0*sqrt5*fIn[4]*xi**2+(2.0*sqrt3*fIn[1]-sqrt15*fIn[7])*xi-sqrt5*fIn[5]-sqrt5*fIn[4]+2.0*fIn[0])/4.0
      else:
        #.Assume modal serendipity:
        if pOrderIn == 1:
          for idx in range(nPoints):
            xi = (xE[idx][0]-xcIn[0])/(0.5*dxIn[0])
            xj = (xE[idx][1]-xcIn[1])/(0.5*dxIn[1])
            fEs[idx] = ((3.0*fIn[3]*xi+sqrt3*fIn[2])*xj+sqrt3*fIn[1]*xi+fIn[0])/2.0
        elif pOrderIn == 2:
          for idx in range(nPoints):
            xi = (xE[idx][0]-xcIn[0])/(0.5*dxIn[0])
            xj = (xE[idx][1]-xcIn[1])/(0.5*dxIn[1])
            fEs[idx] = ((3.0*sqrt15*fIn[7]*xi+3.0*sqrt5*fIn[5])*xj**2+(3.0*sqrt15*fIn[6]*xi**2+6.0*fIn[3]*xi-sqrt15*fIn[6]+2.0*sqrt3*fIn[2])*xj+3.0*sqrt5*fIn[4]*xi**2+(2.0*sqrt3*fIn[1]-sqrt15*fIn[7])*xi-sqrt5*fIn[5]-sqrt5*fIn[4]+2.0*fIn[0])/4.0
    else:
      #.Assume a evaluation over a grid is requested.
      NxE = [np.size(xE[0]), np.size(xE[1])]
      fEs = np.zeros(NxE)
      evalOnGrid()
  else:
    #.Assume a evaluation over a grid is requested.
    NxE = [np.size(xE[0]), np.size(xE[1])]
    fEs = np.zeros(NxE)
    evalOnGrid()
  return fEs

#.Interpolate DG data.....................................#
def getInterpData(dataFile,polyOrder,basisType,**opKey):
  zs = {"z0" : None, "z1" : None, "z2" : None, "z3" : None, "z4" : None, "z5" : None} 
  if 'pLoad' in opKey:
    for k, v in opKey['pLoad'].items():
      zs[k] = v

  vName = 'CartGridField'
  if 'varName' in opKey:
    vName = opKey['varName']

  pgData = pg.GData(dataFile, z0=zs["z0"], z1=zs["z1"], z2=zs["z2"], z3=zs["z3"], z4=zs["z4"], z5=zs["z5"], var_name=vName)
#  gridFile = getGridFileName(dataFile)
#  gridType = getGridType(dataFile)
  if 'numInterp' in opKey:
    if basisType=='ms':
      pgInterp       = pg.GInterpModal(pgData, polyOrder, basisType, opKey['numInterp'])    #.Interpolate data.
    elif basisType == 'ns':
      pgInterp       = pg.GInterpNodal(pgData, polyOrder, basisType, opKey['numInterp'])    #.Interpolate data.
  else:
    if basisType=='ms':
      pgInterp       = pg.GInterpModal(pgData, polyOrder, basisType)    #.Interpolate data.
    elif basisType == 'ns':
      pgInterp       = pg.GInterpNodal(pgData, polyOrder, basisType)    #.Interpolate data.
  if 'comp' in opKey:
    xOut, dataOut = pgInterp.interpolate(opKey['comp'])
  else:
    xOut, dataOut = pgInterp.interpolate()

  return dataOut

#.Read the time variable from frames frameI to frameF ....#
def getTimeStamps(dataFileRoot,fileExt,frameI,frameF):
  timesOut = np.zeros(frameF-frameI+1)
  for i in range(frameI, frameF+1):
    pgData = pg.GData(dataFileRoot+str(i)+'.'+fileExt)
    timesOut[i] = pgData.ctx['time']
  return timesOut

#.........................................................#
#.Plot cell-wise polynomial for 0<p<4 for 1D, or 0<p<3 for 2D.
def plotLocalPoly(axisIn,xNodal,fIn,pIn,**opKey):
  dim          = len(xNodal)

  if pIn==1:
    if dim==1:
      nLocPoints1D = pIn+1
    else:
      nLocPoints1D = 1*pIn+1
  else:
    nLocPoints1D = 8*pIn+1


  cellsN       = 1                #.Total number of cells.
  nCells       = np.zeros(dim, dtype='int')    #.Number of cells along each direction.
  for d in range(dim):
    cellsN    = cellsN*(np.size(xNodal[d])-1)
    nCells[d] = np.size(xNodal[d])-1

  hpOut = [0]*cellsN  #.Handle to plots outputted.
  dxLoc = [0]*dim
  xcLoc = [0]*dim
  xLoc  = [0]*dim
  ix    = np.zeros(dim, dtype='int')  #.Multidimensional index.
  for idx in range(cellsN):
    
    #.Assume column-major order (x is the fastest varying coordinate).
    ix[0] = np.mod(idx,nCells[0]).astype('int')
    for d in range(1,dim):
      ix[d] = ((idx - ix[d-1])/np.prod(nCells[:d])).astype('int')

    for d in range(dim):
      dxLoc[d] = xNodal[d][ix[d]+1]-xNodal[d][ix[d]]
      xcLoc[d] = 0.5*(xNodal[d][ix[d]+1]+xNodal[d][ix[d]])
      xLoc[d]  = [xNodal[d][ix[d]]]
      for p in range(nLocPoints1D-1):
        xLoc[d].append(xNodal[d][ix[d]]+float(p+1)*dxLoc[d]/float(nLocPoints1D-1))
    
    if dim==1:
      if 'basis' in opKey:
        yLoc  = evalF1x_e(fIn[ix[0]],xLoc[0],xcLoc[0],dxLoc[0],pIn,basis=opKey['basis'])
      else:
        yLoc  = evalF1x_e(fIn[ix[0]],xLoc[0],xcLoc[0],dxLoc[0],pIn)
  
      if 'lines' in opKey:
          opKey['lines'][idx].set_data(xLoc[0],yLoc)
      else:
        if 'color' in opKey:
          if 'linestyle' in opKey:
            hpOut[idx], = axisIn.plot(xLoc[0], yLoc, color=opKey['color'], linestyle=opKey['linestyle'])
          else:
            hpOut[idx], = axisIn.plot(xLoc[0], yLoc, color=opKey['color'], linestyle='--')
        else:
          if 'linestyle' in opKey:
            hpOut[idx], = axisIn.plot(xLoc[0], yLoc, linestyle=opKey['linestyle'])
          else:
            hpOut[idx], = axisIn.plot(xLoc[0], yLoc)
    elif dim==2:
      if 'basis' in opKey:
        zLoc  = evalF2x_e(fIn[ix[0],ix[1]],xLoc,xcLoc,dxLoc,pIn,basis=opKey['basis'])
      else:
        zLoc  = evalF2x_e(fIn[ix[0],ix[1]],xLoc,xcLoc,dxLoc,pIn)

      #.Create mesh for 2D color plot.
      Xloc = np.outer(xLoc[0], np.ones(np.shape(xLoc[1])))
      Yloc = np.outer(np.ones(np.shape(xLoc[0])), xLoc[1])
      
      if 'colorplots' in opKey:
          opKey['colorplots'][idx].set_array(zLoc.ravel())
      else:
        if 'cmap' in opKey:
          hpOut[idx] = axisIn.pcolormesh(Xloc, Yloc, zLoc, cmap=opKey['cmap'])
        else:
          hpOut[idx] = axisIn.pcolormesh(Xloc, Yloc, zLoc)

      if 'clims' in opKey:
        hpOut[idx].set_clim(opKey['clims'][0],opKey['clims'][1])

  if ('lines' not in opKey) or ('colorplots' not in opKey):
    print(" Number of piecewise polynomials plotted: ",cellsN)
    return hpOut

#.........................................................#
#.Plot value of local polynomial at specific points for 0<p<4 in 1D plot.
#.The points are given in plotX, and these are assumed to be defined on
#.the computational cell space [-1,1].
def plotLocalPoints(axisIn,xNodal1D,fIn,pIn,plotX,**opKey):
  hpOut = [0]*(np.size(xNodal1D)-1)
  for i in range(np.size(xNodal1D)-1):    #.Loop over cells.
    dxLoc = xNodal1D[i+1]-xNodal1D[i]
    xcLoc = 0.5*(xNodal1D[i+1]+xNodal1D[i])
    xLoc  = xcLoc + 0.5*dxLoc*plotX
    if 'basis' in opKey:
      yLoc  = evalF1x_e(fIn[i],xLoc,xcLoc,dxLoc,pIn,basis=opKey['basis'])
    else:
      yLoc  = evalF1x_e(fIn[i],xLoc,xcLoc,dxLoc,pIn)
    isNegative = False
    for j in range(len(xLoc)):
      if (yLoc[j] < 0.0):
        isNegative = True
    if isNegative:
      print(" Negative control point at x = ", xLoc)

    if 'points' in opKey:
        opKey['points'][i].set_data(xLoc,yLoc)
    else:
      if 'color' in opKey:
        hpOut[i], = axisIn.plot(xLoc, yLoc, color=opKey['color'], linestyle='None', marker='o')
      else:
        hpOut[i], = axisIn.plot(xLoc, yLoc)

  if 'points' not in opKey:
    return hpOut


#............... NODAL OPERATIONS AND FEM ..................#

#.Serendipity nodes (0<p<4).
nodesSer1x = [ [[-1],[1]], \
               [[-1],[0],[1]], \
               [[-1],[-1./3.],[1./3.],[1]] ]
nodesSer2x = [ [[-1,-1],[1,-1],[-1,1],[1,1]], \
               [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]], \
               [[-1,-1],[-1./3.,-1],[1./3.,-1],[1,-1],[-1,-1./3.],[1,-1./3.],[-1,1./3.],[1,1./3.],[-1,1],[-1./3.,1],[1./3.,1],[1,1]] ]
#.Tensor product nodes.
nodesTensor1x = [ [[-1],[1]], \
                  [[-1],[0],[1]], \
                  [[-1],[-1./3.],[1./3.],[1]] ]
nodesTensor2x = [ [[-1,-1],[1,-1],[-1,1],[1,1]], \
                  [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]], \
                  [[-1,-1],[-1./3.,-1],[1./3.,-1],[1,-1],[-1,-1./3.],[-1./3.,-1./3.],[1./3.,-1./3.],[1,-1./3.],[-1,1./3.],[-1./3.,1./3.],[1./3.,1./3.],[1,1./3.],[-1,1],[-1./3.,1],[1./3.,1],[1,1]] ]

#.Read Gkeyll CartField as FEM data.
def getRawDataFEM(dataFile):
  pgData        = pg.GData(dataFile)     #.Read data with pgkyl.
  cartFieldData = pgData.get_values()

  basisType = getBasisType(dataFile)
  pOrder    = getPolyOrder(dataFile)
  nCells    = getNumCells(dataFile)
  dim       = np.size(nCells)

  dataOut   = np.empty([])  #.Function values at node locations.

  if dim==1:
    #.Do the nodes contained in all cells first.
    for i in range(nCells[0]):
      for k in range(1+(pOrder-1)*dim):
        dataOut = np.append(dataOut,cartFieldData[i,k])
    #.The last cell also has the nodes on the boundary.
    dataOut = np.append(dataOut,cartFieldData[-1,pOrder])
  elif dim==2:
    if basisType == "Ser":
      numCellNodesI    = 1+(pOrder-1)*dim          #.Number of nodes in interior cells.
      numCellNodesUx   = numCellNodesI+pOrder      #.Number of nodes in upper x-boundary cells.
      numCellNodesUy   = numCellNodesI+pOrder      #.Number of nodes in upper y-boundary cells.
      numCellNodesUxUy = numCellNodesI+2*pOrder+1  #.Number of nodes in upper right corner cell.
    elif basisType == "Tensor":
      numCellNodesI    = pOrder**dim               #.Number of nodes in interior cells.
      numCellNodesUx   = numCellNodesI+pOrder      #.Number of nodes in upper x-boundary cells.
      numCellNodesUy   = numCellNodesI+pOrder      #.Number of nodes in upper y-boundary cells.
      numCellNodesUxUy = numCellNodesI+2*pOrder+1  #.Number of nodes in upper right corner cell.

    ix = np.zeros(dim, dtype='int')  #.Multidimensional index.
    for idx in range(np.prod(nCells)):
      #.Assume column-major order (x is the fastest varying coordinate).
      ix[0] = np.mod(idx,nCells[0]).astype('int')
      for d in range(1,dim):
        ix[d] = ((idx - ix[d-1])/np.prod(nCells[:d])).astype('int')

      if (ix[0] < (nCells[0]-1)) and (ix[1] < (nCells[1]-1)):
        numCellNodes = numCellNodesI     #.Interior/lower left cells.
      elif (ix[0] == (nCells[0]-1)) and (ix[1] == (nCells[1]-1)):
        numCellNodes = numCellNodesUxUy  #.Upper right corner cell.
      elif (ix[1] < (nCells[1]-1)):
        numCellNodes = numCellNodesUx    #.Upper x-boundary cells.
      elif (ix[0] < (nCells[0]-1)):
        numCellNodes = numCellNodesUy    #.Upper y-boundary cells.

      for k in range(numCellNodes):
        dataOut = np.append(dataOut,cartFieldData[ix[0],ix[1],k])


  dataOut = np.delete(dataOut,0)  #.Because the initialization of dataOut used np.empty.

  return dataOut

#.Compute the node coordinates of the FEM grid.
def getNodeCoordsFEM(basisType,pOrder,nCells,lower,upper):
  dim  = np.size(nCells)
  dx   = (np.array(upper)-np.array(lower))/np.array(nCells)
  xOut = list()  #.Coordinates of node locations.

  if dim==1:
    nodes = nodesSer1x[pOrder-1]
  elif dim==2:
    if basisType == "Ser":
      nodes = nodesSer2x[pOrder-1]
    elif basisType == "Tensor":
      nodes = nodesTensor2x[pOrder-1]

  if dim==1:
    #.Do the nodes contained in all cells first.
    for i in range(nCells[0]):
      xc = lower[0]+(0.5*dx[0])*(2*i+1)
      for k in range(np.size(nodes)-1):
        xOut.append([xc+(0.5*dx[0])*nodes[k][0]])
    #.The last cell also has the nodes on the boundary.
    xc = lower[0]+(0.5*dx[0])*(2*(nCells[0]-1)+1)
    xOut.append([xc+(0.5*dx[0])*nodes[-1][0]])
  elif dim==2:
    ix = np.zeros(dim, dtype='int')  #.Multidimensional index.
    for idx in range(np.prod(nCells)):
      #.Assume column-major order (x is the fastest varying coordinate).
      ix[0] = np.mod(idx,nCells[0]).astype('int')
      for d in range(1,dim):
        ix[d] = ((idx - ix[d-1])/np.prod(nCells[:d])).astype('int')

      if (ix[0] < (nCells[0]-1)) and (ix[1] < (nCells[1]-1)):
        locNodes = [nod for nod in nodes if (nod[0]<1 and nod[1]<1)] #.Interior/lower left cells.
      elif (ix[0] == (nCells[0]-1)) and (ix[1] == (nCells[1]-1)):
        locNodes = nodes   #.Upper right corner cell.
      elif (ix[1] < (nCells[1]-1)):
        locNodes = [nod for nod in nodes if (nod[1]<1)]   #.Upper x-boundary cells.
      elif (ix[0] < (nCells[0]-1)):
        locNodes = [nod for nod in nodes if (nod[0]<1)]   #.Upper y-boundary cells.

      xc = [lower[0]+(0.5*dx[0])*(2*ix[0]+1), lower[1]+(0.5*dx[1])*(2*ix[1]+1)]
      for k in range(np.shape(locNodes)[0]):
        xOut.append([xc[0]+(0.5*dx[0])*locNodes[k][0], xc[1]+(0.5*dx[1])*locNodes[k][1]])

  return xOut

#.Compute the node coordinates of the FEM data stored in a Gkeyll CartField.
def getNodeCoordsFEMfromFile(dataFile):
  basisType = getBasisType(dataFile)
  pOrder    = getPolyOrder(dataFile)
  nCells    = getNumCells(dataFile)
  lower     = getLowerBounds(dataFile)
  upper     = getUpperBounds(dataFile)

  return getNodeCoordsFEM(basisType,pOrder,nCells,lower,upper)

#.Compute the node coordinates of the FEM grid.
def evDGatNodes(fIn,basisType,pOrder,nCells,lower,upper):
  dim     = np.size(nCells)
  dx      = (np.array(upper)-np.array(lower))/np.array(nCells)
  dataOut = np.empty([])  #.Function values at node locations.

  if dim==1:
    nodes = nodesSer1x[pOrder-1]
  elif dim==2:
    if basisType == "Ser":
      nodes = nodesSer2x[pOrder-1]
    elif basisType == "Tensor":
      nodes = nodesTensor2x[pOrder-1]

  if dim==1:
    #.Do the nodes contained in all cells first.
    for i in range(nCells[0]):
      xc = lower[0]+(dx[0]/2)*(2*i+1)
      xN = list()
      for k in range(np.size(nodes)-1):
        xN.append(xc+(0.5*dx[0])*nodes[k][0])
      dataOut = np.append(dataOut,evalF1x_e(fIn[i],xN,xc,dx[0],pOrder))
    #.The last cell also has the nodes on the boundary.
    xc = lower[0]+(0.5*dx[0])*(2*(nCells[0]-1)+1)
    xN = [xc+(0.5*dx[0])*nodes[-1][0]]
    dataOut = np.append(dataOut,evalF1x_e(fIn[-1],xN,xc,dx[0],pOrder))
  elif dim==2:
    ix = np.zeros(dim, dtype='int')  #.Multidimensional index.
    for idx in range(np.prod(nCells)):
      #.Assume column-major order (x is the fastest varying coordinate).
      ix[0] = np.mod(idx,nCells[0]).astype('int')
      for d in range(1,dim):
        ix[d] = ((idx - ix[d-1])/np.prod(nCells[:d])).astype('int')

      if (ix[0] < (nCells[0]-1)) and (ix[1] < (nCells[1]-1)):
        locNodes = [nod for nod in nodes if (nod[0]<1 and nod[1]<1)] #.Interior/lower left cells.
      elif (ix[0] == (nCells[0]-1)) and (ix[1] == (nCells[1]-1)):
        locNodes = nodes   #.Upper right corner cell.
      elif (ix[0] == (nCells[0]-1)) and (ix[1] < (nCells[1]-1)):
        locNodes = [nod for nod in nodes if (nod[1]<1)]   #.Upper x-boundary cells.
      elif (ix[0] < (nCells[0]-1)) and (ix[1] == (nCells[1]-1)):
        locNodes = [nod for nod in nodes if (nod[0]<1)]   #.Upper y-boundary cells.

      xc = [lower[0]+(0.5*dx[0])*(2*ix[0]+1), lower[1]+(0.5*dx[1])*(2*ix[1]+1)]
      xN = []
      for k in range(np.shape(locNodes)[0]):
        xN.append([xc[0]+(0.5*dx[0])*locNodes[k][0], xc[1]+(0.5*dx[1])*locNodes[k][1]])
      dataOut = np.append(dataOut,evalF2x_e(fIn[ix[0],ix[1]],xN,xc,dx,pOrder,points=True))

  dataOut = np.delete(dataOut,0)  #.Because the initialization of dataOut used np.empty.

  return dataOut

#............... END of NODAL OPERATIONS AND FEM ..................#


#.Return a variable name to put on the figure.............# 
def commonVarName(fileVarName,**opKey):
  fVnameSplit = fileVarName.split("_")
  if len(fVnameSplit)>1:
    species = fVnameSplit[0]
    var     = fVnameSplit[1]
  else:
    var     = fVnameSplit[0]

  if var == 'GkM0':
    if species == 'electron':
      varStrOut = 'n_e'
    else:
      varStrOut = 'n_i'
    unitStrOut  = ' (m$^{-3}$)'
  if var == 'GkM1':
    if species == 'electron':
      varStrOut = r'$n_eu_{\parallel,e}$'
    else:
      varStrOut = r'$n_eu_{\parallel,i}$'
    unitStrOut  = ' (m$^{-2}$/s)'
  if var == 'uPar':
    if species == 'electron':
      varStrOut = r'$u_{\parallel,e}$'
    else:
      varStrOut = r'$u_{\parallel,i}$'
    unitStrOut  = ' (m/s)'
  elif var == 'vthSq':
    if species == 'electron':
      varStrOut = 'T_e'
    else:
      varStrOut = 'T_i'
    unitStrOut  = ' (eV)'
  elif var == 'phi':
    varStrOut  = 'phi'
    unitStrOut = ' (V)'
  elif var == 'field':
    if opKey['comp']==0:
      varStrOut  = 'E_x'
      unitStrOut = ''
    elif opKey['comp']==1:
      varStrOut  = 'E_y'
      unitStrOut = ''
    elif opKey['comp']==2:
      varStrOut  = 'E_z'
      unitStrOut = ''
    elif opKey['comp']==3:
      varStrOut  = 'B_x'
      unitStrOut = ''
    elif opKey['comp']==4:
      varStrOut  = 'B_y'
      unitStrOut = ''
    elif opKey['comp']==5:
      varStrOut  = 'B_z'
      unitStrOut = ''

  return varStrOut, unitStrOut

#.........................................................#
#.This function reads the time average if it is already computed
#.and stored in a file, or computes a new one (and stores it in
#.a file if saveAv=True).
def getTimeAv(dataDir,simName,varName,iFrame,fFrame,p,b,saveAv,tAvDir):
  #.Check or create post data directory.
  checkMkdir(tAvDir)
  #.Check if time average file already exists.
  tAvFile = tAvDir+simName+'_'+varName+'_TimeAv'+str(iFrame)+'-'+str(fFrame)+'.bp' 
  if not os.path.isfile(tAvFile):
    #.Compute time average and store it in new file.
    fileName = dataDir+simName+'_'+varName+'_%d.bp'
    x, gridDim, nx, lx, dx = getGrid(fileName % iFrame,p,b,location='center')

    q0AvT = np.zeros(nx)
    for nFr in range(iFrame,fFrame+1):
      #.Read 3D data into q0.
      q0AvT = np.add(q0AvT,np.squeeze(getInterpData(fileName % nFr,p,b)))

    q0AvT = np.divide(q0AvT,float(fFrame-iFrame+1))

    if saveAv:
      #.Save time average to a file for reuse.
      print(" ")
      print(" Saving time average in "+tAvFile+" ...")
      #.Function to write DG coefficients to Gkeyll-style ADIOS file.
      sNumCells  = ""
      sOffsets   = ""
      for i in range(np.size(nx)):
        sNumCells += "{:d},".format(int(nx[i]))
        sOffsets  += "0,"
      #.ADIOS init.
      ad.init_noxml()
      ad.set_max_buffer_size(1000)
      groupId = ad.declare_group("CartFieldInterp", "")
      ad.select_method(groupId, "POSIX1", "", "")
      #.Define variables and attributes.
      ad.define_attribute_byvalue(groupId, "numCells", "", nx)
      lo = np.zeros(np.size(nx), dtype='double')
      up = np.zeros(np.size(nx), dtype='double')
      for i in range(np.size(nx)):
        lo[i], up[i] = x[i][0], x[i][-1]
      ad.define_attribute_byvalue(groupId, "lowerBounds", "", lo)
      ad.define_attribute_byvalue(groupId, "upperBounds", "", up)
      ad.define_var(groupId, "CartGridFieldInterpTimeAv", "",
              ad.DATATYPE.double,
              sNumCells, sNumCells, sOffsets)
      fh = ad.open("CartFieldInterp", tAvFile, 'w')
      ad.write(fh, "CartGridFieldInterpTimeAv", q0AvT)
      ad.close(fh)
      ad.finalize()
      #.Deal with weird file output where a '.bp.0' file is created.
      if len(tAvFile.split('/')) > 1:
          nm = tAvFile.split('/')[-1]
      else:
          nm = tAvFile
      shutil.move(tAvFile + '.dir/' + nm + '.0', tAvFile)
      shutil.rmtree(tAvFile + '.dir')
  else:
    #.Read time average from existent file.
    print(" ")
    print(" Reading time average in "+tAvFile+" ...")
    hF    = ad.file(tAvFile)
    q0AvT = hF['CartGridFieldInterpTimeAv'].read()
    hF.close()

  return q0AvT
  
#.Set minimum and maximum values in an array..............#
def setMinMax(aIn,minIn,maxIn):
  if np.amin(aIn)<minIn:
    minOut = np.amin(aIn)
  else:
    minOut = minIn
  if np.amax(aIn)>maxIn:
    maxOut = np.amax(aIn)
  else:
    maxOut = maxIn
  return minOut, maxOut

#.Derivative along X......................................#
def derX(aIn,dx,xBC,acc):
#.aIn: 2D field.
#.xBC: integer indicating boundary condition along X.
#.acc: accuracy, 2 for 2nd order, 4 for 4th order.
  s0 = aIn.shape[0]
  s1 = aIn.shape[1]
  ax = np.zeros((s0, s1))
  rdxd2  = 1.0/(2.0*dx)
  rdx2d3 = 2.0/(3.0*dx)
  rdxd12 = 1.0/(12.0*dx)

  aBu = np.zeros((s0+acc, s1+acc))
  aBu[2:s0+2,2:s1+2] = aIn
  if xBC == 1:
    #.Even symmetry.
    aBu[2:s0+2,0]    = aIn[:,3]
    aBu[2:s0+2,1]    = aIn[:,2]
    aBu[2:s0+2,s1+2] = aIn[:,s1-2]
    aBu[2:s0+2,s1+3] = aIn[:,s1-3]
  elif xBC == -1:
    #.Odd symmetry.
    aBu[2:s0+2,0]    = -aIn[:,3]
    aBu[2:s0+2,1]    = -aIn[:,2]
    aBu[2:s0+2,s1+2] = -aIn[:,s1-2]
    aBu[2:s0+2,s1+3] = -aIn[:,s1-3]

  for j in range(0,s0):
    for i in range(0,s1):
      ax[j,i]=(aBu[j+2,i+3]-aBu[j+2,i+1])*rdx2d3-(aBu[j+2,i+4]-aBu[j+2,i])*rdxd12
  return ax

#.Derivative along Y......................................#
def derY(aIn,dy,yBC,acc):
#.aIn: 2D field.
#.yBC: integer indicating boundary condition along Y.
#.acc: accuracy, 2 for 2nd order, 4 for 4th order.
  s0   = aIn.shape[0]
  s1   = aIn.shape[1]
  ay   = np.zeros([s0, s1])
  rdyd2  = 1.0/(2.0*dy)
  rdy2d3 = 2.0/(3.0*dy)
  rdyd12 = 1.0/(12.0*dy)

  aBu = np.zeros([s0+acc, s1+acc])
  aBu[2:s0+2,2:s1+2] = aIn
  if yBC == 9:
    #.Periodic.
    aBu[0,2:s1+2]    = aIn[s0-3,:]
    aBu[1,2:s1+2]    = aIn[s0-1,:]
    aBu[s0+2,2:s1+2] = aIn[0,:]
    aBu[s0+3,2:s1+2] = aIn[1,:]

  for j in range(0,s0):
    for i in range(0,s1):
      ay[j,i]=(aBu[j+3,i+2]-aBu[j+1,i+2])*rdy2d3-(aBu[j+4,i+2]-aBu[j,i+2])*rdyd12
  return ay

#.Average all of the dimIn dimension......................#
def avAllDim(fIn,dimIn):
  #.fIn: 2D field.
  #.dimIn: dimension to be averaged.
  fAv = np.mean(fIn, axis=dimIn)
  return fAv

