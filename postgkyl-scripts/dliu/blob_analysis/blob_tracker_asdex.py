##### Movie of Electric field and density contour:
#https://jychoi-hpc.github.io/adios-python-docs/quick.html
#import adios as ad
import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from random import seed
from random import random
from time import sleep
from tqdm import tqdm
from shapely import geometry
from PIL import Image
import os

seed(99999999) # for generation of reproducible sequences of random numbers

#========== User Inputs ==========
show_anim = True
save_anim = False
euler = np.exp(1)

fstart = 80
fend = 351
dt = 1

dataDir = '/scratch/gpfs/dingyunl/gk-g0-app-gpu-hack2024-positivity/gkylzero/'
outDir = dataDir + 'blobData2'
simName = 'gk_bgk_im_asdex_extend_2xIC_3x2v_p1'

polyOrder = 1
basisType = 'ms'


#========== Physical Constants ==========
eV = 1.602176487e-19  # [C]
me = 9.1093837015e-31 # [kg]
mp = 1.672621637e-27  # [kg]
mi = 2.014*mp         # [kg]


#========== Blob Data Directory Setup ==========
if os.path.exists(outDir):
    checkFile = input('Dir ' + outDir + ' exists. Overwrite? [y/n] ')
    if checkFile == 'y':
        os.system('rm -rf '+outDir)
        os.system('mkdir '+outDir)
    else:
        exit()
else:
    os.system('mkdir '+outDir)


#========== Cell Info ==========
data_num = np.arange(start=fstart, stop=fend, step=dt, dtype=int)
data = pg.GData(dataDir+simName+'-field_%d'%data_num[0]+'.gkyl')
#print("t=%f\n" % data.ctx["time"])
grid = data.get_grid()

blob_size_file = open(outDir+"/blob_size.txt", "w")

Nx = len(grid[0]) - 1   # number of cells
Ny = len(grid[1]) - 1
Nz = len(grid[2]) - 1

xMin = grid[0][0]
yMin = grid[1][0]
zMin = grid[2][0]

xMax = grid[0][-1]
yMax = grid[1][-1]
zMax = grid[2][-1]

dx = (xMax - xMin) / Nx
dy = (yMax - yMin) / Ny

Lx = xMax - xMin
x_start = 10 # int(Nx*2*(x_source-xMin)/Lx)
x_end = -12 # int(Nx*2*(xMax-0.168)/Lx)

#print("Nx=%d, Ny=%d, Nz=%d\n" % (Nx, Ny, Nz))
#print("xMin=%10.8e, yMin=%10.8e, zMin=%10.8e\n" % (xMin, yMin, zMin))


#========== Plot Settings ==========
lblSize        = 14
ttlSize        = 16
tickSize       = 12
txtSize        = 14

cnum = 100 # number of levels for contourf
cnumout = 40 # number of levels for contour
color = 'bwr' # diverging
vmin = -2.0e19
vmax = 2.0e19

#========== Function Definition ==========
def Random_Points_in_Bounds(polygon, number):   
    minx, miny, maxx, maxy = polygon.bounds
    x = np.random.uniform( minx, maxx, number )
    y = np.random.uniform( miny, maxy, number )
    return x, y

def func_data(sp, fr, phiInt):
    spFile=dataDir+simName+'-'+sp+'_MaxwellianMoments_%d'%fr+'.gkyl'
    spData = pg.data.GData(spFile)
    spInterp = pg.GInterpModal(spData, polyOrder, basisType)    
    xInt, spDenInt = spInterp.interpolate(0)
    xInt, spTempInt = spInterp.interpolate(2)

    if sp=='elc':
        spTempInt = spTempInt * me / eV
    elif sp=='ion':
        spTempInt = spTempInt * mi / eV
    else:
        assert('Exit due to unknown species!')

    EyInt = - np.gradient(phiInt, dy , axis=1)
   
    # get cell center coordinates
    CCC = []
    for j in range(len(xInt)):
        CCC.append((xInt[j][1:] + xInt[j][:-1])/2)

    x = CCC[0][x_start:x_end]
    y = CCC[1]
    z = CCC[2]
    z_slice = len(z) // 4  # slice at outboard midplane
    #print('zLen=%d, z_slice=%d' % (len(z), z_slice))
    X, Y = np.meshgrid(x, y)
    spDen = np.transpose(spDenInt[x_start:x_end, : , z_slice , 0])
    spTemp = np.transpose(spTempInt[x_start:x_end, : , z_slice , 0])
    Ey = np.transpose(EyInt[x_start:x_end, : , z_slice , 0])
    #print(np.shape(spDenInt), np.shape(spDen))
   
    del spData
    #return x, y, X, Y, spDen, spTemp, Ey
    return X, Y, spDen, spTemp, Ey


#========== Blob Analysis ==========
# Determine threshold density from standard deviation
blobThreshold = []
DenAveArray = []
sigmaArray = []
dp_p = []
# calculate ave density
for i in range(len(data_num)):
#for i in [0, 1, 2]:
    phiFile=dataDir+simName+'-field_%d'%data_num[i]+'.gkyl'
    phiData = pg.data.GData(phiFile)
    phiInterp = pg.GInterpModal(phiData, polyOrder, basisType)    
    xInt, phiInt = phiInterp.interpolate()
    
    #x, y, X, Y, elcDen, elcTemp, Ey = func_data('elc', data_num[i], phiInt)
    #x, y, X, Y, ionDen, ionTemp, Ey = func_data('ion', data_num[i], phiInt)
    X, Y, elcDen, elcTemp, Ey = func_data('elc', data_num[i], phiInt)
    X, Y, ionDen, ionTemp, Ey = func_data('ion', data_num[i], phiInt)

    elcDenAve = np.mean(elcDen, axis=1, keepdims=True)
    dn = elcDen - elcDenAve
    sigma = np.sqrt(np.mean(dn**2, axis=1, keepdims=True))
    #blobThreshold.append(elcDenAve + 2.*sigma)
    blobThreshold.append(elcDenAve + sigma)
    
    pressure = elcDen*(elcTemp+ionTemp)
    p0 = np.mean(pressure)
    dp = np.amax(pressure) - p0
    dp_p.append(dp/(dp + p0))
    
#print('X, Y shape:', np.shape(X)i, '\n')
blobThreshold = np.array(blobThreshold)
dp_p = np.mean(np.array(dp_p))
#print('dp/p = %10.8e\n' % dp_p)

pstep = 1
pbar = tqdm(total=len(data_num))

prevPoly = []
total_blobs = 0
allBlobPoly = []
allBlobXHW = [] # XHW: half width in x direction
noBlobs = False
oneTimeBlobs = 0
for i in range(len(data_num)):
#for i in [0, 1, 2]:
    fig = plt.figure(figsize=(4.5, 4.5))
    ax1 = fig.add_axes([0.18, 0.12, 0.72, 0.82])
    plt.rcParams["font.size"] = "12"
    blob_counter = 0

    phiFile=dataDir+simName+'-field_%d'%data_num[i]+'.gkyl'
    phiData = pg.data.GData(phiFile)
    phiInterp = pg.GInterpModal(phiData, polyOrder, basisType)    
    xInt, phiInt = phiInterp.interpolate()
    
    #x, y, X, Y, elcDen, elcTemp, Ey = func_data('elc', data_num[i], phiInt)
    X, Y, elcDen, elcTemp, Ey = func_data('elc', data_num[i], phiInt)
    nDiff = elcDen - blobThreshold[i]
    
    #Nx = len(x)
    #Ny = len(y)
    Nx, Ny = np.shape(X)[1], np.shape(X)[0]
    #print('Nx=%d, Ny=%d' % (Nx, Ny))

    ax1.cla()
    ax1.set_title('Time = %d'%data_num[i]+' $\mu s$', fontsize=ttlSize)
    ax1.set_xlabel(r'$x$', fontsize=lblSize)
    ax1.set_ylabel(r'$y$', fontsize=lblSize)
    thresholdDensity = 0 #blobThreshold[i]
    #print(thresholdDensity)
    #meanSigma = np.mean(sigmaArray[i])

    cp1 = ax1.contourf(X, Y, nDiff, levels=np.linspace(vmin, vmax, 41), norm=colors.CenteredNorm(), cmap=color)
    cp3 = ax1.contour(X, Y, nDiff, levels=np.linspace(vmin, vmax, 41), linewidths=0.1, colors='black', linestyles='solid') 
    #cp4 = ax1.contour(X, Y, nDiff, [thresholdDensity, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.0], linewidths=1, colors='black', linestyles='solid')
    cp4 = ax1.contour(X, Y, nDiff, [thresholdDensity], linewidths=1, colors='black', linestyles='solid')
    cbar = fig.colorbar(cp1)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r'$n_e-(\langle n_e\rangle+\sigma_{n_e})(m^{-3})$', rotation=270, labelpad=18, fontsize=lblSize)
    hmag = cbar.ax.yaxis.get_offset_text().set_size(12)
    
    #ax1.set_aspect('equal')
    #plt.show()

    # detect for closed contours
    closed_contours = []
    for path in cp4.allsegs:
        for single_path in path:
            #print(np.shape(single_path))
            x = single_path[:,0]
            y = single_path[:,1]
            x_min = np.min(x)
            x_max = np.max(x)
            y_min = np.min(y)
            y_max = np.max(y)
            blobMidX = (x_min + x_max)/2
            blobMidY = (y_min + y_max)/2
            blobLimX = abs(x_max - x_min)
            blobLimY = abs(y_max - y_min)
            # check if the first point is the same as the last (closed contour) and if the contour is larger than grid spacing
            if np.allclose(single_path[0], single_path[-1]) and blobLimX>dx/2 and blobLimY>dy/2:
                closed_contours.append(single_path)
    #print(f"Found {len(closed_contours)} closed contours.")

    blobs = []
    # filter out holes
    for closed_path in closed_contours:
        x = closed_path[:,0]
        y = closed_path[:,1]
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        blobMidX = (x_min + x_max)/2
        blobMidY = (y_min + y_max)/2
        blobLimX = abs(x_max - x_min)
        blobLimY = abs(y_max - y_min)
        allBlobXHW.append(blobLimX/2)
        pointList = []
        for plgn in range(len(x)):
            pointList.append(geometry.Point(x[plgn],y[plgn]))
        p = geometry.Polygon([[po.x, po.y] for po in pointList])
        polyArea = p.area
        checkCounter = 0
        # check density inside blob
        xR,yR = Random_Points_in_Bounds(p, 100)
        test_points = list(zip(xR,yR))
        pointInside = False
        testDensity = []
        for k in range(len(test_points)):
            xT = test_points[k][0]
            yT = test_points[k][1]
            pT = geometry.Point(xT,yT)
            if p.contains(pT):
                pointInside = True
                xd = abs(X[0]-xT)
                yd = abs(Y[:,0]-yT)
                idx = np.where(xd <= 0.25*dx)[0][0]
                idy = np.where(yd <= 0.25*dy)[0][0]
                testDensity.append(nDiff[idy,idx])
        meanTestDens = np.mean(np.array(testDensity))
        if ( meanTestDens >= thresholdDensity):
            isBlob = True
            blobs.append(closed_path)
            blob_counter += 1
        else :
            print('hole detected...')
    #print('Number of blobs: %d' % blob_counter)
    #print(np.shape(blobs))

    allBlobPolyCurr = allBlobPoly
    for j in range(blob_counter):
        x = blobs[j][:,0]
        y = blobs[j][:,1]
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        blobMidX = (x_min + x_max)/2
        blobMidY = (y_min + y_max)/2
        blobLimX = abs(x_max - x_min)
        blobLimY = abs(y_max - y_min)
        allBlobXHW.append(blobLimX/2)
        pointList = []
        for plgn in range(len(x)):
            pointList.append(geometry.Point(x[plgn],y[plgn]))
        p = geometry.Polygon([[po.x, po.y] for po in pointList])
        polyArea = p.area
        checkCounter = 0
        #print("Area of blob%d: %10.8e" % (j+1, polyArea))

        # Blob identification scheme -- NEEDS IMPROVEMENT
        if i == 0 or noBlobs:
            blobID = j + total_blobs
            tf = 0
            allBlobPolyCurr.append((p,blobID,i,tf))
            noBlobs = False
        else:
            # check current blob against previous blobs for ID
            for k in range(len(allBlobPoly)):
                # previous polygon for comparison
                q = allBlobPoly[k][0]
                areaFrac1 = q.intersection(p).area/q.area
                areaFrac2 = q.intersection(p).area/p.area
                framenum = allBlobPoly[k][2]
                if areaFrac1 > 0.1 and areaFrac2 > 0.1 and framenum == i-1:
                    blobID = allBlobPoly[k][1]
                    tf = allBlobPoly[k][3]+1
                    allBlobPolyCurr[k] = (p,blobID,i,tf)
                    ax1.plot(*q.exterior.xy)
                else:
                    checkCounter += 1               
            if checkCounter == len(allBlobPoly):
                # this is a new blob
                blobID = len(allBlobPoly)+total_blobs
                tf = 0
                allBlobPolyCurr.append((p,blobID,i,tf))
        ax1.text(blobMidX,blobMidY,'%d'%blobID,color='red', fontsize=txtSize)
        #blob_counter = blob_counter + 1
        blob_size_file.write('%d'%data_num[i]+'\t%d'%blobID+'\t%d'%tf+'\t%d'%j+'\t%.8f'%blobLimX+'\t%.8f'%blobLimY+'\t%.8f'%blobMidX+'\t%.8f'%blobMidY+'\t%.8f'%polyArea+'\n')
        blob_file = open(outDir+"/file_number%d"%data_num[i]+"_contour_number_%d"%j+".txt", "w")
        for k in range(len(x)):
            blob_file.write('%.8f'%x[k]+'\t%.8f'%y[k]+'\n')
        blob_file.close()

    plt.savefig(outDir+"/file_number%d"%data_num[i]+"_blob_snap.png")   # save the figure to file
    plt.close()
    allBlobPoly = allBlobPolyCurr
    
    if blob_counter == 0:
        noBlobs = True
        print("No blob found for file number = %d"%data_num[i])
        total_blobs += len(allBlobPoly) #max(len(allBlobPoly),0)

        # test real blob count
        for k in range(len(allBlobPoly)):
            tf = allBlobPoly[k][3]
            if tf == 0:
                oneTimeBlobs += 1
        allBlobPoly = []

    # sleep(0.1)
    # pbar.update(pstep)
    # ax1.set_xlabel("X",fontsize=14)
    # ax1.set_ylabel("Y",fontsize=14)
    # plt.clf()
    
    #del elcDensityData
    del phiData

allBlobXHW = np.array(allBlobXHW)
ab_ave = np.mean(allBlobXHW)

true_blobs = total_blobs - oneTimeBlobs
print('\n-----------\nTOTAL BLOBS: %d \n-----------'%true_blobs)
print('\n-----------\nBLOB FREQ: %d \n-----------'%(true_blobs/(len(data_num)*1e-6)))
print('\n-----------\nBLOB WIDTH: %f \n-----------'%(ab_ave*2))
print('one time blobs', oneTimeBlobs)


#========== Generate Animation ==========
images = []
for i in range(fstart,fend):
    image_name = outDir+'/file_number%d_blob_snap.png'%i
    img = Image.open(image_name)
    images.append(img)

fig, ax = plt.subplots(1,1,figsize=(4.5,5.5),dpi=150)
ax.set_aspect('equal')
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
img_display = ax.imshow(images[0]) # Display the first image initially

def update(i):
    img_display.set_data(images[i]) # Update the image data for each frame
    ax.set_axis_off()
    return img_display,

ani = animation.FuncAnimation(fig, update, frames=len(images), interval=1000, blit=True) # Create the animation

if (save_anim == True):
    ani.save(outDir+'/animation.mp4', writer='ffmpeg', fps=5)
if (show_anim == True):
    plt.show()


exit()
