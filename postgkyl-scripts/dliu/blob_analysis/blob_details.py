import numpy as np

# Path to the directory
Dir = '/scratch/gpfs/dingyunl/gk-g0-app-gpu-hack2024-positivity/gkylzero/blobData2/'
N = 128  # number of rows in blob_size.txt
NB = 32 # number of blobs
blob_detail_x = open(Dir+'blob_details_x.dat', 'w')
blob_detail_y = open(Dir+'blob_details_y.dat', 'w')

# Physical constants and simulation parameters
B0 = 2.57             # [T]
q0 = 4.67
r0 = 0.5              # [m]

eV = 1.602176487e-19  # [C]
me = 9.1093837015e-31 # [kg]
mp = 1.672621637e-27  # [kg]
mi = 2.014*mp         # [kg]
Te = 40*eV
Ti = 72*eV
c_s = np.sqrt(Te/mi)
omega_ci = eV*B0/mi
rho_s = c_s / omega_ci

xMin, xMax = 0.15, 0.172
Lx = xMax - xMin
Ly = 200.0*rho_s*q0/r0
yMin, yMax = -Ly/2.0, Ly/2.0 
Nx = 32
dx = Lx / Nx
simArea = Lx * Ly

microsec = 1.0
second = microsec * 1.0e6

# Initialize arrays
blobLimX = np.zeros(N)
blobLimY = np.zeros(N)
blobMidX = np.zeros(N)
blobMidY = np.zeros(N)
polyArea = np.zeros(N)

blobTotalF = np.zeros((N, NB))
blobWidthX = np.zeros((N, NB))
blobWidthY = np.zeros((N, NB))
blobCenterX = np.zeros((N, NB))
blobCenterY = np.zeros((N, NB))
blobArea = np.zeros((N, NB))

velocityX = np.zeros((N, NB))
velocityY = np.zeros((N, NB))
velocity = np.zeros((N, NB))

diffWidthX = np.zeros((N, N, NB))
diffWidthY = np.zeros((N, N, NB))
diffCenterX = np.zeros((N, N, NB))
diffCenterY = np.zeros((N, N, NB))
diffArea = np.zeros((N, N, NB))

corrPos = np.zeros((N, N))
corrVal = np.zeros((N, N))
freqPos = np.zeros((N, N))
freqVal = np.zeros((N, N))

# Placeholder for file reading
frame = np.zeros(N, dtype=int)
bN = np.zeros(N, dtype=int)
tf = np.zeros(N)
G = np.zeros(N)
time = np.zeros(N)

# File opening (Assume 'blob_size.txt' exists in the working directory)
with open(Dir+'blob_size.txt', 'r') as f:
    for i in range(N):
        data = f.readline().split()
        frame[i] = int(data[0])
        bN[i] = int(data[1])
        tf[i] = float(data[2])+1.0
        G[i] = int(data[3])
        blobLimX[i] = float(data[4])
        blobLimY[i] = float(data[5])
        blobMidX[i] = float(data[6])
        blobMidY[i] = float(data[7])
        polyArea[i] = float(data[8])

# Summary information
blobLimXSum = np.sum(blobLimX)
blobLimYSum = np.sum(blobLimY)
polyAreaSum = np.sum(polyArea)

print("blobLimXSum=%10.8e"%(blobLimXSum/float(N)))
print("blobLimYSum=%10.8e"%(blobLimYSum/float(N)))
print("polyAreaSum=%10.8e"%(polyAreaSum/float(N)))
print("percentage of area=%5.4f"%(polyAreaSum*100.0/(float(N)*simArea)))
print("Place 1")

# Initialize variables
time[0] = frame[0]
blobTotalF[0, 0] = tf[0]
blobWidthX[0, 0] = blobLimX[0]
blobWidthY[0, 0] = blobLimY[0]
blobCenterX[0, 0] = blobMidX[0]
blobCenterY[0, 0] = blobMidY[0]
blobArea[0, 0] = polyArea[0]

# Iterate over frames
iteration = 0
nBlob = 0
maxBlob = 0
for i in range(1, N):
    if frame[i] != frame[i-1]:
        iteration += 1
        nBlob = bN[i] 
        time[iteration] = frame[i]
        blobTotalF[iteration, nBlob] = tf[i]
        blobWidthX[iteration, nBlob] = blobLimX[i]
        blobWidthY[iteration, nBlob] = blobLimY[i]
        blobCenterX[iteration, nBlob] = blobMidX[i]
        blobCenterY[iteration, nBlob] = blobMidY[i]
        blobArea[iteration, nBlob] = polyArea[i]
    elif frame[i] == frame[i-1]:
        nBlob = bN[i] 
        blobTotalF[iteration, nBlob] = tf[i]
        blobWidthX[iteration, nBlob] = blobLimX[i]
        blobWidthY[iteration, nBlob] = blobLimY[i]
        blobCenterX[iteration, nBlob] = blobMidX[i]
        blobCenterY[iteration, nBlob] = blobMidY[i]
        blobArea[iteration, nBlob] = polyArea[i]
    if nBlob > maxBlob:
        maxBlob = nBlob
#np.set_printoptions(threshold=np.inf)
#print(iteration, maxBlob)
#print(blobTotalF)
#print(blobCenterX)
#print(blobCenterY)

# Total frame average
totalFrameAve = np.mean(tf)
print(f"Total Frame Average: {totalFrameAve}")
print("Place 2")

maxIter = iteration

# Velocity calculation
for j in range(maxBlob+1):
    for i in range(maxIter+1):
      if (blobTotalF[i,j]>1):
         velocityX[i,j] = (blobCenterX[i,j]-blobCenterX[i-1,j])*second
         velocityY[i,j] = (blobCenterY[i,j]-blobCenterY[i-1,j])*second 

#print(velocityX)
#print(velocityY)

for i in range(maxIter+1):
    for j in range(maxBlob+1):
        if (velocityX[i,j]>1.0e-10): # only consider blobs with positive radial velocity
            blob_detail_x.write('%d'%i+'\t%d'%j+'\t%.8f'%blobWidthX[i,j]+'\t%.8f'%velocityX[i,j]+'\t%.8f'%blobCenterX[i,j]+'\n')
        if (abs(velocityY[i,j])>1.0e-10):
            blob_detail_y.write('%d'%i+'\t%d'%j+'\t%.8f'%blobWidthY[i,j]+'\t%.8f'%velocityY[i,j]+'\t%.8f'%blobCenterY[i,j]+'\n')

blob_detail_x.close()
blob_detail_y.close()

exit()
"""
# Initialize tempBin and freqBin
tempBin = 0
freqBin = 0

# Velocity and correlation calculations
for iteration in range(1, maxIter+1):
    for nBlob in range(0, maxBlob+1):
        correlation = 0
        flag = 0
        for iterBack in range(iteration-1, -1, -1):
            diffWidthX[iteration, iterBack, nBlob] = blobWidthX[iteration, nBlob] - blobWidthX[iterBack, nBlob]
            diffWidthY[iteration, iterBack, nBlob] = blobWidthY[iteration, nBlob] - blobWidthY[iterBack, nBlob]
            diffCenterX[iteration, iterBack, nBlob] = blobCenterX[iteration, nBlob] - blobCenterX[iterBack, nBlob]
            diffCenterY[iteration, iterBack, nBlob] = blobCenterY[iteration, nBlob] - blobCenterY[iterBack, nBlob]
            diffArea[iteration, iterBack, nBlob] = blobArea[iteration, nBlob] - blobArea[iterBack, nBlob]
print(diffCenterX[:, :, 1])
print(diffCenterY[:, :, 1])

            diffCenterXBlob = abs(diffCenterX[iteration, iterBack, nBlob]) * second / c_s
            diffCenterYBlob = abs(diffCenterY[iteration, iterBack, nBlob]) * second / c_s
           
            if 0.2 > diffCenterXBlob > 1.0e-10 and 0.2 > diffCenterYBlob > 1.0e-10:
                correlation += 1
                flag = 1
                velocityX[iteration, iterBack, nBlob] = diffCenterX[iteration, iterBack, nBlob] * second / c_s
                velocityY[iteration, iterBack, nBlob] = diffCenterY[iteration, iterBack, nBlob] * second / c_s
                velocity[iteration, iterBack, nBlob] = np.sqrt(velocityX[iteration, iterBack, nBlob]**2 + velocityY[iteration, iterBack, nBlob]**2)
                
                if iterBack == iteration-1 and velocityX[iteration, iterBack, nBlob] > 0:
                    print(f"Blob {nBlob} X velocity: {velocityX[iteration, iterBack, nBlob]}")
                if iterBack == iteration-1 and velocityY[iteration, iterBack, nBlob] > 0:
                    print(f"Blob {nBlob} Y velocity: {velocityY[iteration, iterBack, nBlob]}")
                
                if iterBack < iteration-1 and correlation == iteration - iterBack:
                    tempBin += 1
                    corrPos[tempBin] = blobCenterX[iteration, nBlob]
                    corrVal[tempBin] = float(correlation)
                    print(f"Correlation position: {corrPos[tempBin]}, value: {corrVal[tempBin]}")

            if iterBack == iteration-1 and flag == 0 and blobCenterX[iteration, nBlob] > xMin:
                freqBin += 1
                freqPos[freqBin] = blobCenterX[iteration, nBlob]
                freqVal[freqBin] = time[iteration] - time[0]

# File writing
with open(Dir+'blob_details_x.dat', 'w') as f:
    for i in range(1, maxBlob + 1):
        f.write(f"{i}, {blobCenterX[i-1, 0]}\n")

with open(Dir+'blob_details_y.dat', 'w') as f:
    for i in range(1, maxBlob + 1):
        f.write(f"{i}, {blobCenterY[i-1, 0]}\n")
"""
