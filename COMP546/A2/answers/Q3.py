#  Q3.py
#
#  author:  Michael Langer  COMP 546  Winter 2024
#  Assignment 2  Question 3

import numpy as np
import matplotlib.pyplot as plt
from make2DGabor import make2DGabor
from scipy import signal

N = 128
NY = N
NX = N

image_disparity = 4

I_R = np.array( np.random.choice(2, [N, N]))
I_L = np.roll( I_R, image_disparity, axis=1) 

#  Show the image 

fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.imshow(I_L,cmap='gray') 

#  Filter the image with N_THETA orientations, 0, 180/N_THETA, 
#  2*(180/N_THETA),  ... (N_THETA-1)*180/N_THETA  degrees.
#  
#  Since Y axis is down,  theta will increase clockwise, rather than
#  counter-clockwise.

N_THETA = 4

M = 32     # window width on which Gabor is defined
k = 4      # frequency

#  wavelength of underlying sinusoid is M/k pixels per cycle.
#  k_x = cos(theta)*k, k_y = sin(theta)*k   
#     so when theta = 0,  (kx, ky) = (k, 0) and we have vertical sinusoid.

NXvalid = NX-M+1
NYvalid = NY-M+1

thetaRange = np.arange(N_THETA)*np.pi/N_THETA     # in radians

MAX_DISPARITY = 8
disparities = np.array( np.arange(-MAX_DISPARITY,MAX_DISPARITY+1))
N_DISPARITIES = 2*MAX_DISPARITY+1

cosGaborResponsesL =  np.zeros((N_THETA, NYvalid, NXvalid))  # will use 'valid' results of filter
sinGaborResponsesL =  np.zeros((N_THETA, NYvalid, NXvalid))
cosGaborResponsesR =  np.zeros((N_THETA, NYvalid, NXvalid))  # will use 'valid' results of filter
sinGaborResponsesR =  np.zeros((N_THETA, NYvalid, NXvalid))
binocularComplexResponses =  np.zeros((N_DISPARITIES, N_THETA, NYvalid, NXvalid - 2*MAX_DISPARITY))

for thetaCt in range(N_THETA):
    theta = thetaRange[thetaCt]
    # note parameter ordering of mk2DGabor...  x before y 
    (cosGabor, sinGabor) = make2DGabor(M, np.cos(theta)*k, np.sin(theta)*k  )
    cosGaborResponsesL[thetaCt,:,:]  = signal.convolve2d( cosGabor, I_L, mode='valid')
    sinGaborResponsesL[thetaCt,:,:]  = signal.convolve2d( sinGabor, I_L, mode='valid')
    cosGaborResponsesR[thetaCt,:,:]  = signal.convolve2d( cosGabor, I_R, mode='valid')
    sinGaborResponsesR[thetaCt,:,:]  = signal.convolve2d( sinGabor, I_R, mode='valid')

for disparityIndex in range(N_DISPARITIES):  
    for thetaCt in range(N_THETA):
        theta = thetaRange[thetaCt]
        d = disparities[disparityIndex]
        c = cosGaborResponsesL[thetaCt,:,MAX_DISPARITY+d:NXvalid-MAX_DISPARITY+d] - cosGaborResponsesR[thetaCt,:,MAX_DISPARITY:NXvalid-MAX_DISPARITY]
        s = sinGaborResponsesL[thetaCt,:,MAX_DISPARITY+d:NXvalid-MAX_DISPARITY+d] - sinGaborResponsesR[thetaCt,:,MAX_DISPARITY:NXvalid-MAX_DISPARITY]
        binocularComplexResponses[disparityIndex, thetaCt,:,:] =  np.sqrt(c * c + s * s)

fig = plt.figure(2)
ax = fig.add_subplot(111)

mean_response = np.zeros((N_DISPARITIES, N_THETA))
for thetaCt in range(N_THETA):
    for disparityIndex in range(N_DISPARITIES):    
        mean_response[disparityIndex, thetaCt] = np.mean(binocularComplexResponses[disparityIndex, thetaCt,:,:])
    plt.plot(disparities, mean_response[:, thetaCt] , label = str(thetaCt*180//N_THETA) + ' deg' )

ax.legend()
ax.set_xlabel('disparity to which cells are tuned')
ax.set_ylabel('mean responses for binocular complex cells')

plt.show()
