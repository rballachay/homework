#  Q2.py
#
#  author:  Michael Langer  COMP 546  Winter 2024
#  Assignment 2  Question 2

import numpy as np
import matplotlib.pyplot as plt
from make2DGabor import make2DGabor
from scipy import signal

N = 256
maxK = N//2-1

meanComplexResponse = np.zeros((maxK,1))

M=32
kx = 4   #  4 cycles per 32 pixels,  32 cycles per 256 pixels   
(cosGabor, sinGabor) = make2DGabor(M, kx, 0 )

#  raised sine images
   
fig = plt.figure(1)
ax = fig.add_subplot(111)

for k in range(0,maxK):
    I =  np.ones((N,1)) * (1 + np.sin( 2 * np.pi * k * np.array(range(N)) /N))/2
    sinGaborResponses  = signal.convolve2d( sinGabor, I, mode='valid')
    cosGaborResponses  = signal.convolve2d( cosGabor, I, mode='valid')
    complexResponses  = np.sqrt(sinGaborResponses * sinGaborResponses +  cosGaborResponses * cosGaborResponses); 
    meanComplexResponse[k] =  np.mean(complexResponses);

plt.plot( np.arange(maxK)+1, meanComplexResponse) 
ax.set_xlabel('frequency, k')
ax.set_ylabel('Q1 (a):  mean complex cell response')

'''
The maximum response occurs when the raised sin image has frequency N / (M/kx).    Why?    
The wavelength should match that of the Gabor frequency (kx=4 cycles per M=32 pixels)
Thus, for the responses to the raised sine, the max should for a wavelength of M/kx pixels.
So for the raised sinusoid, the number of cycles should be k = N/(M/kx). 
'''

fig1, axes1 = plt.subplots(2,3)

k= N/(M/kx)
I =  np.ones((N,1)) * (1 + np.sin( 2 * np.pi * k * np.array(range(N)) /N))/2
sinGaborResponses  = signal.convolve2d( sinGabor, I, mode='valid')
cosGaborResponses  = signal.convolve2d( cosGabor, I, mode='valid')
complexResponses  = np.sqrt(sinGaborResponses * sinGaborResponses +  cosGaborResponses * cosGaborResponses) 

axes1[0][0].imshow(sinGaborResponses, cmap='gray') 
axes1[0][1].imshow(cosGaborResponses, cmap='gray') 
axes1[0][2].imshow(complexResponses, cmap='gray') 
axes1[1][0].hist(np.reshape(sinGaborResponses,(-1)),bins=20) 
axes1[1][1].hist(np.reshape(cosGaborResponses,(-1)),bins=20) 
axes1[1][2].hist(np.reshape(complexResponses,(-1)), bins=20) 
fig1.suptitle('Q1 (a):  responses for k = ' + str(N // (M//kx)) )
 
#  TODO:  how to set the box aspect ratio to 1?   The following doesn't work.
#
# axes1.set_box_aspect(1)
# plt.gca().set_aspect('equal')

#  noise image plots

Inoise = np.array( np.random.choice(2, [N, N]))

fig2, axes2 = plt.subplots(2,3)

sinGaborResponses  = signal.convolve2d( sinGabor, Inoise, mode='valid')
cosGaborResponses  = signal.convolve2d( cosGabor, Inoise, mode='valid')
complexResponses  = np.sqrt(sinGaborResponses * sinGaborResponses +  cosGaborResponses * cosGaborResponses) 

axes2[0][0].imshow(sinGaborResponses, cmap='gray') 
axes2[0][1].imshow(cosGaborResponses, cmap='gray') 
axes2[0][2].imshow(complexResponses, cmap='gray') 
axes2[1][0].hist(np.reshape(sinGaborResponses,(-1)),bins=20) 
axes2[1][1].hist(np.reshape(cosGaborResponses,(-1)),bins=20) 
axes2[1][2].hist(np.reshape(complexResponses,(-1)), bins=20) 
fig2.suptitle('Q2 (b):  responses to noise')
plt.show()