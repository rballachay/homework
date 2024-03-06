#  Q1.py
#
#  author:  Michael Langer  COMP 546  Winter 2024
#  Assignment 2  Question 1

import numpy as np
import matplotlib.pyplot as plt
from make2DGabor import make2DGabor

M = 32    # window width on which Gabor is defined
k = 4     # frequency

#  wavelength of underlying sinusoid is M/k pixels per cycle.
#  k_x = cos(theta)*k, k_y = sin(theta)*k   
#     so when theta = 0,  (kx, ky) = (k, 0) and we have vertical sinusoid.

I = np.zeros((M,M))  

N_THETA = 12

responses_bar   =  np.zeros((N_THETA,1))
responses_edge  =  np.zeros((N_THETA,1))

(cosGabor, sinGabor) = make2DGabor(M, k, 0 )

fig1, axes1 = plt.subplots(3,4)
fig2, axes2 = plt.subplots(3,4)

for thetaCt in range(N_THETA): 
    theta = np.pi/180* (180/N_THETA) * thetaCt;
    # note parameter ordering of mk2DGabor...  x before y

    #  response to bar

    for row in range(M):
        for col in range(M):
            tmp = - (row-M/2)*np.sin(theta) + (col-M/2)* np.cos(theta)
            I[row,col] = np.exp( - (tmp*tmp)/(2*2*2) ) 
 
    i = thetaCt // 4
    j = thetaCt % 4
    axes1[i][j].imshow(I, cmap='gray')
    axes1[i][j].set_title('theta = ' + str(thetaCt*180/N_THETA), fontsize=10);
 
    c  = np.sum(cosGabor * I)
    s  = np.sum(sinGabor * I)
    responses_bar[thetaCt]  = np.sqrt(c*c + s*s);

    #  response to edge
    for row in range(M):
        for col in range(M):
            tmp = - (row-M/2)*np.sin(theta) + (col-M/2)*np.cos(theta)
            I[row,col] = 1 / ( 1 + np.exp( - tmp ) ) 
    
    c  = np.sum(cosGabor * I)
    s  = np.sum(sinGabor * I)
    responses_edge[thetaCt]  = np.sqrt(c*c + s*s)

    axes2[i][j].imshow(I, cmap='gray')
    axes2[i][j].set_title('theta = ' + str(thetaCt*180//N_THETA), fontsize=10)
    
fig = plt.figure(3)
ax = fig.add_subplot(111)

xs = [t* 180/N_THETA for t in list(range(0,N_THETA))]
plt.plot( xs ,  responses_bar ,'-*r', label = 'bar')
plt.plot( xs ,  responses_edge ,'-*g', label = 'edge')

ax.legend()
ax.set_xlabel('theta')
plt.xticks(np.arange(0, 180, step=180/N_THETA))
ax.set_ylabel('response from complex cell')
ax.set_box_aspect(1)

plt.subplots_adjust(wspace=1, hspace=0.4)
plt.show()
