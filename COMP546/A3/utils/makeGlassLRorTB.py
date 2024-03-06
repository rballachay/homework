import numpy as np
import matplotlib.pyplot as plt

def makeGlassLRorTB(N, pNoise, p0, condition,dx,dy): 
    '''
    Make an image that is composed of translation Glass patterns, 
    separated in orientation by 90 degrees.   There are two conditions:   
    either the two Glass patterns are in the top and bottom halves
    or they are in the left and right halves.

    Parameters:   
        N (width of image)    
        pNoise  (probability of noise point being white)
        p0      (probability of a dot pair placed with first point at a pixel) 
        condition  ('lr', 'tb'  (left-right or top-bottom)
        dx, dy  (step defining the Glass pattern)
    '''
    dx0 = dx        
    dy0 = dy 
    dx1 = -dy0  # rotated by 90 degrees 
    dy1 =  dx0 

    glass = np.random.rand(N,N) < p0

#  Make the Glass pattern by shifting the set of dots.

    if condition == 'lr': 
        glass[:,N//2+1:]  = np.logical_or(glass[:,N//2+1:], np.roll(glass[:,N//2+1:], (dx1, dy1), axis=(0,1))) 
        glass[:,:N//2]  =   np.logical_or(glass[:,:N//2]  , np.roll(glass[:,:N//2],  (dx0, dy0), axis=(0,1))) 
    else:
        glass[N//2+1:,:]  = np.logical_or(glass[N//2+1:,:], np.roll(glass[N//2+1:,:], (dx1, dy1), axis=(0,1))) 
        glass[:N//2,:]  =   np.logical_or(glass[:N//2,:]  , np.roll(glass[:N//2,:],  (dx0, dy0), axis=(0,1))) 

    glass = np.logical_or(glass, (np.random.rand(N,N) < pNoise))
    return glass.astype(float)

if __name__ == '__main__':
    I = makeGlassLRorTB(128, .01, .05, 'tb',2,0)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.imshow(I,cmap='gray')
    plt.show()