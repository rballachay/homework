import numpy as np
import matplotlib.pyplot as plt

def make2DGabor(M, kx, ky):
    '''
    This function returns a 2D cosine Gabor and a 2D sine Gabor with 
    center frequency (k0,k1) cycles per M samples.   

    The sigma of the Gaussian is chosen automatically to be one half cycle
    of the underlying sinusoidal wave.   
     
    Example:  make2DGabor( M, cos(theta)*k, sin(theta)*k  ) returns a tuple of
    array containing the cosine and sine Gabors.
  
    e.g.    (cosGabor, sinGabor) = make2DGabor(32,4,0)
  
    Note that a more general definition of a Gabor would pass the sigma as a
    parameter, rather than defining it in terms of k and N.   I've used a 
    more specific definition here because it is all we need for Assignment 2.
    '''
    
    k     =  np.sqrt(kx*kx + ky*ky)
    sigma =  0.5 * M/k        # N/k is number of pixels in one wavelength 

    x = range(-M//2, M//2)
    y = range(-M//2, M//2)
    [X,Y] = np.meshgrid(x,y)
    cos2D = np.cos(  2*np.pi/M * (kx * X + ky * Y) )
    sin2D = np.sin(  2*np.pi/M * (kx * X + ky * Y) )

#  Here the x are the column indices and y are the row indices, which is
#  what we want.   x increases to right and y increases down.

    xarray = np.array(x)
    g = 1/(np.sqrt(2*np.pi)*sigma) *  np.exp(- xarray*xarray / (2 * sigma*sigma) )
    Gaussian2D = np.outer(g,g)

    cosGabor = Gaussian2D * cos2D
    sinGabor = Gaussian2D * sin2D
    return cosGabor, sinGabor
