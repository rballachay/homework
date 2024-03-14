from PIL import Image
import numpy as np
from utils.make2DGabor import make2DGabor
from scipy.signal import convolve2d

def get_response(cosGabor,sinGabor,image, NXValid, NYValid):
    
    cosGaborResponses = convolve2d( cosGabor, image, mode='valid')[:NYValid,:NXValid]
    sinGaborResponses = convolve2d( sinGabor, image, mode='valid')[:NYValid,:NXValid]
    response = np.sqrt(cosGaborResponses ** 2 + sinGaborResponses ** 2)
    return np.mean(response)

def Q2(I):
    M=32
    Ny,Nx = I.shape

    NXvalid = Ny-M+1
    NYvalid = Nx-M+1

    left_right = I[:,:Nx//2], I[:,Nx//2+1:]
    top_bottom = I[:Ny//2,:], I[Ny//2+1:,:]
    
    fracs = {
        'tb':top_bottom,
        'lr':left_right,
    }
    results = {}
    for title,image_fracs in fracs.items():
        first, second = image_fracs
        (vertCosGabor, vertSinGabor) = make2DGabor(M, 4, 0)
        (horCosGabor, horSinGabor) = make2DGabor(M, 0, 4)

        first_vert = get_response(vertCosGabor,vertSinGabor,first,NXvalid,NYvalid)
        first_hor = get_response(horCosGabor,horSinGabor,first,NXvalid,NYvalid)

        sec_vert = get_response(vertCosGabor,vertSinGabor,second,NXvalid,NYvalid)
        sec_hor = get_response(horCosGabor,horSinGabor,second,NXvalid,NYvalid)
        results[title] = max([first_vert+sec_hor,first_hor+sec_vert])

    return max(results,key=results.get)
