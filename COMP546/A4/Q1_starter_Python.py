import numpy as np
import random
from PIL import Image, ImageDraw

def deg_to_rad(sigma):
    return sigma/180*np.pi
   
def rad_to_deg(rad):
    return rad*180/np.pi

def xy_to_pix(x, N, FOV_DEG):
    return x/np.tan(deg_to_rad(FOV_DEG/2)) * N/2

def XYZ_to_xy(X,Y,Z):
    '''
    Takes 3D point and returns the position on image plane for projection plane at Z=1
    '''
    x1 = X/Z
    y1 = Y/Z 
    return x1,y1    

def xy_to_xpyp(x, y, N, FOV_DEG ):
    '''
    x,y : float
        x,y positions in the projection plane (radians)
    N:  int
        number of pixels in the x and y directions
    FOV_DEG:  the field of view angle in degrees.
    Returns  pixel positions (xp, yp) ints
    '''
    return ( N/2 + xy_to_pix(x, N, FOV_DEG), N/2 + xy_to_pix(y, N, FOV_DEG))

def draw_slanted_plane(N_PIX, N_POINTS, slant, FOV_DEG):  
    '''   N_POINTS is the number of points that would be dropped for the case of slant = 0.
          When slant is different from 0,  we will need to drop more dots so
          that dots can hit anywhere in the field of view.  See N_POINTS_AUGMENTED below.
    '''
    f = 50
    
    if abs(slant) >= 90 - FOV_DEG/2:
        raise ValueError("we require FOV_DEG/2 < 90 - slant")
        
    #  make a canvas for drawing the image
    img = Image.new("RGB", (N_PIX, N_PIX), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    '''
    If it not obvious to you what the images are,  then you should multiply the XMAX and YMAX
    below by a small constant so that you restrict the dots to falling in a smaller rectangular
    region, and so the boundaries of the projected slanted rectangle will be visible in the image.
    But for your calculations,  you should use the XMAX and YMAX below so that the texture is
    guarenteed to reach all parts of the image.
    
    The XMAX and YMAX bounds define where the dots are sampled in a frontoparallel plane.
    In case you are wondering, these bounds are derived from the intersection of:
       Z = f + Y tan(slant)
       Y = Z tan(FOV/2)
    '''

    Y = f * np.tan(deg_to_rad(FOV_DEG/2)) / (1 - np.tan(deg_to_rad(FOV_DEG/2)) *( np.tan(deg_to_rad(abs(slant)))))
    delta_Z = Y * np.tan(deg_to_rad(abs(slant)))
    XMAX = (f + delta_Z) * np.tan(deg_to_rad(FOV_DEG/2))
    YMAX =  Y / np.cos(deg_to_rad( abs(slant)))

    N_POINTS_AUGMENTED = int(N_POINTS * XMAX * YMAX / (f**2 * np.tan(deg_to_rad(FOV_DEG/2))**2))

    for i in range(N_POINTS_AUGMENTED):
        #  Drop point randomly in the Z=f plane
        X = XMAX * (2*random.random()-1)
        Y0 = YMAX * (2*random.random()-1)

        #  Rotate plane by theta degrees about the parametric line (X,Y,Z) = (t, 0, f)

        #  Since Python has y axis increasing by growing down, a positive slant would
        #  correspond to a negative slant if y were growing up.  So, multiply slant by -1
        #  so that the positive slant looks like a floor and negative slant looks like a
        #  ceiling.
        
        Y = Y0 * np.cos( deg_to_rad(-slant) )
        Z  = f + Y0 * np.sin( deg_to_rad(-slant) )
        (x,y) = XYZ_to_xy(X,Y,Z)
        (x,y) = xy_to_xpyp(x, y, N_PIX, FOV_DEG)

        #  Check if points fall within field of view
        
        if (x < N_PIX) and (x >=0) and (y < N_PIX) and (y >=0):
            draw.point([(x,y)],  fill =(0,0,0))
    #img.show()
    return img

if __name__ == "__main__":
    N_PIX = 256
    N_POINT = 2000
    SLANT = -60
    FOV_DEG = 50
    draw_slanted_plane(N_PIX, N_POINT, SLANT, FOV_DEG) 
