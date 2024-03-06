from utils.make2DGabor import make2DGabor
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import itertools 

N = 256
NX=NY=N
MAX_SPEED = 8

Q1_PART_A_PLOT = 'results/2d_image_velocities.png'

def main():
    I_R = np.array( np.random.choice(2, [N, N]))
    I_L = np.roll(I_R, 2, axis=1) 

    M = 32     # window width on which Gabor is defined

    NXvalid = NX-M+1
    NYvalid = NY-M+1

    MAX_SPEED = 8


    disps = np.array( np.arange(-MAX_SPEED,MAX_SPEED+1))
    disparities = list(itertools.product(disps,disps))

    N_DISPARITIES = len(disparities)

    binocularComplexResponses =  np.zeros((N_DISPARITIES, NYvalid - 2*MAX_SPEED, NXvalid - 2*MAX_SPEED))

    # note parameter ordering of mk2DGabor...  x before y 
    (cosGabor, sinGabor) = make2DGabor(32, 4, 0)

    for disparityIndex in range(N_DISPARITIES):
        dx, dy = disparities[disparityIndex]
        cosGaborResponsesL = convolve2d( cosGabor, I_L, mode='valid')
        sinGaborResponsesL = convolve2d( sinGabor, I_L, mode='valid')
        cosGaborResponsesR = convolve2d( cosGabor, I_R, mode='valid')
        sinGaborResponsesR = convolve2d( sinGabor, I_R, mode='valid')
        c = cosGaborResponsesL[MAX_SPEED+dy:NYvalid-MAX_SPEED+dy:,MAX_SPEED+dx:NXvalid-MAX_SPEED+dx] - cosGaborResponsesR[MAX_SPEED:NYvalid-MAX_SPEED:,MAX_SPEED:NXvalid-MAX_SPEED]
        s = sinGaborResponsesL[MAX_SPEED+dy:NYvalid-MAX_SPEED+dy:,MAX_SPEED+dx:NXvalid-MAX_SPEED+dx] - sinGaborResponsesR[MAX_SPEED:NYvalid-MAX_SPEED:,MAX_SPEED:NXvalid-MAX_SPEED]
        binocularComplexResponses[disparityIndex,:,:] =  np.sqrt(c * c + s * s)

    mean_response = np.zeros((N_DISPARITIES))
    for disparityIndex in range(N_DISPARITIES):
        mean_response[disparityIndex] = np.mean(binocularComplexResponses[disparityIndex,:,:])

    image = mean_response.reshape((len(disps),len(disps)))
    fig = plt.figure(figsize=(10, 10))  # Your image (W)idth and (H)eight in inches
    # Stretch image to full figure, removing "grey region"
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    im = plt.imshow(image, cmap="viridis")  # Show the image
    pos = fig.add_axes([0.93, 0.1, 0.02, 0.35])  # Set colorbar position in fig
    fig.colorbar(im, cax=pos)  # Create the colorbar
    plt.savefig(Q1_PART_A_PLOT)


if __name__=="__main__":
    main()