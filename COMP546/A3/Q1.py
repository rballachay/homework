from utils.make2DGabor import make2DGabor
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import itertools 

N = 128
NX=NY=N
MAX_SPEED = 8

Q1_PART_A_PLOT = 'results/q1/2d_image_velocities.png'
Q1_PART_B_PLOT = 'results/q1/image_velocity_angled_.png'
Q1_PART_C_PLOT = 'results/q1/summed_respones_c.png'
Q1_PART_D_PLOT = 'results/q1/larger_target_velocity.png'

def main():
    I_R = np.array( np.random.choice(2, [N, N]))
    I_L = np.roll(I_R, 2, axis=1) 

    M = 32     # window width on which Gabor is defined

    NXvalid = NX-M+1
    NYvalid = NY-M+1

    MAX_SPEED = 8

    disps = np.array( np.arange(-MAX_SPEED,MAX_SPEED+1))
    disparities = list(itertools.product(disps,disps))


    ## question a
    N_DISPARITIES = len(disparities)

    binocularComplexResponses =  np.zeros((N_DISPARITIES, NYvalid - 2*MAX_SPEED, NXvalid - 2*MAX_SPEED))
    
    # note parameter ordering of mk2DGabor...  x before y 
    (cosGabor, sinGabor) = make2DGabor(M, 4, 0)

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
    fig,ax = plt.subplots(figsize=(10, 10))  # Your image (W)idth and (H)eight in inches
    # Stretch image to full figure, removing "grey region"
    plt.subplots_adjust(left=0.03, right=1, top=1, bottom=0.03)
    im = ax.imshow(image.T, cmap="viridis")  # Show the image
    pos = fig.add_axes([0.93, 0.1, 0.02, 0.35])  # Set colorbar position in fig
    fig.colorbar(im, cax=pos)  # Create the colorbar
    ax.set_xticks(np.arange(len(disps)),labels=disps)
    ax.set_yticks(np.arange(len(disps)),labels=disps)
    plt.savefig(Q1_PART_A_PLOT)


    summed_complex = np.zeros((len(disps),len(disps),4))
    summed_complex[...,0] = image.T

    for i,angle in enumerate((45,90,135)):
        ## question b
        (cosGabor, sinGabor) = make2DGabor(M, 4*np.cos(np.deg2rad(angle)), 4*np.sin(np.deg2rad(angle)))

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
        fig,ax = plt.subplots(figsize=(10, 10))  # Your image (W)idth and (H)eight in inches
        # Stretch image to full figure, removing "grey region"
        plt.subplots_adjust(left=0.03, right=1, top=1, bottom=0.03)
        im = ax.imshow(image.T, cmap="viridis")  # Show the image
        pos = fig.add_axes([0.93, 0.1, 0.02, 0.35])  # Set colorbar position in fig
        fig.colorbar(im, cax=pos)  # Create the colorbar
        ax.set_xticks(np.arange(len(disps)),labels=disps)
        ax.set_yticks(np.arange(len(disps)),labels=disps)
        plt.savefig(Q1_PART_B_PLOT.replace('.png',f'{angle}.png'))

        summed_complex[...,i+1] = image.T

    summed_complex = summed_complex.mean(axis=-1)
    fig,ax = plt.subplots(figsize=(10, 10))  # Your image (W)idth and (H)eight in inches
    # Stretch image to full figure, removing "grey region"
    plt.subplots_adjust(left=0.03, right=1, top=1, bottom=0.03)
    im = ax.imshow(summed_complex, cmap="viridis")  # Show the image
    pos = fig.add_axes([0.93, 0.1, 0.02, 0.35])  # Set colorbar position in fig
    fig.colorbar(im, cax=pos)  # Create the colorbar
    ax.set_xticks(np.arange(len(disps)),labels=disps)
    ax.set_yticks(np.arange(len(disps)),labels=disps)
    plt.savefig(Q1_PART_C_PLOT)


    ## question d

    summed_complex = np.zeros((len(disps),len(disps),4))
    I_L = np.roll(I_R, 6, axis=1) 
  
    for i,angle in enumerate((0,45,90,135)):
        ## question b
        (cosGabor, sinGabor) = make2DGabor(M, 4*np.cos(np.deg2rad(angle)), 4*np.sin(np.deg2rad(angle)))

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
        summed_complex[...,i] = image.T

    summed_complex = summed_complex.mean(axis=-1)
    fig,ax = plt.subplots(figsize=(10, 10))  # Your image (W)idth and (H)eight in inches
    # Stretch image to full figure, removing "grey region"
    plt.subplots_adjust(left=0.03, right=1, top=1, bottom=0.03)
    im = ax.imshow(summed_complex, cmap="viridis")  # Show the image
    pos = fig.add_axes([0.93, 0.1, 0.02, 0.35])  # Set colorbar position in fig
    fig.colorbar(im, cax=pos)  # Create the colorbar
    ax.set_xticks(np.arange(len(disps)),labels=disps)
    ax.set_yticks(np.arange(len(disps)),labels=disps)
    plt.savefig(Q1_PART_D_PLOT)



if __name__=="__main__":
    main()