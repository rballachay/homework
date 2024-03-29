from Q1_starter_Python import draw_slanted_plane
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

N_PIX = 256
N_POINT = 2000
FOV_DEG = 50

IMG_PATH='results/slant_{degree}_sample_img.png'
Q1_PART_A='results/2d_conditional_probability.png'
NUMPY_ARR = "results/numpy_arr.out"
RESULTS_COUNTS="results/counts_each_y_bin.png"
RESULTS_LIKELIHOOD="results/likelihoog_each_angle.png"

def part_a():
    all_rows = []
    for slant in range(-60,70,10):
        img_stack = []
        for i in range(10):
            img = draw_slanted_plane(N_PIX, N_POINT, slant, FOV_DEG) 

            # save a sample image for each angle
            if not i:
                img.save(IMG_PATH.format(degree=slant))
                
            img=np.array(img)
            # we know this is b-w image, but returning as rgb
            img[img==0]=1
            img[img==255]=0
            img_stack.append(img[...,0])
        img_stack = np.stack(img_stack)
        img_stack_avg = np.sum(img_stack,axis=0)
        avg_n_rows = groupedSum(img_stack_avg)

        # take the average again, now accross all the columns
        avg_n_rows = np.sum(avg_n_rows, axis=1)

        avg_n_rows=avg_n_rows/avg_n_rows.sum()

        """
        data=pd.DataFrame({
            'Y Location in Image (from top)':np.round(np.linspace(0,1,avg_n_rows.shape[0]),3),
            'Conditional Probability':avg_n_rows
        })

        fig, ax = plt.subplots(figsize=(10,5),dpi=200)
        ax = sns.barplot(data, x='Y Location in Image (from top)', y="Conditional Probability",ax=ax)
        ax.set_xticks(range(0, len(data['Y Location in Image (from top)']), 4), data['Y Location in Image (from top)'][::4])
        fig.savefig(IMG_PATH.format(degree=slant))
        """
        all_rows.append(avg_n_rows)
    all_rows = np.stack(all_rows)

    fig,ax = plt.subplots(figsize=(20,5),dpi=100)  # Your image (W)idth and (H)eight in inches
    # Stretch image to full figure, removing "grey region"
    plt.subplots_adjust(left=0.03, right=0.93, top=1, bottom=0.03)
    im = ax.imshow(all_rows, cmap="viridis")  # Show the image
    pos = fig.add_axes([0.95, 0.1, 0.02, 0.35])  # Set colorbar position in fig
    fig.colorbar(im, cax=pos)  # Create the colorbar
    ax.set_yticks(np.arange(len(range(-60,70,10))),labels=range(-60,70,10))
    ax.set_ylabel("Plane slant (degrees)")
    ax.set_xlabel("Location in y-axis (normalized 0->1)")
    ax.set_xticks(np.arange(0,64)[::4],labels=np.round(np.linspace(0,1,64),3)[::4])
    plt.savefig(Q1_PART_A)
    return all_rows

def part_b(prob_arr):
    lookup=dict(zip(range(-60,70,10),prob_arr))

    results_counts = {
        'slant':[],
        'n-points':[],
        'y-location':[]
    }
    results_likelihood={
        'slant':[],
        'log-likelihood':[],
        'angle':[]
    }
    for slant in [-60,-30,0,30,60]:
        img = draw_slanted_plane(N_PIX, N_POINT, slant, FOV_DEG) 
        img=np.array(img)[...,-1]
        img[img==0]=1
        img[img==255]=0
        avg_n_rows = groupedSum(img).mean(axis=1)

        avg_n_rows = avg_n_rows/avg_n_rows.sum()
        
        prob_angle=[]

        total_likelihood=0
        for angle in lookup:
            likelihood = np.log((avg_n_rows*lookup[angle]).sum())
            total_likelihood = np.log(np.exp(total_likelihood) + np.exp(likelihood))
            prob_angle.append(likelihood)

        all_probs = 10**(np.stack(prob_angle)-total_likelihood)
        pts_bin = groupedSum(img).sum(axis=1)

        results_counts['slant'].extend([slant]*len(pts_bin))
        results_counts['n-points'].extend(list(pts_bin))
        results_counts['y-location'].extend(list(np.linspace(0,1,64)))

        results_likelihood['slant'].extend([slant]*len(all_probs))
        results_likelihood['log-likelihood'].extend(list(all_probs))
        results_likelihood['angle'].extend(list(range(-60,70,10)))

    results_counts=pd.DataFrame(results_counts)
    results_likelihood=pd.DataFrame(results_likelihood)

    print(results_counts)

    plot = sns.catplot(data=results_counts,x='y-location',col='slant',y='n-points',kind="bar")
    for ax in plot.axes:
        ax[0].set_xticks(np.arange(0,64)[::10],labels=np.round(np.linspace(0,1,64),3)[::10])
    plot.fig.savefig(RESULTS_COUNTS) 

    plt.clf()

    plot = sns.catplot(data=results_likelihood,x='angle',col='slant',y='log-likelihood',ax=ax,kind="bar")
    for ax in plot.axes:
        ax[0].set_yscale("log")
    plot.fig.savefig(RESULTS_LIKELIHOOD)
        



def groupedSum(myArray, N=4):
    result = np.cumsum(myArray, 0)[N-1::N]
    result[1:] = result[1:] - result[:-1]
    return result

if __name__=="__main__":
    if not os.path.exists(NUMPY_ARR):
        numpy_arr = part_a()
        np.savetxt(NUMPY_ARR, numpy_arr, delimiter=',')
    else:
        numpy_arr=np.loadtxt(NUMPY_ARR, delimiter=',')
    
    part_b(numpy_arr)