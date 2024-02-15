import numpy as np
from PIL import Image
from utils.make2DGabor import make2DGabor
from scipy.signal import convolve2d
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

PART_A_FIG = "results/q2/part_a_fig.png"
PART_A_RESPONSE_K = "results/q2/response_vs_k.png"
PART_B_FIG = "results/q2/part_b_fig.png"

N = 256
IMG_DIM = (N, N)


def construct_image(k, dims=IMG_DIM, n=N):
    image = np.zeros(dims)
    for x in range(dims[0]):
        image[x, :] = 0.5 * (1 + np.sin((2 * np.pi / n) * k * x))
    return image.T


class Questions:
    part_a_cache = "results/q2/response_cache.csv"
    part_b_cache = "results/q2/noise_cache.csv"

    @property
    def part_a(self):
        if not os.path.exists(self.part_a_cache):
            (cosGabor, sinGabor) = make2DGabor(32, 4, 0)
            results = {"k": [], "response": [], "cos": [], "sin": []}

            for k in np.arange(1, N / 2 + 1):
                img = construct_image(k)
                _cos = convolve2d(img, cosGabor, mode="valid").sum()
                _sin = convolve2d(img, sinGabor, mode="valid").sum()
                response = np.sqrt(_cos ** 2 + _sin ** 2)

                results["k"].append(k)
                results["sin"].append(_sin)
                results["cos"].append(_cos)
                results["response"].append(response)

            results = pd.DataFrame(results)
            results.to_csv(self.part_a_cache, index=False)
        else:
            results = pd.read_csv(self.part_a_cache)

        fig, ax = plt.subplots(2, 3, dpi=200, figsize=(20, 10))
        (cosGabor, sinGabor) = make2DGabor(32, 4, 0)

        for i, key in enumerate(("response", "sin", "cos")):
            r_dict = results.iloc[results[key].idxmax()].to_dict()
            print(f"for {key}, {r_dict['k']}")
            img = construct_image(r_dict["k"])
            _cos = convolve2d(img, cosGabor, mode="valid")
            _sin = convolve2d(img, sinGabor, mode="valid")
            if key == "response":
                img = np.sqrt(_cos ** 2 + _sin ** 2)
            elif key == "cos":
                img = _cos
            elif key == "sin":
                img = _sin

            ax[0, i].set_title(
                f"{key.capitalize().replace('Response','Complex')} Response, k={r_dict['k']}",
                fontdict={"fontsize": 17, "fontweight": 20},
            )
            ax[0, i].imshow(img, cmap=plt.cm.viridis)
            ax[0, i].axes.get_xaxis().set_ticks([])
            ax[0, i].axes.get_yaxis().set_ticks([])

            ax[1, i].hist(img.flatten(), bins=50)

        plt.tight_layout()

        # plot the lineplot
        _, ax = plt.subplots(1, 1, dpi=200)

        plot = sns.lineplot(data=results, x="k", y="response", ax=ax)

        return fig, plot.get_figure()

    @property
    def part_b(self):
        (cosGabor, sinGabor) = make2DGabor(32, 4, 0)
        # 2 because its non-inclusive
        img = np.random.randint(0, 2, (N, N))
        _cos = convolve2d(img, cosGabor, mode="valid")
        _sin = convolve2d(img, sinGabor, mode="valid")
        response = np.sqrt(_cos ** 2 + _sin ** 2)

        fig, ax = plt.subplots(2, 3, dpi=200, figsize=(20, 10))

        for i, key in enumerate(("response", "sin", "cos")):

            if key == "response":
                img = response
                mean_response = np.sqrt((_cos.sum())**2+(_sin.sum())**2)
            elif key == "cos":
                img = _cos
                mean_response = _cos.sum()
            elif key == "sin":
                img = _sin
                mean_response = _sin.sum()

            ax[0, i].set_title(
                f"{key.capitalize().replace('Response','Complex')} Response: {mean_response:.2f}",
                fontdict={"fontsize": 17, "fontweight": 20},
            )
            ax[0, i].imshow(img, cmap=plt.cm.viridis)
            ax[0, i].axes.get_xaxis().set_ticks([])
            ax[0, i].axes.get_yaxis().set_ticks([])

            ax[1, i].hist(img.flatten(), bins=50)

        plt.tight_layout()
        return fig


if __name__ == "__main__":

    sns.set_theme()

    questions = Questions()

    part_a_fig, response_k = questions.part_a
    part_a_fig.savefig(PART_A_FIG)
    response_k.savefig(PART_A_RESPONSE_K)

    plt.clf()

    part_b_fig = questions.part_b
    part_b_fig.savefig(PART_B_FIG)


    """
    sns.set_theme()
    image = construct_image(37,(256,256))
    (cosGabor, sinGabor) = make2DGabor(32, 4, 0)

    cosSlice = cosGabor[cosGabor.shape[0]//2,:]
    cosSlice = np.tile(cosSlice,256//32)/cosSlice.max()

    sinSlice = sinGabor[sinGabor.shape[0]//2,:]
    sinSlice = np.tile(sinSlice,256//32)/sinSlice.max()

    imgSlice = image[image.shape[0]//2,:]

    data = pd.DataFrame({
        'location':np.tile(np.arange(256),3),
        'value':np.concatenate([imgSlice,sinSlice,cosSlice],axis=0),
        'type':['image']*256+['sin']*256+['cos']*256
    })    

    sns.set(rc={'figure.figsize':(15,7.5)})
    plot = sns.lineplot(data=data,x='location',y='value',hue='type')
    fig = plot.get_figure()
    plt.tight_layout()
    fig.savefig('results/q2/part_a_sliced_filter.png',dpi=200)
    """