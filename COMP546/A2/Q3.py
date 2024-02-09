import numpy as np
from utils.make2DGabor import make2DGabor
from PIL import Image
from scipy.signal import convolve2d
import os
import pandas as pd
import seaborn as sns

RESULTS = "results/q3/response_results.png"

MAX_DISPARITY = 8
N = 256


def get_binocular(theta, disparity):
    """left gabor will be shifted by disparity to the right
    relative to the right cell
    """
    right_cells = make2DGabor(
        32, 4 * np.cos(np.deg2rad(theta)), 4 * np.sin(np.deg2rad(theta))
    )

    left_cells = []
    for right_cell in right_cells:
        if disparity > 0:
            left_cell = np.pad(
                right_cell, [(0, 0), (disparity, 0)], "constant", constant_values=(0, 0)
            )
            left_cells.append(left_cell[:, :-disparity])
        elif disparity < 0:
            left_cell = np.pad(
                right_cell,
                [(0, 0), (0, -disparity)],
                "constant",
                constant_values=(0, 0),
            )
            left_cells.append(left_cell[:, -disparity:])
        else:
            left_cells.append(right_cell)

    return tuple(right_cells), tuple(left_cells)


def get_stereo_images(disparity, n=N):
    img_right = np.random.randint(0, 2, (n, n))
    # img_right = np.zeros((n,n))
    # img_right[:,:img_right.shape[0]//2] = 1
    img_left = np.roll(img_right, disparity, axis=1)
    return img_right, img_left


class Questions:
    cache = "results/q3/response_cache.csv"

    @property
    def results(self):
        if not os.path.exists(self.cache):
            results = {"img_shift": [], "theta": [], "gabor_shift": [], "response": []}
            images = ((i, get_stereo_images(i)) for i in [0, 2])
            for i_shift, (img_r, img_l) in images:
                for angle in (0, 45, 90, 135):
                    for disparity in np.arange(-MAX_DISPARITY, MAX_DISPARITY + 1):
                        right_cells, left_cells = get_binocular(angle, disparity)

                        cosGaborR, sinGaborR = right_cells
                        cosGaborL, sinGaborL = left_cells

                        _cos = (
                            convolve2d(img_l, cosGaborL, mode="valid").sum()
                            - convolve2d(img_r, cosGaborR, mode="valid").sum()
                        )
                        _sin = (
                            convolve2d(img_l, sinGaborL, mode="valid").sum()
                            - convolve2d(img_r, sinGaborR, mode="valid").sum()
                        )
                        response = np.sqrt(_cos ** 2 + _sin ** 2)

                        # results
                        results["img_shift"].append(i_shift)
                        results["theta"].append(angle)
                        results["gabor_shift"].append(disparity)
                        results["response"].append(response)

            results = pd.DataFrame(results)
            results.to_csv(self.cache, index=False)
        else:
            results = pd.read_csv(self.cache)

        plot = sns.relplot(
            data=results,
            kind="scatter",
            x="gabor_shift",
            y="response",
            col="img_shift",
            hue="theta",
        )

        return plot.fig


if __name__ == "__main__":
    sns.set_theme()

    questions = Questions()

    fig = questions.results
    fig.savefig(RESULTS)

    """
    cells_r,cells_l = get_binocular(90,0)
    cells_r = np.concatenate(cells_r,axis=1)
    cells_l = np.concatenate(cells_l,axis=1)
    cells = np.concatenate([cells_r,cells_l],axis=0)
    cells = ((cells-cells.mean()) / cells.var() )+1
    image = Image.fromarray(np.uint8(cells) * 255)
    image.save("temp_bino.png")

    imgs = get_stereo_images(2)
    img = np.concatenate(imgs,axis=0)
    image = Image.fromarray(np.uint8(img) * 255)
    image.save("temp.png")
    """
