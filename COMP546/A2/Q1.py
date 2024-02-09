import numpy as np
from scipy.ndimage import rotate
from PIL import Image
from utils.make2DGabor import make2DGabor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PART_A_IMAGE = "results/q1/part_a_image.png"
PART_A_RESULTS = "results/q1/part_a_response.png"
PART_B_IMAGE = "results/q1/part_b_image.png"
PART_B_RESULTS = "results/q1/part_b_response.png"


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_ridge_1d(n, sigma, rotation_degrees):
    # Create a one-dimensional grid of coordinates
    x = np.linspace(0.99, 0, n // 2)
    x = np.concatenate([x, x[::-1]])

    # Calculate Gaussian ridge values
    ridge = gaussian(x, 0, sigma)
    ridge = np.expand_dims(ridge, axis=1)

    bar = np.tile(ridge, n)
    return rotate(bar, rotation_degrees, mode="nearest", order=0, reshape=False)


def oriented_edge(n, rotation_degrees):
    # Create a one-dimensional grid of coordinates
    x = np.zeros(n // 2)
    edge = np.concatenate([x, 1 + x])

    edge = np.expand_dims(edge, axis=1)
    bar = np.tile(edge, n)
    return rotate(bar, rotation_degrees, mode="nearest", order=0, reshape=False)


class Questions:
    @property
    def part_a(cls):
        (cosGabor, sinGabor) = make2DGabor(32, 4, 0)
        n = 32  # Size of the grid
        sigma = 0.2  # Width of the ridge

        results = {"rotation": [], "response": []}
        bars = []
        for rot in np.arange(0, 180, 15):
            ridge = gaussian_ridge_1d(n, sigma, rot)
            response = np.sqrt(
                ((ridge * cosGabor).sum()) ** 2 + ((ridge * sinGabor).sum()) ** 2
            )
            results["rotation"].append(rot)
            results["response"].append(response)
            bars.append(ridge)

        results = pd.DataFrame(results)
        plot = sns.lineplot(data=results, x="rotation", y="response")

        # arrange the bars in grid
        bars = cls.make_grid(bars)
        return plot.get_figure(), bars

    @property
    def part_b(cls):
        (cosGabor, sinGabor) = make2DGabor(32, 4, 0)
        n = 32  # Size of the grid

        results = {"rotation": [], "response": []}
        edges = []
        for rot in np.arange(0, 180, 15):
            ridge = oriented_edge(n, rot)
            response = np.sqrt(
                ((ridge * cosGabor).sum()) ** 2 + ((ridge * sinGabor).sum()) ** 2
            )
            results["rotation"].append(rot)
            results["response"].append(response)
            edges.append(ridge)

        results = pd.DataFrame(results)
        plot = sns.lineplot(data=results, x="rotation", y="response")

        # arrange the bars in grid
        edges = cls.make_grid(edges)
        return plot.get_figure(), edges

    @classmethod
    def make_grid(cls, bars):
        bars = [arr.reshape(32, 32, 1) for arr in bars]
        bars = [np.hstack(bars[i : i + 4]) for i in range(0, len(bars), 4)]
        bars = np.vstack(bars)[..., 0]
        bars = Image.fromarray(np.uint8(bars) * 255)
        return bars


if __name__ == "__main__":
    sns.set_theme()

    questions = Questions()

    part_a_fig, part_a_img = questions.part_a
    part_a_fig.savefig(PART_A_RESULTS)
    part_a_img.save(PART_A_IMAGE)

    # clear active fig
    plt.clf()

    part_b_fig, part_b_img = questions.part_b
    part_b_fig.savefig(PART_B_RESULTS)
    part_b_img.save(PART_B_IMAGE)

    """
    # Example usage
    n = 32  # Size of the grid
    sigma = 0.2 # Width of the ridge
    rotation_degrees = 45  # Rotation angle in degrees

    gaussian_ridge_image = gaussian_ridge_1d(n, sigma, rotation_degrees)
    oriented_edge_image = oriented_edge(n,15) 

    image = Image.fromarray(np.uint8(oriented_edge_image)*255)
    image.save('temp_ride.png')
    """
