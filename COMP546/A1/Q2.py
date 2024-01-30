import numpy as np
from PIL import Image
import scipy.signal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

RESULTS_A = "results/q2_contrast_plot.png"
RESULTS_B = "results/q2_rms_contrast_plot.png"

IMG_DIM = (256 * 2, 256 * 2)
N = 256


def construct_image(k, dims=IMG_DIM, n=N):
    image = np.zeros(dims)
    for x in range(dims[0]):
        image[x, :] = 0.5 * (1 + np.sin((2 * np.pi / n) * k * x))
    return image


def gaussian_kernel_2d(sigma, size):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2))
        * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)),
        (size, size),
    )
    return kernel / np.sum(kernel)


def michelson_contrast(image):
    max_val = np.max(image[20:-20, 20:-20])
    min_val = np.min(image[20:-20, 20:-20])
    return (max_val - min_val) / (max_val + min_val)


def rms_contrast(image):
    image = image[20:-20, 20:-20].flatten()
    return np.sqrt(np.sum((image - image.mean()) ** 2) / image.shape[0])


class Questions:
    @classmethod
    def a(cls) -> plt.figure:
        plot_obj = []
        for k in range(1, int(N / 2) + 1):
            for stdev in [5, 4]:
                image = construct_image(k=k)

                gauss = gaussian_kernel_2d(stdev, 8 * stdev)

                # half the width of gaussian boundary is at least 20 away,
                # so we are going to calculate 20 in
                image = scipy.signal.convolve2d(image, gauss, mode="valid")

                m_contrast = michelson_contrast(image)

                plot_obj.append({"contrast": m_contrast, "k": k, "stdev": stdev})
        data = pd.DataFrame(plot_obj)
        lineplot = sns.lineplot(data=data, x="k", y="contrast", hue="stdev")
        fig = lineplot.get_figure()
        return fig

    @classmethod
    def b(cls) -> plt.figure:
        plot_obj = []
        for k in range(1, int(N / 2) + 1):

            image = construct_image(k=k)
            dog = gaussian_kernel_2d(4, 40) - gaussian_kernel_2d(5, 40)

            # half the width of gaussian boundary is at least 20 away,
            # so we are going to calculate 20 in
            image = scipy.signal.convolve2d(image, dog, mode="valid")

            contrast = rms_contrast(image)

            plot_obj.append(
                {
                    "contrast": contrast,
                    "k": k,
                }
            )
        data = pd.DataFrame(plot_obj)

        lineplot = sns.lineplot(data=data, x="k", y="contrast")
        fig = lineplot.get_figure()
        return fig


if __name__ == "__main__":
    fig = Questions.a()
    fig.savefig(RESULTS_A, dpi=200)

    fig = Questions.b()
    fig.savefig(RESULTS_B, dpi=200)
