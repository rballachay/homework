import numpy as np
import scipy.signal
from PIL import Image
import matplotlib.pyplot as plt

RESULTS_TEST = "results/q3_test_double_opp.png"

IMG_DIMS = (400, 400)


def gaussian_kernel_2d(sigma, size):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2))
        * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)),
        (size, size),
    )
    return kernel / np.sum(kernel)


def construct_example_img(img_dims=IMG_DIMS):
    image = np.zeros((*img_dims, 3))
    half = img_dims[0] // 2
    size = half // 8
    image[:, half:] = 255
    image[
        half - size // 2 : half + size // 2,
        half // 2 - size // 2 : half // 2 + size // 2,
    ] = (
        255 / 2
    )
    image[
        half - size // 2 : half + size // 2,
        half // 2 - size // 2 + half : half // 2 + size // 2 + half,
    ] = (
        255 / 2
    )
    return image


def construct_rgb_img(img_dims=IMG_DIMS):
    image = np.zeros((*img_dims, 3))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 0), (255, 255, 255)]
    loc = img_dims[0] // 4
    size = img_dims[0] // 32
    for i, color in enumerate(colors):
        other_colors = set(colors) - {color}
        image[:, loc * i : loc * (i + 1), :] = color

        for j, other_color in enumerate(other_colors):
            image[
                loc * (j + 1) - size // 2 : loc * (j + 1) + size // 2,
                loc * (i * 2 + 1) // 2 - size // 2 : loc * (i * 2 + 1) // 2 + size // 2,
            ] = other_color
    #image = Image.fromarray(np.uint8(image))
    return np.uint8(image)


if __name__ == "__main__":
    dog = gaussian_kernel_2d(4, 40) - gaussian_kernel_2d(5, 40)
    double_opp = np.stack([dog, -dog, np.zeros_like(dog)], axis=-1)
    # image = scipy.signal.convolve2d(image,dog, mode='valid')

    image = construct_example_img()
    image = scipy.signal.convolve(image, double_opp, mode="valid")[..., 0]
    image = Image.fromarray(np.uint8(image))
    image.save(RESULTS_TEST)

    image = construct_rgb_img()

    image = scipy.signal.convolve(image, double_opp, mode="valid")[..., 0]

    #plt.imshow(image, cmap="viridis")
    plt.imsave( 'temp.png',image,cmap="viridis")
