import numpy as np
import scipy.signal
from PIL import Image
import matplotlib.pyplot as plt

RESULTS = "results/q3_double_opp.png"
RESULTS_OG = "results/q3_original.png"

IMG_DIMS = (500, 500)


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
    # image = Image.fromarray(np.uint8(image))
    return np.uint8(image)


if __name__ == "__main__":
    dog = gaussian_kernel_2d(4, 40) - gaussian_kernel_2d(5, 40)
    double_opp = np.stack([dog, -dog, np.zeros_like(dog)], axis=-1)

    image = construct_rgb_img()
    image_og = Image.fromarray(np.uint8(image)).convert("RGB")
    image_og.save(RESULTS_OG)

    image_red = scipy.signal.convolve2d(image[..., 0], double_opp[..., 0], mode="valid")
    image_green = scipy.signal.convolve2d(
        image[..., 1], double_opp[..., 1], mode="valid"
    )
    image = image_red + image_green

    fig = plt.figure(figsize=(10, 10))  # Your image (W)idth and (H)eight in inches
    # Stretch image to full figure, removing "grey region"
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    im = plt.imshow(image, cmap="viridis")  # Show the image
    pos = fig.add_axes([0.93, 0.1, 0.02, 0.35])  # Set colorbar position in fig
    fig.colorbar(im, cax=pos)  # Create the colorbar
    plt.savefig(RESULTS)
