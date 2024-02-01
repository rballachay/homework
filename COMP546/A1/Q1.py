import numpy as np
from PIL import Image

RESULTS = "results/q1_visual_direction_angle.png"

# have a cube that is 500x500x500
FIELD_WIDTH = FIELD_HEIGHT = FIELD_LENGTH = 100000
N_POLES = 10000
MAX_HEIGHT = 500
DISTANCE = 100
VIEWER_HEIGHT = 600


def draw_image(
    n_poles: int, field: tuple, max_height: int, viewer_height: int
) -> Image:
    """
    Draw our image. We are using numpy to create the lines.

    Steps:
        1. Pick n random coordinates in x-y axis with height z.
            This is where we are going to be drawing our poles.
        2. Calculate the distance away from the camera and scale
            the height accordingly.
    """

    points = []

    # create poles in x-y with random height
    for _ in range(n_poles):
        half_width = int(field[0] / 2)

        # distance away from the viewer can only be positive
        z_loc = np.random.randint(1, DISTANCE - 25)

        # our x location
        x_loc = np.random.randint(-half_width, half_width)

        # our y location is relative to our viewer
        y_loc_0 = np.random.randint(0, max_height // 8) - max_height
        y_loc_1 = np.random.randint(y_loc_0, max_height) - max_height

        points.append((z_loc, x_loc, y_loc_0, y_loc_1))

    image = np.full((max_height, max_height), 255)
    for Z, X, Y, H in points:
        # now, transform the points
        x = max_height - (int(X / Z) + 250)
        y = max_height - (int(Y / Z) + int(500 * (max_height / viewer_height)))
        h = max_height - (int(H / Z) + int(500 * (max_height / viewer_height)))

        try:
            image[y:h, x] = 0
        except:
            continue

    return Image.fromarray(np.uint8(image))


if __name__ == "__main__":
    image = draw_image(
        n_poles=N_POLES,
        field=(FIELD_WIDTH, FIELD_HEIGHT, FIELD_LENGTH),
        max_height=MAX_HEIGHT,
        viewer_height=VIEWER_HEIGHT,
    )
    image.save(RESULTS)
