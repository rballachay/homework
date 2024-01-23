from typing import Tuple, List
import numpy as np
import itertools


def get_bboxes(points: Tuple[int, int, int], vertices: List[Tuple[int, int, int]]):
    bboxes = []
    triangles = []
    for point in points:
        triangle = []

        for i in range(3):
            triangle.append(vertices[point[i]])

        tri_stack = np.stack(triangle)
        min_dim = tri_stack.min(axis=0)
        max_dim = tri_stack.max(axis=0)
        bboxes.append(np.stack([min_dim, max_dim]))
        triangles.append(triangle)
    return bboxes, triangles


def project_orthograpic(bboxes: List[np.ndarray]):
    ortho_bboxes = []
    for coords in bboxes:
        projected = coords[:, :2]
        ortho_bboxes.append(projected)

    return ortho_bboxes


def draw_boxes(
    bboxes_ortho: List[np.ndarray], pix_buffer: np.ndarray, colors: np.ndarray
):

    for i, box in enumerate(bboxes_ortho):
        box = box.round().astype(int)
        pix_buffer[box[0, 0] : box[1, 0], box[0, 1] : box[1, 1], :] = colors[i, :] / 255


def get_colors(vertices: List[Tuple[int, int, int]]):
    n_triangles = len(vertices)
    colors = np.random.choice(range(256), size=(n_triangles, 3)).astype(int)
    return colors


def convert_bboxes_triangle(
    bboxes: List[np.ndarray],
    triangles: List[np.ndarray],
    pix: np.ndarray,
    depth: np.ndarray,
):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    assert len(bboxes) == len(triangles)
    for bbox, triangle in list(zip(bboxes, triangles)):
        coords = list(
            itertools.product(
                range(int(np.floor(bbox[0, 0])), int(np.ceil(bbox[1, 0]))),
                range(int(np.floor(bbox[0, 1])), int(np.ceil(bbox[1, 1]))),
            )
        )
        for x, y in coords:
            u, v, w = cartesian_to_barycentric(x, y, triangle)
            if any([i < 0 for i in (u, v, w)]):
                continue
            color = interpolate_color((u, v, w), colors)

            # interpolate z value instead of the color
            _, _, z = interpolate_color((u, v, w), triangle)

            if z >= depth[x, y, 0]:
                depth[x, y] = z
                pix[x, y, :] = color / 255


def convert_bboxes_normal(
    bboxes: List[np.ndarray],
    triangles: List[np.ndarray],
    pix: np.ndarray,
    depth: np.ndarray,
    normals: List[np.ndarray],
):
    assert len(bboxes) == len(triangles)
    for bbox, triangle, normal in list(zip(bboxes, triangles, normals)):
        coords = list(
            itertools.product(
                range(int(np.floor(bbox[0, 0])), int(np.ceil(bbox[1, 0]))),
                range(int(np.floor(bbox[0, 1])), int(np.ceil(bbox[1, 1]))),
            )
        )
        for x, y in coords:
            u, v, w = cartesian_to_barycentric(x, y, triangle)
            if any([i < 0 for i in (u, v, w)]):
                continue

            # interpolate the normal
            color = interpolate_color((u, v, w), normal) / 2 + 1 / 2

            # interpolate z value instead of the color
            _, _, z = interpolate_color((u, v, w), triangle)

            if z >= depth[x, y, 0]:
                print(color)
                depth[x, y] = z
                pix[x, y, :] = color[-1]


def cartesian_to_barycentric(x, y, triangle):
    """
    Convert Cartesian coordinates (x, y) to barycentric coordinates (u, v, w)
    with respect to a triangle defined by its vertices.

    Parameters:
        x, y (float): Cartesian coordinates of the point.
        triangle (list of tuples): Vertices of the triangle [(Ax, Ay), (Bx, By), (Cx, Cy)].

    Returns:
        tuple: Barycentric coordinates (u, v, w).
    """
    x1, y1, _ = triangle[0]
    x2, y2, _ = triangle[1]
    x3, y3, _ = triangle[2]

    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

    u = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
    v = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
    w = 1 - u - v

    return u, v, w


def interpolate_color(barycentric_coords, vertex_colors):
    """
    Interpolate color at a point within a triangle using barycentric coordinates.

    Parameters:
        barycentric_coords (tuple): Barycentric coordinates (u, v, w).
        vertex_colors (list of tuples): Colors at the vertices of the triangle [(r1, g1, b1), (r2, g2, b2), (r3, g3, b3)].

    Returns:
        tuple: Interpolated color (r, g, b).
    """
    u, v, w = barycentric_coords

    color = (
        u * vertex_colors[0][0] + v * vertex_colors[1][0] + w * vertex_colors[2][0],
        u * vertex_colors[0][1] + v * vertex_colors[1][1] + w * vertex_colors[2][1],
        u * vertex_colors[0][2] + v * vertex_colors[1][2] + w * vertex_colors[2][2],
    )

    return np.array([component for component in color])


def calculate_triangle_normal(triangle):
    """
    Calculate the normal vector of a triangle in 3D.

    Parameters:
        triangle (numpy.ndarray): 2D NumPy array representing a triangle with vertices in rows.

    Returns:
        numpy.ndarray: Normal vector of the triangle.
    """
    if triangle.shape != (3, 3):
        raise ValueError("Input must be a 2D NumPy array with shape (3, 3).")

    # Form vectors from two edges of the triangle
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]

    # Calculate the cross product to get the normal vector
    normal = np.cross(edge1, edge2)

    # Normalize the normal vector
    normal /= np.linalg.norm(normal)

    return normal


def calculate_normals(V: np.ndarray, T: np.ndarray, TN: np.ndarray):
    normals = []
    for point in T:
        triangle = []

        # only need two of the indexes
        for i in range(3):
            triangle.append(V[point[i]])
        normal = calculate_triangle_normal(np.stack(triangle))
        normals.append(normal)

    N = np.stack(normals)
    rows, cols = N.shape
    TN = np.tile(np.arange(rows), (cols, 1)).T
    return N, TN
