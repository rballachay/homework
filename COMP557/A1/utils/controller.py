from utils.utils import (
    get_bboxes,
    project_orthograpic,
    draw_boxes,
    get_colors,
    convert_bboxes_triangle,
    convert_bboxes_normal,
)


class TestController:
    def __init__(self, test):
        self.__controller = {1: self.test_1, 2: self.test_2, 4: self.test_4}

        self.test_to_run = self.__controller[test]

        # properties of test 1
        self._colors = None

    def __call__(self, **kwargs):
        return self.test_to_run(**kwargs)

    # test 1 call and properties
    def test_1(self, T, Vt, pix, **kwargs):
        bboxes, _ = get_bboxes(T, Vt)
        bboxes_ortho = project_orthograpic(bboxes)
        draw_boxes(bboxes_ortho, pix, self.colors(T))

    def colors(self, T):
        if self._colors is None:
            self._colors = get_colors(T)
        return self._colors

    # test 2 call and properties
    def test_2(self, T, Vt, pix, depth, **kwargs):
        bboxes, triangles = get_bboxes(T, Vt)
        bboxes_ortho = project_orthograpic(bboxes)
        convert_bboxes_triangle(bboxes_ortho, triangles, pix, depth)

    # test 4 call and properties
    def test_4(self, T, Vt, pix, depth, N, TN, **kwargs):
        bboxes, triangles = get_bboxes(T, Vt)
        _, normals = get_bboxes(TN, N)
        bboxes_ortho = project_orthograpic(bboxes)
        convert_bboxes_normal(bboxes_ortho, triangles, pix, depth, normals)
