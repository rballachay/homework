import math
import provided.helperclasses as hc
import glm
import igl
from typing import List
import numpy as np

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

epsilon = 10 ** (-4)


class Geometry:
    def __init__(self, name: str, gtype: str, materials: List[hc.Material]):
        self.name = name
        self.gtype = gtype
        self.materials = materials

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        return intersect


class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: List[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # note that you cannot assume a unit sphere as we did in class. this 
        # can be verified by looking at the objects in the scene json, radius isn't always 1
        # also, make the assumption that we aren't inside of the sphere

        oc = ray.origin - self.center  # Adjusted origin
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2
        discrim = b ** 2 - 4 * a * c

        if discrim >= 0:
            sqrt_discriminant = np.sqrt(discrim)
            t0 = (-b - sqrt_discriminant) / (2 * a)
            t1 = (-b + sqrt_discriminant) / (2 * a)

            t = min([t0,t1])

            if t>0 and t<intersect.time:
                intersect.time = t
                intersect.position = ray.origin + t * ray.direction
                intersect.normal =  normalized ((intersect.position - self.center) / self.radius)
                intersect.mat = self.materials[0]

class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: List[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        denom = np.dot(ray.direction, self.normal)
        if abs(denom) > 1e-6:  # To avoid division by zero
            t = np.dot(self.point - ray.origin, self.normal) / denom


            if t > 0 and t<intersect.time:
                intersect.time=t
                intersect.normal=self.normal
                intersect.position = ray.origin + t*ray.direction

                # we only consider the dimensions that are perpendicular to the normal
                mat_pos = np.abs(np.array(intersect.position,dtype=int)[np.array(self.normal) == 0])
                #print(self.materials[np.sum(mat_pos)%2])
                intersect.mat = self.materials[np.sum(mat_pos)%2]  # will need to add the material at some point 


class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: List[hc.Material], center: glm.vec3, dimension: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        halfside = dimension / 2
        self.minpos = center - halfside
        self.maxpos = center + halfside

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        # TODO: Create intersect code for Cube


class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: List[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str):
        super().__init__(name, gtype, materials)
        verts, _, norms, self.faces, _, _ = igl.read_obj(filepath)
        self.verts = []
        self.norms = []
        for v in verts:
            self.verts.append((glm.vec3(v[0], v[1], v[2]) + translate) * scale)
        for n in norms:
            self.norms.append(glm.vec3(n[0], n[1], n[2]))

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        # TODO: Create intersect code for Mesh


class Hierarchy(Geometry):
    def __init__(self, name: str, gtype: str, materials: List[hc.Material], t: glm.vec3, r: glm.vec3, s: glm.vec3):
        super().__init__(name, gtype, materials)
        self.t = t
        self.M = glm.mat4(1.0)
        self.Minv = glm.mat4(1.0)
        self.make_matrices(t, r, s)
        self.children: list[Geometry] = []

    def make_matrices(self, t: glm.vec3, r: glm.vec3, s: glm.vec3):
        self.M = glm.mat4(1.0)
        self.M = glm.translate(self.M, t)
        self.M = glm.rotate(self.M, glm.radians(r.x), glm.vec3(1, 0, 0))
        self.M = glm.rotate(self.M, glm.radians(r.y), glm.vec3(0, 1, 0))
        self.M = glm.rotate(self.M, glm.radians(r.z), glm.vec3(0, 0, 1))
        self.M = glm.scale(self.M, s)
        self.Minv = glm.inverse(self.M)
        self.t = t

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        # TODO: Create intersect code for Hierarchy


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)