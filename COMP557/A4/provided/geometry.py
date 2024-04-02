import provided.helperclasses as hc
import glm
import igl
from typing import List
import numpy as np
from copy import deepcopy

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
                # (0,0)->(1,1) should be first material 
                # (0,0)->(-1,-1) should be the first material
                # (0,0)->(0,1) should be second material
                # (0,0)->(1,0) should be second material
                mat_pos = np.array(intersect.position,dtype=np.float)[np.array(self.normal) == 0]
                mat_pos = [int(np.ceil(i)) for i in mat_pos]

                intersect.mat = self.materials[np.sum(mat_pos)%2]


class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: List[hc.Material], center: glm.vec3, dimension: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        halfside = dimension / 2
        self.minpos = center - halfside
        self.maxpos = center + halfside

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        tmin = (self.minpos - ray.origin) / ray.direction
        tmax = (self.maxpos - ray.origin) / ray.direction

        t_entry = np.max(np.minimum(tmin, tmax))
        t_exit = np.min(np.maximum(tmin, tmax))

        if (t_entry <= t_exit)  and (intersect.time>t_entry):
            # Intersection occurred
            intersect.time = t_entry
            intersect.position = ray.origin + t_entry * ray.direction
            # Calculate normal of the intersected face
            normal = self.calculate_normal(intersect.position)
            intersect.normal = normal
            intersect.mat = self.materials[0]
            return True
        return False

    def calculate_normal(self, intersection_point):
        normals = []
        for i in range(3):
            if np.isclose(intersection_point[i], self.minpos[i], atol=epsilon):
                normals.append(-glm.vec3(*np.eye(3)[i]))  # Negative normal along i-th axis
            elif np.isclose(intersection_point[i], self.maxpos[i], atol=epsilon):
                normals.append(glm.vec3(*np.eye(3)[i]))  # Positive normal along i-th axis
        if len(normals) > 1:
            return sum(normals, glm.vec3()) / len(normals)
        else:
            return normals[0]


class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: List[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str):
        super().__init__(name, gtype, materials)
        verts, _, norms, self.faces, _, _ = igl.read_obj(filepath)
        self.verts = []
        self.norms = []
        for v in verts:
            self.verts.append((glm.vec3(v[0], v[1], v[2]) + translate) * scale)
        
        if norms:
            for n in norms:
                self.norms.append(glm.vec3(n[0], n[1], n[2]))
        else:
            self.norms = [glm.vec3(0.0, 0.0, 0.0) for _ in range(len(verts))]
            
            # Calculate normals for each face and accumulate to vertices
            for face in self.faces:
                v0, v1, v2 = [self.verts[i] for i in face]
                e1 = v1 - v0
                e2 = v2 - v0
                normal = glm.normalize(glm.cross(e1, e2))
                for vertex_index in face:
                    self.norms[vertex_index] += normal

            # Normalize the accumulated normals
            for i in range(len(self.norms)):
                self.norms[i] = glm.normalize(self.norms[i])


    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        for face in self.faces:
            v0, v1, v2 = [self.verts[i] for i in face]
            e1 = v1 - v0
            e2 = v2 - v0
            pvec = glm.cross(ray.direction, e2)
            det = glm.dot(e1, pvec)

            if abs(det) < epsilon:
                continue

            inv_det = 1 / det
            tvec = ray.origin - v0
            u = glm.dot(tvec, pvec) * inv_det

            if u < 0 or u > 1:
                continue

            qvec = glm.cross(tvec, e1)
            v = glm.dot(ray.direction, qvec) * inv_det

            if v < 0 or u + v > 1:
                continue

            t = glm.dot(e2, qvec) * inv_det

            if t > epsilon and t < intersect.time:
                intersect.time = t
                intersect.position = ray.origin + t * ray.direction

                # Interpolate normals
                n0, n1, n2 = [self.norms[i] for i in face]
                normal = glm.normalize((1 - u - v) * n0 + u * n1 + v * n2)

                intersect.normal = normal
                intersect.mat = self.materials[0]
                



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
        origin_ = self.Minv*glm.vec4(ray.origin,1)
        direction = self.Minv*glm.vec4(ray.direction,0)
        origin = glm.vec3(origin_)/origin_.w
        newRay = hc.Ray(origin,glm.vec3(direction))

        global_didintersect = False
        for child in self.children:
            didintersect = child.intersect(newRay,intersect)
            
            if didintersect:
                normal = self.M*glm.vec4(intersect.normal,0)
                intersect.normal = glm.normalize(glm.vec3(normal))
                position = self.M*glm.vec4(intersect.position,1)
                intersect.position = glm.vec3(position)/position.w  
                global_didintersect=True

        return global_didintersect

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)