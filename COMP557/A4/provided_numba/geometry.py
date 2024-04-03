import provided_numba.helperclasses as hc
import glm
from typing import List
import numpy as np
import numba as nb
import typing as pt


# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

epsilon = 10 ** (-4)


@nb.experimental.jitclass([
    ('name', nb.types.string),
    ('gtype', nb.types.string),
    ('materials',nb.types.ListType(hc.Material.class_type.instance_type)),
    ('center', nb.float32[:]),
    ('radius', nb.types.float64),
])
class Sphere:
    def __init__(self, name: str, gtype: str, materials: List[hc.Material], center: glm.vec3, radius: float):
        self.name = name
        self.gtype = gtype 
        self.materials = materials
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection, M:np.ndarray):
        # note that you cannot assume a unit sphere as we did in class. this 
        # can be verified by looking at the objects in the scene json, radius isn't always 1
        # also, make the assumption that we aren't inside of the sphere

        oc = ray.origin - self.center  # Adjusted origin
        a = np.dot(ray.direction.astype(np.float64), ray.direction.astype(np.float64))
        b = 2.0 * np.dot(oc.astype(np.float64), ray.direction.astype(np.float64))
        c = np.dot(oc, oc) - self.radius ** 2
        discrim = b ** 2 - 4 * a * c

        if discrim >= 0:
            sqrt_discriminant = np.sqrt(discrim)
            t0 = (-b - sqrt_discriminant) / (2 * a)
            t1 = (-b + sqrt_discriminant) / (2 * a)

            t = np.min(np.array([t0,t1],dtype=np.float32))

            if t>0 and t<intersect.time:
                intersect.time = np.float32(t)
                intersect.position = (ray.origin + t * ray.direction).astype(np.float32)
                intersect.normal =  normalized ((intersect.position - self.center) / self.radius).astype(np.float32)
                intersect.mat = self.materials[0]

@nb.experimental.jitclass([
    ('name', nb.types.string),
    ('gtype', nb.types.string),
    ('materials',nb.types.ListType(hc.Material.class_type.instance_type)),
    ('point', nb.float32[:]),
    ('normal', nb.float32[:]),
])
class Plane:
    def __init__(self, name: str, gtype: str, materials: pt.List[hc.Material], point: nb.float32[:], normal: nb.float32[:]):
        self.name = name
        self.gtype = gtype
        self.materials = materials
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection, M:np.ndarray):
        direction = ray.direction.astype(np.float64)
        normal = self.normal.astype(np.float64)
        denom = np.dot(direction, normal)
        if abs(denom) > 1e-6:  # To avoid division by zero
            t = np.dot(self.point - ray.origin, self.normal) / denom

            if t > 0 and t<intersect.time:
                intersect.time=t
                intersect.normal=self.normal
                intersect.position = (ray.origin + t*ray.direction).astype(np.float32)

                # we only consider the dimensions that are perpendicular to the normal
                # (0,0)->(1,1) should be first material 
                # (0,0)->(-1,-1) should be the first material
                # (0,0)->(0,1) should be second material
                # (0,0)->(1,0) should be second material
                mat_pos = intersect.position.astype(np.float64)[self.normal == 0]
                mat_pos = np.array([int(np.ceil(i)) for i in mat_pos])
                intersect.mat = self.materials[np.sum(mat_pos)%2]

@nb.experimental.jitclass([
    ('name', nb.types.string),
    ('gtype', nb.types.string),
    ('materials',nb.types.ListType(hc.Material.class_type.instance_type)),
    ('minpos', nb.types.float32[:]),
    ('maxpos', nb.types.float32[:]),
])
class AABB:
    def __init__(self, name: str, gtype: str, materials: List[hc.Material], center: glm.vec3, dimension: glm.vec3):
        # dimension holds information for length of each size of the box
        self.name = name
        self.gtype = gtype
        self.materials = materials
        halfside = dimension / 2.0
        self.minpos = (center - halfside).astype(np.float32)
        self.maxpos = (center + halfside).astype(np.float32)

    def intersect(self, ray, intersection, M_in):
        M = M_in[:,:,0]
        for i in range(1,M_in.shape[-1]):
            M = np.dot(M, M_in[:,:,i])
        
        Minv =  np.linalg.inv(M)

        # Transform the ray
        origin_homogeneous = np.append(ray.origin, 1)
        origin_transformed = np.dot(Minv, origin_homogeneous)
        origin = origin_transformed[:3] / origin_transformed[3]

        direction_homogeneous = np.append(ray.direction, 0)
        direction_transformed = np.dot(Minv, direction_homogeneous)
        direction = direction_transformed[:3]

        new_ray = hc.Ray(origin.astype(np.float32), direction.astype(np.float64))

        # Check for intersection with the transformed ray
        did_intersect = self.check_intersect(new_ray, intersection)

        if did_intersect:
            # Transform intersection properties back to the original coordinate system
            normal_homogeneous = np.append(intersection.normal, 0)
            normal_transformed = np.dot(Minv.T, normal_homogeneous)
            intersection.normal = normalized(normal_transformed[:3]).astype(np.float32)

            position_homogeneous = np.append(intersection.position, 1)
            position_transformed = np.dot(M, position_homogeneous)
            intersection.position = (position_transformed[:3] / position_transformed[3]).astype(np.float32)

    def check_intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        tmin = (self.minpos - ray.origin) / ray.direction
        tmax = (self.maxpos - ray.origin) / ray.direction

        t_entry = np.max(np.minimum(tmin, tmax))
        t_exit = np.min(np.maximum(tmin, tmax))

        if (t_entry <= t_exit)  and (intersect.time>t_entry):
            # Intersection occurred
            intersect.time = t_entry
            intersect.position = (ray.origin + t_entry * ray.direction).astype(np.float32)
            # Calculate normal of the intersected face
            normal = self.calculate_normal(intersect.position)
            intersect.normal = normal.astype(np.float32)
            intersect.mat = self.materials[0]
            return True
        return False

    def calculate_normal(self, intersection_point):
        normals = []
        for i in range(3):
            if np.isclose(intersection_point[i], self.minpos[i], atol=epsilon):
                normals.append(-np.eye(3)[i])  # Negative normal along i-th axis
            elif np.isclose(intersection_point[i], self.maxpos[i], atol=epsilon):
                normals.append(np.eye(3)[i]) # Positive normal along i-th axis
        if len(normals) > 1:
            return sum(normals, np.zeros(3)) / len(normals)
        else:
            return normals[0]

@nb.experimental.jitclass([
    ('name', nb.types.string),
    ('gtype', nb.types.string),
    ('materials',nb.types.ListType(hc.Material.class_type.instance_type)),
    ('faces', nb.types.ListType(nb.int32[:])),
    ('verts', nb.types.ListType(nb.float32[:])),
    ('norms', nb.types.ListType(nb.float32[:])),
    ('t_coords', nb.float64[:,:]),
    ('t_faces',nb.int64[:,:]),
])
class Mesh:
    def __init__(self, name: str, gtype: str, materials: List[hc.Material], translate: glm.vec3, scale: float,
                 verts: str, norms, faces, t_coords, t_faces):
        self.name = name
        self.gtype = gtype
        self.materials = materials
        self.verts = nb.typed.List.empty_list(nb.float32[:])
        self.norms = nb.typed.List.empty_list(nb.float32[:])
        self.faces = nb.typed.List.empty_list(nb.int32[:])
        self.t_coords = t_coords
        self.t_faces = t_faces

        for f in faces:
            self.faces.append(f.astype(np.int32))

        for v in verts:
            self.verts.append((( v + translate) * scale).astype(np.float32))
        
        # this will be an empty list otherwise
        #if norms.size!=0:
        #    for n in norms:
        #        self.norms.append(n.astype(np.float32))
        #else:
        for _ in range(len(verts)):
            self.norms.append(np.array([0.0, 0.0, 0.0]).astype(np.float32))
        
        # Calculate normals for each face and accumulate to vertices
        for face in self.faces:
            v0, v1, v2 = [self.verts[i] for i in face]
            e1 = v1 - v0
            e2 = v2 - v0
            normal = normalized(np.cross(e1, e2))
            for vertex_index in face:
                self.norms[vertex_index] += normal

        # Normalize the accumulated normals
        for i in range(len(self.norms)):
            self.norms[i] = normalized(self.norms[i])


    def intersect(self, ray, intersect, M:np.ndarray):
        for k, face in enumerate(self.faces):
            v0, v1, v2 = [self.verts[i] for i in face]
            e1 = v1 - v0
            e2 = v2 - v0
            pvec = np.cross(ray.direction, e2).astype(np.float32)
            det = np.dot(e1, pvec)

            if np.abs(det) < epsilon:
                continue

            inv_det = 1.0 / det
            tvec = ray.origin - v0
            u = np.dot(tvec, pvec) * inv_det

            if u < 0 or u > 1:
                continue

            qvec = np.cross(tvec, e1).astype(np.float32)
            v = np.dot(ray.direction.astype(np.float32), qvec) * np.float32(inv_det)

            if v < 0 or u + v > 1:
                continue

            t = np.dot(e2, qvec) * inv_det

            if t > epsilon and t < intersect.time:
                intersect.time = t
                intersect.position = (ray.origin + t * ray.direction.astype(np.float32)).astype(np.float32)

                # Interpolate normals
                n0, n1, n2 = [self.norms[i] for i in face]
                normal = normalized((1 - u - v) * n0 + u * n1 + v * n2)

                intersect.normal = normal.astype(np.float32)

                intersect.mat =  self.get_material(k, u, v)

    def get_material(self, k, u, v):
        if np.all(self.materials[0].lookup==0):
            return self.materials[0]

        if k>231:
            material=np.array([1.0,1.0,1.0])
        else:
            # otherwise, we can use our face number to look up our texture coordinates
            n0, n1, n2 = [self.t_coords[j] for j in self.t_faces[k]]
            normal = (((1 - u - v) * n0 + u * n1 + v * n2)*220).astype(np.int64)

            # this is negative for some reason
            material = self.materials[0].lookup[normal[0],normal[1]]/255.0

        return hc.Material("texture-surface",np.array([0.7,0.7,0.7]).astype(np.float32),
                           material.astype(np.float32),15,1,np.zeros((221,221,1)).astype(np.int64))

    
@nb.experimental.jitclass([
    ('spheres', nb.types.ListType(Sphere.class_type.instance_type)),
    ('planes', nb.types.ListType(Plane.class_type.instance_type)),
    ('meshes', nb.types.ListType(Mesh.class_type.instance_type)),
    ('boxes', nb.types.ListType(AABB.class_type.instance_type)),
    ('sphereTransforms', nb.types.ListType(nb.float64[:,:,:])),
    ('planeTransforms', nb.types.ListType(nb.float64[:,:,:])),
    ('meshTransforms', nb.types.ListType(nb.float64[:,:,:])),
    ('boxTransforms', nb.types.ListType(nb.float64[:,:,:])),
]) 
class ObjectContainer:
    def __init__(self, spheres, planes, meshes, boxes, 
                 sphereTransforms, planeTransforms, meshTransforms,
                 boxTransforms):
        self.spheres = spheres
        self.planes = planes
        self.meshes = meshes
        self.boxes = boxes
        self.sphereTransforms = sphereTransforms
        self.planeTransforms = planeTransforms
        self.meshTransforms = meshTransforms
        self.boxTransforms = boxTransforms

@nb.jit(nopython=True)
def normalized(a):
    a=a.flatten()
    return a / np.sqrt(np.sum(a**2))
