import math

import glm
import numpy as np

import provided.geometry as geom
import provided.helperclasses as hc
from typing import List
import taichi as ti

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

shadow_epsilon = 10**(-6)


class Scene:

    def __init__(self,
                 width: int,
                 height: int,
                 jitter: bool,
                 samples: int,
                 position: glm.vec3,
                 lookat: glm.vec3,
                 up: glm.vec3,
                 fov: float,
                 ambient: glm.vec3,
                 lights: List[hc.Light],
                 materials: List[hc.Material],
                 objects: List[geom.Geometry]
                 ):
        self.width = width  # width of image
        self.height = height  # height of image
        self.aspect = width / height  # aspect ratio
        self.jitter = jitter  # should rays be jittered
        self.samples = samples  # number of rays per pixel
        self.position = position  # camera position in 3D
        self.lookat = lookat  # camera look at vector
        self.up = up  # camera up position
        self.fov = fov  # camera field of view
        self.ambient = ambient  # ambient lighting
        self.lights = lights  # all lights in the scene
        self.materials = materials  # all materials of objects in the scene
        self.objects = objects  # all objects in the scene

    def render(self):
        image = np.zeros((self.width, self.height, 3))

        cam_dir = self.position - self.lookat
        d = 1.0
        top = d * math.tan(0.5 * math.pi * self.fov / 180)
        right = self.aspect * top
        bottom = -top
        left = -right

        w = glm.normalize(cam_dir)
        u = glm.cross(self.up, w)
        u = glm.normalize(u)
        v = glm.cross(w, u)

        for i in range(self.width):
            for j in range(self.height):
                colour = glm.vec3(0, 0, 0)

                for u_subpixel in range(self.samples):
                    for v_subpixel in range(self.samples):
                        u_offset = (u_subpixel + 0.5) / self.samples
                        v_offset = (v_subpixel + 0.5) / self.samples

                        # TODO: Generate rays
                        u_pix = left + (right-left)*(i+u_offset)/self.width
                        v_pix = bottom + (top-bottom)*(j+v_offset)/self.height
                        s = self.position+u_pix*u + v_pix*v - d*w
                        ray = hc.Ray(ti.math.vec3(np.array(self.position)), ti.math.vec3(np.array(s-self.position)))

                        # TODO: Test for intersection
                        intersection = hc.defaultIntersection()
                        for object in self.objects:
                            #if isinstance(object,geom.Plane):
                            #    continue
                            object.intersect(ray,intersection)

                        if sum(intersection.position)==0:
                            continue

                        # ambient has no relation to the light
                        ambient = intersection.mat.diffuse * self.ambient 
                        diffuse_factor = glm.vec3(0, 0, 0)
                        blinn_phong = glm.vec3(0, 0, 0)

                        for light in self.lights:
                            light_vector =  normalized(light.vector - intersection.position)

                            # TODO: Cast shadow ray
                            shadow_ray = hc.Ray(intersection.position+shadow_epsilon, light_vector)

                            in_shadow = False
                            for obj in self.objects:
                                shadow_intersection = hc.defaultIntersection()
                                obj.intersect(shadow_ray, shadow_intersection)
                                if sum(shadow_intersection.position) != 0 and shadow_intersection.time>0:
                                    in_shadow = True
                                    break
                            
                            if not in_shadow:
                                diffuse_factor+=intersection.mat.diffuse * light.colour * max(0, np.dot(light_vector, intersection.normal))
                                blinn_phong+=blinn_phong_specular_shading(light, intersection, glm.vec3(np.array(ray.direction)), light_vector)
                        
                        colour += ambient+diffuse_factor+blinn_phong

                colour /= self.samples** 2
                image[i, j, 0] = max(0.0, min(1.0, colour.x))
                image[i, j, 1] = max(0.0, min(1.0, colour.y))
                image[i, j, 2] = max(0.0, min(1.0, colour.z))

        return image


def blinn_phong_specular_shading(light, intersection, viewer_direction, light_vector):
    halfway_vector = normalized(light_vector + (-viewer_direction))
    specular_factor = max(0, np.dot(glm.vec3(intersection.normal), halfway_vector)) ** intersection.mat.hardness
    return intersection.mat.specular * light.colour * specular_factor

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)