import glm
import numpy as np

import provided_numba.geometry as geom
import provided_numba.helperclasses as hc
from typing import List
from provided_numba.render_nb import render_nb    

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
                 objects: list,
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
        image = render_nb(self.width, self.height, np.array(self.position), 
                     np.array(self.lookat), np.array(self.aspect), self.fov, np.array(self.up), 
                     self.ambient, self.objects, self.lights)            
        return image