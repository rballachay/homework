import glm
import numba as nb
import numpy as np

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

@nb.experimental.jitclass([
    ('origin',  nb.float32[:]),
    ('direction', nb.float64[:]),# i have no idea in god why i can't downcast to float32
])
class Ray:
    def __init__(self, origin: glm.vec3, direction: glm.vec3):
        self.origin = origin
        self.direction = direction

    def getDistance(self, point: glm.vec3):
        return glm.length(point - self.origin)

    def getPoint(self, t: float):
        return self.origin + self.direction * t

@nb.experimental.jitclass([
    ('name', nb.types.string),
    ('specular', nb.float32[:]),
    ('diffuse', nb.float32[:]),
    ('hardness', nb.types.float64),
    ('ID', nb.int32),
    ('lookup', nb.int64[:,:,:]),
])
class Material:
    def __init__(self, name: str, specular: nb.float32[:], diffuse: nb.float32[:], hardness: float, ID: int, lookup):
        self.name = name
        self.specular = specular
        self.diffuse = diffuse
        self.hardness = hardness
        self.ID = ID
        self.lookup = lookup

@nb.jit(nopython=True)
def defaultMaterial():
    name = "default"
    specular = diffuse = nb.float32([0, 0, 0])
    hardness = 1.0
    ID=-1
    lookup=np.zeros((221,221,3)).astype(np.int64)
    return Material(name, specular, diffuse, hardness, ID, lookup)

@nb.experimental.jitclass([
    ('type', nb.types.string),
    ('name', nb.types.string),
    ('colour', nb.float32[:]),
    ('vector', nb.float32[:]),
    ('power', nb.types.float64),
])
class Light:
    def __init__(self, ltype: str, name: str, colour: nb.float32[:], vector: nb.float32[:], power: float):
        self.type = ltype
        self.name = name
        self.colour = colour
        self.vector = vector
        self.power = power

@nb.experimental.jitclass([
    ('time', nb.types.float64),
    ('normal', nb.float32[:]),
    ('position', nb.float32[:]),
    ('mat', Material.class_type.instance_type),
])
class Intersection:
    def __init__(self, time: float, normal: glm.vec3, position: glm.vec3, material: Material):
        self.time = time
        self.normal = normal
        self.position = position
        self.mat = material

@nb.jit(nopython=True)
def defaultIntersection():
    time = np.inf
    normal = nb.float32([0, 0, 0])
    position = nb.float32([0, 0, 0])
    mat = defaultMaterial()
    return Intersection(time, normal, position, mat)
