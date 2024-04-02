import glm
import taichi as ti

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

@ti.dataclass
class Ray:
    origin: ti.math.vec3
    direction: ti.math.vec3

    @ti.func
    def getDistance(self, point: ti.math.vec3):
        return ti.length(point - self.origin)

    @ti.func
    def getPoint(self, t: float):
        return self.origin + self.direction * t

@ti.dataclass
class Material:
    #    def __init__(self, name: str, specular: glm.vec3, diffuse: glm.vec3, hardness: float, ID: int):
    #name: ti.str
    specular: ti.math.vec3
    diffuse: ti.math.vec3
    hardness: ti.f32
    ID: ti.i32


def defaultMaterial():
    specular = diffuse = ti.math.vec3(0, 0, 0)
    hardness = ID = -1
    return Material(specular, diffuse, hardness, ID)

@ti.dataclass
class Light:
    #def __init__(self, ltype: str, name: str, colour: glm.vec3, vector: glm.vec3, power: float):
    #self.type = ltype
    #self.name = name
    colour: ti.math.vec3
    vector: ti.math.vec3
    power: ti.f32


@ti.dataclass
class Intersection:

    #def __init__(self, time: float, normal: glm.vec3, position: glm.vec3, material: Material):
    time: ti.f32
    normal: ti.math.vec3
    position: ti.math.vec3
    mat: Material

def defaultIntersection():
    time = float("inf")
    normal = glm.vec3(0, 0, 0)
    position = glm.vec3(0, 0, 0)
    mat = defaultMaterial()
    return Intersection(time, normal, position, mat)
