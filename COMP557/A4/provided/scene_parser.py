import copy
import json
import provided.helperclasses as hc
import provided.geometry as geom
import provided.scene as scene
import glm
from typing import List

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry


def populateVec(array: list):
    return glm.vec3(array[0], array[1], array[2])


def load_scene(infile):
    print("Parsing file:", infile)
    f = open(infile)
    data = json.load(f)

    # Loading camera
    cam_pos = populateVec(data["camera"]["position"])
    cam_lookat = populateVec(data["camera"]["lookAt"])
    cam_up = populateVec(data["camera"]["up"])
    cam_fov = data["camera"]["fov"]

    # Loading resolution
    try:
        width = data["resolution"][0]
        height = data["resolution"][1]
    except KeyError:
        print("No resolution found, defaulting to 1080x720.")
        width = 1080
        height = 720

    # Loading ambient light
    try:
        ambient = populateVec(data["ambient"])
    except KeyError:
        print("No ambient light defined, defaulting to [0, 0, 0]")
        ambient = populateVec([0, 0, 0])

    # Loading Anti-Aliasing options
    try:
        jitter = data["AA"]["jitter"]
        samples = data["AA"]["samples"]
    except KeyError:
        print("No Anti-Aliasing options found, setting to defualt")
        jitter = False
        samples = 1

    # Loading scene lights
    lights = []
    try:
        for light in data["lights"]:


            l_type = light["type"]
            l_name = light["name"]
            l_colour = populateVec(light["colour"])
            
            if l_type == "point":
                l_vector = populateVec(light["position"])
                l_power = light["power"]

            elif l_type == "directional":
                l_vector = populateVec(light["direction"])
                l_power = 1.0
            elif l_type == "area":
                areaLightToPoint(light,lights)
                continue
            else:
                print("Unkown light type", l_type, ", skipping initialization")
                continue
            lights.append(hc.Light(l_type, l_name, l_colour, l_vector, l_power))
    except KeyError:
        lights = []

    # Loading materials
    materials = []
    for material in data["materials"]:
        mat_diffuse = populateVec(material["diffuse"])
        mat_specular = populateVec(material["specular"])
        mat_hardness = material["hardness"]
        mat_name = material["name"]
        mat_id = material["ID"]
        materials.append(hc.Material(mat_name, mat_specular, mat_diffuse, mat_hardness, mat_id))

    # Loading geometry
    objects = []

    # Extra stuff for hierarchies
    rootNames = []
    roots = []
    for geometry in data["objects"]:
        # Elements common to all objects: name, type, position, material(s)
        g_name = geometry["name"]
        g_type = geometry["type"]
        g_pos = populateVec(geometry["position"])
        g_mats = associate_material(materials, geometry["materials"])

        if add_basic_shape(g_name, g_type, g_pos, g_mats, geometry, objects):
            # Non-hierarchies are straightforward
            continue
        elif g_type == "node":
            g_ref = geometry["ref"]
            g_r = populateVec(geometry["rotation"])
            g_s = populateVec(geometry["scale"])

            if g_ref == "":
                # Brand-new hierarchy
                rootNames.append(g_name)
                node = geom.Hierarchy(g_name, g_type, g_mats, g_pos, g_r, g_s)
                traverse_children(node, geometry["children"], materials)
                roots.append(node)
                objects.append(node)
            else:
                # Hierarchy that depends on a previously defined one
                rid = -1
                for i in range(len(rootNames)):
                    # Find hierarchy that this references
                    if g_ref == rootNames[i]:
                        rid = i
                        break
                if rid != -1:
                    node = copy.deepcopy(roots[rid])
                    node.name = g_name
                    node.materials = g_mats
                    node.make_matrices(g_pos, g_r, g_s)
                    objects.append(node)
                else:
                    print("Node reference", g_ref, "not found, skipping creation")

        else:
            print("Unkown object type", g_type, ", skipping initialization")
            continue

    print("Parsing complete")
    return scene.Scene(width, height, jitter, samples,  # General settings
                       cam_pos, cam_lookat, cam_up, cam_fov,  # Camera settings
                       ambient, lights,  # Light settings
                       materials, objects)  # General settings


def add_basic_shape(g_name: str, g_type: str, g_pos: glm.vec3, g_mats: List[hc.Material], geometry, objects: List[geom.Geometry]):
    # Function for adding non-hierarchies to a list, since there's nothing extra to do with them
    # Returns True if a shape was added, False otherwise
    if g_type == "sphere":
        g_radius = geometry["radius"]
        objects.append(geom.Sphere(g_name, g_type, g_mats, g_pos, g_radius))
    elif g_type == "plane":
        g_normal = populateVec(geometry["normal"])
        objects.append(geom.Plane(g_name, g_type, g_mats, g_pos, g_normal))
    elif g_type == "box":
        try:
            g_size = populateVec(geometry["size"])
            objects.append(geom.AABB(g_name, g_type, g_mats, g_pos, g_size))
        except KeyError:
            # Boxes can also be directly declared with a min and max position
            box = geom.AABB(g_name, g_type, g_mats, g_pos, glm.vec3(0, 0, 0))
            box.minpos = populateVec(geometry["min"])
            box.maxpos = populateVec(geometry["max"])
            objects.append(box)
    elif g_type == "mesh":
        g_path = geometry["filepath"]
        g_scale = geometry["scale"]
        objects.append(geom.Mesh(g_name, g_type, g_mats, g_pos, g_scale, g_path))
    elif g_type == "ellipsoid":
        g_radius = geometry["radii"]
        objects.append(geom.Ellipsoid(g_name, g_type, g_mats, g_pos, g_radius))
    else:
        return False
    return True


def traverse_children(node: geom.Hierarchy, children, materials: List[hc.Material]):
    for geometry in children:
        # Obtain info common to all shapes like in the main body of the parser
        g_name = geometry["name"]
        g_type = geometry["type"]
        try:
            g_pos = populateVec(geometry["position"])
        except KeyError:
            g_pos = glm.vec3(0, 0, 0)
        g_mats = associate_material(materials, geometry["materials"])

        if add_basic_shape(g_name, g_type, g_pos, g_mats, geometry, node.children):
            # Nothing fancy to do for non-hierarchies
            continue
        elif g_type == "node":
            # Hierarchy within a hierarchy, recurse
            g_r = populateVec(geometry["rotation"])
            g_s = populateVec(geometry["scale"])
            inner = geom.Hierarchy(g_name, g_type, g_mats, g_pos, g_r, g_s)
            node.children.append(inner)
            traverse_children(inner, geometry["children"], materials)
        else:
            print("Unkown child object type", g_type, ", skipping initialization")


def associate_material(mats: List[hc.Material], ids: List[int]):
    new_list = []
    for i in ids:
        for mat in mats:
            if i == mat.ID:
                new_list.append(mat)
    return new_list


def areaLightToPoint(light, lights):
    step_width = step_height = light['samples']

    sampling_frequency = int(light["shape"][0]/step_height)

    # Iterate over the width and height of the area light
    for i in range(sampling_frequency):
        for j in range(sampling_frequency):
            # Calculate the position of the current point light
            x = light["position"][0] - light["shape"][0] / 2 + i * step_width
            y = light["position"][1]
            z = light["position"][2] - light["shape"][1] / 2 + j * step_height

            lights.append(hc.Light(light["type"], light["name"], 
                                         populateVec(light["colour"]), populateVec([x,y,z]), 
                                         light["power"]))