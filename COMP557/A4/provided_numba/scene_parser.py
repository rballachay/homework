import json
import provided_numba.helperclasses as hc
import provided_numba.geometry as geom
import provided_numba.scene as scene
import glm
from typing import List
import numba as nb
import igl
import numpy as np
import cv2

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry


def populateVec(array: list):
    return nb.float32([array[0], array[1], array[2]])


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

    try:
        depth = data["depth"]
    except KeyError:
        depth = False

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
    lights = nb.typed.List()
    try:
        for light in data["lights"]:
            l_type = light["type"]
            l_name = light["name"]
            l_colour = populateVec(light["colour"])
            l_reflect = light.get("reflectivity",0)

            if l_type == "point":
                l_vector = populateVec(light["position"])
                l_power = light["power"]

            elif l_type == "directional":
                l_vector = populateVec(light["direction"])
                l_power = 1.0
            elif l_type == "area":
                # convert area light into a bunch of point lights
                areaLightToPoint(light, lights)
                continue
            else:
                print("Unkown light type", l_type, ", skipping initialization")
                continue
            lights.append(hc.Light(l_type, l_name, l_colour, l_vector, l_power, l_reflect))
    except KeyError:
        lights = nb.typed.List()

    # Loading materials
    materials = nb.typed.List()
    for material in data["materials"]:
        mat_diffuse = populateVec(material["diffuse"])
        mat_specular = populateVec(material["specular"])
        mat_hardness = material["hardness"]
        mat_name = material["name"]
        mat_id = material["ID"]

        # this is in the case we are loading from a material file
        if "file" in material:
            lookup =  cv2.cvtColor(cv2.resize(cv2.imread(material["file"]),(512,512)),cv2.COLOR_BGR2RGB)
            
            # change orange/yellow color to a little bit more gold 
            #mask=cv2.inRange(lookup,lookup[0,0,:]-1,lookup[0,0,:]+1)
            #lookup[mask>0]=np.array([212,175,55])
            lookup = lookup.astype(np.int64)
        else:
            lookup = np.zeros((1,1,1)).astype(np.int64)

        materials.append(hc.Material(mat_name, mat_specular, mat_diffuse, mat_hardness, mat_id, lookup))

    # Loading geometry
    objects = geom.ObjectContainer(spheres=nb.typed.List.empty_list(geom.Sphere.class_type.instance_type),
                                   planes=nb.typed.List.empty_list(geom.Plane.class_type.instance_type),
                                   meshes=nb.typed.List.empty_list(geom.Mesh.class_type.instance_type),
                                   boxes=nb.typed.List.empty_list(geom.AABB.class_type.instance_type),
                                   ellipsoids=nb.typed.List.empty_list(geom.Ellipsoid.class_type.instance_type),
                                   sphereTransforms=nb.typed.List.empty_list(nb.float64[:,:,:]),
                                   planeTransforms=nb.typed.List.empty_list(nb.float64[:,:,:]),
                                   meshTransforms=nb.typed.List.empty_list(nb.float64[:,:,:]),
                                   boxTransforms=nb.typed.List.empty_list(nb.float64[:,:,:]),
                                   ellipsoidTransforms=nb.typed.List.empty_list(nb.float64[:,:,:]))

    # Extra stuff for hierarchies
    for geometry in data["objects"]:
        # Elements common to all objects: name, type, position, material(s)
        g_name = geometry["name"]
        g_type = geometry["type"]
        g_pos = populateVec(geometry["position"])
        g_mats = associate_material(materials, geometry["materials"])

        if add_basic_shape(g_name, g_type, g_pos, g_mats, geometry, objects, np.expand_dims(np.eye(4),axis=2).astype(np.float64)): 
            continue
        elif g_type == "node":
            g_ref = geometry["ref"]
            g_r = populateVec(geometry["rotation"])
            g_s = populateVec(geometry["scale"])

            if not g_ref:
                children=geometry["children"]
            else:
                children = list(filter(lambda x: x['name']==g_ref, data["objects"]))[0]["children"]

            transformation = np.expand_dims(make_mat(g_pos, g_r, g_s),axis=2)

            traverse_children(children, materials, objects, transformation)

        else:
            print("Unkown object type", g_type, ", skipping initialization")
            continue


    return scene.Scene(width, height, jitter, samples,  # General settings
                       cam_pos, cam_lookat, cam_up, cam_fov,  # Camera settings
                       ambient, lights,  # Light settings
                       materials, objects, depth)  # General settings


def add_basic_shape(g_name: str, g_type: str, g_pos: glm.vec3, g_mats: List[hc.Material], geometry, objects: list, transformation):
    # Function for adding non-hierarchies to a list, since there's nothing extra to do with them
    # Returns True if a shape was added, False otherwise
    if g_type == "sphere":
        g_radius = geometry["radius"]
        obj=geom.Sphere(g_name, g_type, g_mats, g_pos, g_radius)
        objects.spheres.append(obj)
        objects.sphereTransforms.append(transformation)
    elif g_type == "plane":
        g_normal = populateVec(geometry["normal"])
        obj=geom.Plane(g_name, g_type, g_mats, g_pos, g_normal)
        objects.planes.append(obj)
        objects.planeTransforms.append(transformation)
    elif g_type == "box":
        try:
            g_size = populateVec(geometry["size"])
            obj = geom.AABB(g_name, g_type, g_mats, g_pos, g_size)
            objects.boxes.append(obj)
            objects.boxTransforms.append(transformation)
        except KeyError:
            # Boxes can also be directly declared with a min and max position
            box = geom.AABB(g_name, g_type, g_mats, g_pos, np.zeros(3))
            box.minpos = populateVec(geometry["min"])
            box.maxpos = populateVec(geometry["max"])
            objects.boxes.append(box)
            objects.boxTransforms.append(transformation)
    elif g_type == "mesh":
        g_path = geometry["filepath"]
        g_scale = geometry["scale"]
        verts, t_coords, norms, faces, t_faces, _ = igl.read_obj(g_path)
        # t_faces is our map into our texture coordinate file
        # each of the faces corresponds to one of the vertices
        obj = geom.Mesh(g_name, g_type, g_mats, g_pos, g_scale, verts, norms, faces, t_coords, t_faces)
        objects.meshes.append(obj)
        objects.meshTransforms.append(transformation)
    elif g_type == "ellipsoid":
        g_radius = populateVec(geometry["radii"])
        obj=geom.Ellipsoid(g_name, g_type, g_mats, g_pos, g_radius)
        objects.ellipsoids.append(obj)
        objects.ellipsoidTransforms.append(transformation)
    else:
        return False
    return True


def traverse_children(children, materials: List[hc.Material], objects, transformation):
    for geometry in children:
        # Obtain info common to all shapes like in the main body of the parser
        g_name = geometry["name"]
        g_type = geometry["type"]
        try:
            g_pos = populateVec(geometry["position"])
        except KeyError:
            g_pos = np.array([0, 0, 0]).astype(np.float32)
            
        g_mats = associate_material(materials, geometry["materials"])

        if add_basic_shape(g_name, g_type, g_pos, g_mats, geometry, objects, transformation.astype(np.float64)):
            continue
        elif g_type == "node":
            # Hierarchy within a hierarchy, recurse
            g_r = populateVec(geometry["rotation"])
            g_s = populateVec(geometry["scale"])

            new_mat = np.expand_dims(make_mat(g_pos, g_r, g_s).astype(np.float32),axis=2)
            transformation = np.append(transformation.astype(np.float32), new_mat,axis=2)

            traverse_children(geometry["children"], materials, objects, transformation)
        else:
            print("Unkown child object type", g_type, ", skipping initialization")


def associate_material(mats: List[hc.Material], ids: List[int]):
    new_list = nb.typed.List.empty_list(hc.Material.class_type.instance_type)
    for i in ids:
        for mat in mats:
            if i == mat.ID:
                new_list.append(mat)
    return new_list


@nb.jit(nopython=True)
def make_mat(t, r, s):
    translation_matrix = np.eye(4, dtype=np.float32)
    translation_matrix[:3, 3] = t

    rotation_matrix = np.eye(4, dtype=np.float32)

    cos_r0 = np.cos(np.radians(r[0]))
    sin_r0 = np.sin(np.radians(r[0]))
    cos_r1 = np.cos(np.radians(r[1]))
    sin_r1 = np.sin(np.radians(r[1]))
    cos_r2 = np.cos(np.radians(r[2]))
    sin_r2 = np.sin(np.radians(r[2]))

    rotation_matrix[0, 0] = cos_r1 * cos_r2
    rotation_matrix[0, 1] = -cos_r1 * sin_r2
    rotation_matrix[0, 2] = sin_r1

    rotation_matrix[1, 0] = sin_r0 * sin_r1 * cos_r2 + cos_r0 * sin_r2
    rotation_matrix[1, 1] = -sin_r0 * sin_r1 * sin_r2 + cos_r0 * cos_r2
    rotation_matrix[1, 2] = -sin_r0 * cos_r1

    rotation_matrix[2, 0] = -cos_r0 * sin_r1 * cos_r2 + sin_r0 * sin_r2
    rotation_matrix[2, 1] = cos_r0 * sin_r1 * sin_r2 + sin_r0 * cos_r2
    rotation_matrix[2, 2] = cos_r0 * cos_r1

    scale_matrix = np.eye(4, dtype=np.float32)
    scale_matrix[:3, :3] = np.diag(s)

    return np.dot(np.dot(translation_matrix, rotation_matrix), scale_matrix)


def areaLightToPoint(light, lights):
    step_width = step_height = light['samples']

    reflectivity = light.get("reflectivity", 0.0)

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
                                         np.float64(light["power"]),
                                         reflectivity))