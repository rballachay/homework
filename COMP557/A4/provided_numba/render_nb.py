import numba as nb
import numpy as np
import math
import provided_numba.helperclasses as hc
import random

shadow_epsilon = 10**(-6)

@nb.jit(nopython=True)
def zip(arr1, arr2):
    result = []
    for i in range(min(len(arr1), len(arr2))):
        result.append((arr1[i], arr2[i]))
    return result

@nb.jit(nopython=True)
def rotate_vector_safe(vector):
    vector=vector.astype(np.float64)
    
    limit = 0.1

    # Define the rotation angles in radians (small amounts)
    angle_x = np.random.uniform(low=-limit,high=limit)
    angle_y = np.random.uniform(low=-limit,high=limit)
    angle_z = np.random.uniform(low=-limit,high=limit)

    # Define the rotation matrices
    rotation_matrix_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle_x), -np.sin(angle_x)],
        [0.0, np.sin(angle_x), np.cos(angle_x)]
    ])

    rotation_matrix_y = np.array([
        [np.cos(angle_y), 0.0, np.sin(angle_y)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle_y), 0.0, np.cos(angle_y)]
    ])

    rotation_matrix_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0.0],
        [np.sin(angle_z), np.cos(angle_z), 0.0],
        [0.0, 0.0, 1.0]
    ])

    rotated_vector = np.dot(rotation_matrix_x, np.dot(rotation_matrix_y, np.dot(rotation_matrix_z, vector)))
    return rotated_vector.astype(np.float32)


@nb.jit(nopython=True)
def render_nb(width, height, position, lookat, aspect, fov, up, ambient, objects, lights, samples, jitter, depth):
    """
    For us to get the effect of depth of field, we are going to move our camer around a bit and then
    average those images together. This will cause the further objects to move a greater amount, resulting
    in blurred objects 
    """
    num_images = 4 if depth else 1
    images_aperture = np.zeros((width,height,3,num_images))

    for a in range(num_images):
        image = np.zeros((width, height, 3))

        cam_dir = position  - lookat
        d = 1.0
        top = d * math.tan(0.5 * math.pi * fov / 180)
        right = aspect * top
        bottom = -top
        left = -right

        if depth:
            w = normalized(rotate_vector_safe(cam_dir))
        else:
            w = normalized(cam_dir)

        u = np.cross(up, w)
        u = normalized(u)
        v = np.cross(w, u)

        for i in range(width):
            for j in range(height):
                colour = np.array([0, 0, 0]).astype(np.float32)

                for si in range(samples):
                    for sj in range(samples):
                        if jitter:
                            noise_x = random.uniform(0,1/10)
                            noise_y = random.uniform(0,1/10)
                        else:
                            noise_x = noise_y = 0

                        u_pix = left + (right - left) * (i + (si + 0.5 + noise_x) / samples) / width
                        v_pix = bottom + (top - bottom) * (j + (sj + 0.5 + noise_y) / samples) / height

                        # we move the ray to the left and right on our "aperature" 
                        s = position+u_pix*u + v_pix*v - d*w

                        ray = hc.Ray(position, s-position)

                        # TODO: Test for intersection
                        intersection = hc.defaultIntersection()
                        for object, transform in zip(objects.planes,objects.planeTransforms):
                            object.intersect(ray,intersection,transform)
                        for object, transform in zip(objects.spheres,objects.sphereTransforms):
                            object.intersect(ray,intersection,transform)
                        for object, transform in zip(objects.meshes,objects.meshTransforms):
                            object.intersect(ray,intersection,transform)
                        for object, transform in zip(objects.boxes,objects.boxTransforms):
                            object.intersect(ray,intersection,transform)
                        for object, transform in zip(objects.ellipsoids,objects.ellipsoidTransforms):
                            object.intersect(ray,intersection,transform)

                        if np.sum(intersection.position)==0:
                            continue

                        # ambient has no relation to the light
                        ambient_light = intersection.mat.diffuse * ambient 
                        diffuse_factor = np.array([0, 0, 0], dtype=np.float32)
                        blinn_phong = np.array([0, 0, 0], dtype=np.float32)
                        mirror_reflection = np.array([0, 0, 0], dtype=np.float32)

                        for light in lights:
                            light_vector =  normalized(light.vector - intersection.position)

                            # TODO: Cast shadow ray starting from position that is just a little 
                            # bit backwards on the light vector
                            shadow_ray = hc.Ray((intersection.position+shadow_epsilon).astype(np.float32),
                                                light_vector.astype(np.float64))

                            in_shadow = False
                            for object, transform in zip(objects.planes,objects.planeTransforms):
                                shadow_intersection = hc.defaultIntersection()
                                object.intersect(shadow_ray, shadow_intersection, transform)
                                if np.sum(shadow_intersection.position) != 0 and shadow_intersection.time>0:
                                    in_shadow = True
                                    break
                            
                            if not in_shadow:
                                for object, transform in zip(objects.spheres,objects.sphereTransforms):
                                    shadow_intersection = hc.defaultIntersection()
                                    object.intersect(shadow_ray, shadow_intersection, transform)
                                    if np.sum(shadow_intersection.position) != 0 and shadow_intersection.time>0:
                                        in_shadow = True
                                        break
                            
                            if not in_shadow:
                                for object, transform in zip(objects.meshes,objects.meshTransforms):
                                    shadow_intersection = hc.defaultIntersection()
                                    object.intersect(shadow_ray, shadow_intersection, transform)
                                    if np.sum(shadow_intersection.position) != 0 and shadow_intersection.time>0:
                                        in_shadow = True
                                        break

                            if not in_shadow:
                                for object, transform in zip(objects.boxes,objects.boxTransforms):
                                    shadow_intersection = hc.defaultIntersection()
                                    object.intersect(shadow_ray, shadow_intersection, transform)
                                    if np.sum(shadow_intersection.position) != 0 and shadow_intersection.time>0:
                                        in_shadow = True
                                        break
                            
                            if not in_shadow:
                                for object, transform in zip(objects.ellipsoids,objects.ellipsoidTransforms):
                                    shadow_intersection = hc.defaultIntersection()
                                    object.intersect(shadow_ray, shadow_intersection, transform)
                                    if np.sum(shadow_intersection.position) != 0 and shadow_intersection.time>0:
                                        in_shadow = True
                                        break

                            if not in_shadow:
                                _dot = np.dot(light_vector.astype(np.float32), intersection.normal.astype(np.float32))
                                _max = np.float32(np.max(np.array([0.0, np.float32(_dot)])))
                                diffuse_factor = diffuse_factor + intersection.mat.diffuse * light.colour  * _max  * np.float32(light.power)
                                blinn_phong += blinn_phong_specular_shading(light, intersection, ray.direction, light_vector).astype(np.float32)

                                if light.reflectivity:  
                                    reflection_vector = normalized(ray.direction.astype(np.float32) - 2.0 * np.dot(ray.direction.astype(np.float32), intersection.normal.astype(np.float32)) * intersection.normal)
                                    mirror_intensity = np.dot(reflection_vector.astype(np.float32), light_vector.astype(np.float32)) ** np.float32(light.reflectivity)
                                    mirror_reflection += mirror_intensity * intersection.mat.specular * light.colour * light.power
                        
                        colour += ambient_light+diffuse_factor+blinn_phong+mirror_reflection

                colour /= samples ** np.float32(2.0)

                image[i, j, 0] = max(0.0, min(1.0, colour[0]))
                image[i, j, 1] = max(0.0, min(1.0, colour[1]))
                image[i, j, 2] = max(0.0, min(1.0, colour[2]))


        images_aperture[...,a] = image

    return images_aperture

@nb.jit(nopython=True)
def blinn_phong_specular_shading(light, intersection, viewer_direction, light_vector):
    halfway_vector = normalized(light_vector + (-viewer_direction))
    specular_factor = np.max(np.array([0.0, np.dot(intersection.normal.astype(np.float32), halfway_vector.astype(np.float32))],dtype=np.float32)) ** intersection.mat.hardness
    return intersection.mat.specular * light.colour * specular_factor * light.power

@nb.jit(nopython=True)
def normalized(a):
    return a / np.sqrt(np.sum(a**2))