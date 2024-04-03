import numba as nb
import numpy as np
import math
import provided_numba.helperclasses as hc

shadow_epsilon = 10**(-6)

@nb.njit
def zip(arr1, arr2):
    result = []
    for i in range(min(len(arr1), len(arr2))):
        result.append((arr1[i], arr2[i]))
    return result
  
@nb.jit(nopython=True)
def render_nb(width, height, position, lookat, aspect, fov, up, ambient, objects, lights):
    image = np.zeros((width, height, 3))

    cam_dir = position - lookat
    d = 1.0
    top = d * math.tan(0.5 * math.pi * fov / 180)
    right = aspect * top
    bottom = -top
    left = -right

    w = normalized(cam_dir)
    u = np.cross(up, w)
    u = normalized(u)
    v = np.cross(w, u)

    for i in range(width):
        for j in range(height):
            colour = np.array([0, 0, 0])

            # TODO: Generate rays
            u_pix = left + (right-left)*(i+0.5)/width
            v_pix = bottom + (top-bottom)*(j+0.5)/height
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

            if np.sum(intersection.position)==0:
                continue

            # ambient has no relation to the light
            ambient_light = intersection.mat.diffuse * ambient 
            diffuse_factor = np.array([0, 0, 0], dtype=np.float32)
            blinn_phong = np.array([0, 0, 0], dtype=np.float32)

            for light in lights:
                light_vector =  normalized(light.vector - intersection.position)

                # TODO: Cast shadow ray
                shadow_ray = hc.Ray((intersection.position+shadow_epsilon).astype(np.float32), light_vector.astype(np.float64))

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
                    _dot = np.dot(light_vector.astype(np.float32), intersection.normal.astype(np.float32))
                    _max = np.float32(np.max(np.array([0.0, np.float32(_dot)])))
                    diffuse_factor = diffuse_factor + intersection.mat.diffuse * light.colour  * _max
                    blinn_phong = blinn_phong.astype(np.float32) + blinn_phong_specular_shading(light, intersection, ray.direction, light_vector).astype(np.float32)
            
            colour = ambient_light+diffuse_factor+blinn_phong

            image[i, j, 0] = max(0.0, min(1.0, colour[0]))
            image[i, j, 1] = max(0.0, min(1.0, colour[1]))
            image[i, j, 2] = max(0.0, min(1.0, colour[2]))

    return image

@nb.jit(nopython=True)
def blinn_phong_specular_shading(light, intersection, viewer_direction, light_vector):
    halfway_vector = normalized(light_vector + (-viewer_direction))
    specular_factor = np.max(np.array([0.0, np.dot(intersection.normal.astype(np.float32), halfway_vector.astype(np.float32))],dtype=np.float32)) ** intersection.mat.hardness
    return intersection.mat.specular * light.colour * specular_factor

@nb.jit(nopython=True)
def normalized(a):
    return a / np.sqrt(np.sum(a**2))