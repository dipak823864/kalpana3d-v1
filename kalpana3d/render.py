import numpy as np
from numba import njit, prange
from PIL import Image
from kalpana3d.math_core import vec3, normalize, cross, dot

@njit(fastmath=True)
def get_camera_ray(uv, ro, lookat, fov):
    f = normalize(lookat - ro)
    world_up = vec3(0.0, 1.0, 0.0)
    r = normalize(cross(world_up, f))
    u = cross(f, r)
    
    zoom = np.float32(1.0) / np.tan(np.radians(np.float32(fov)) / np.float32(2.0))
    
    c = ro + f * zoom
    i = c + uv[0] * r + uv[1] * u
    rd = normalize(i - ro)
    return rd

@njit(fastmath=True)
def calc_normal(p, sdf_func):
    eps = np.float32(0.0001)
    x = np.float32(sdf_func(p + vec3(eps, 0.0, 0.0))) - np.float32(sdf_func(p - vec3(eps, 0.0, 0.0)))
    y = np.float32(sdf_func(p + vec3(0.0, eps, 0.0))) - np.float32(sdf_func(p - vec3(0.0, eps, 0.0)))
    z = np.float32(sdf_func(p + vec3(0.0, 0.0, eps))) - np.float32(sdf_func(p - vec3(0.0, 0.0, eps)))
    return normalize(vec3(x, y, z))

@njit(fastmath=True)
def ray_march(ro, rd, sdf_func):
    dO = np.float32(0.0)
    for i in range(256):
        p = ro + rd * dO
        dS = np.float32(sdf_func(p))
        if dS < 0.001:
            return dO
        if dO > 100.0:
            break
        dO += dS
    return np.float32(100.0)

@njit(fastmath=True)
def calc_soft_shadow(ro, rd, sdf_func, k):
    res = np.float32(1.0)
    t = np.float32(0.01)
    for i in range(64):
        h = np.float32(sdf_func(ro + rd * t))
        if h < 0.001:
            return np.float32(0.0)
        res = min(res, k * h / t)
        t += h
        if t > 50.0:
            break
    return res

@njit(fastmath=True)
def calc_ao(p, n, sdf_func):
    occ = np.float32(0.0)
    w = np.float32(1.0)
    for i in range(1, 6):
        d = np.float32(i * 0.1)
        occ += (d - np.float32(sdf_func(p + n * d))) * w
        w *= np.float32(0.5)
    val = occ
    return np.float32(1.0) - max(np.float32(0.0), min(np.float32(1.0), val))

@njit(fastmath=True, parallel=True)
def render_kernel(width, height, ro, lookat, fov, sdf_func, output_buffer):
    for y in prange(height):
        for x in range(width):
            uv_y = -((np.float32(y) / np.float32(height)) * np.float32(2.0) - np.float32(1.0))
            uv_x = ((np.float32(x) / np.float32(width)) * np.float32(2.0) - np.float32(1.0)) * (np.float32(width) / np.float32(height))
            
            uv = np.array([uv_x, uv_y], dtype=np.float32)
            
            rd = get_camera_ray(uv, ro, lookat, fov)
            
            d = ray_march(ro, rd, sdf_func)
            
            col = vec3(0.1, 0.1, 0.15)
            
            if d < 100.0:
                p = ro + rd * d
                n = calc_normal(p, sdf_func)
                
                light_pos = vec3(2.0, 4.0, 3.0)
                l = normalize(light_pos - p)
                
                shadow = calc_soft_shadow(p + n * np.float32(0.001), l, sdf_func, np.float32(16.0))

                diff = max(dot(n, l), np.float32(0.0))
                ambient = np.float32(0.1)
                
                ao = calc_ao(p, n, sdf_func)

                mat_col = vec3(1.0, 1.0, 1.0)
                
                final_col = mat_col * (diff * shadow + ambient * ao)
                col = final_col
            
            f0 = np.float32(0.0)
            f1 = np.float32(1.0)
            col[0] = min(max(col[0], f0), f1)
            col[1] = min(max(col[1], f0), f1)
            col[2] = min(max(col[2], f0), f1)
            
            output_buffer[y, x, 0] = col[0]
            output_buffer[y, x, 1] = col[1]
            output_buffer[y, x, 2] = col[2]

def render_image(width, height, ro, lookat, fov, sdf_func, filename):
    output_buffer = np.zeros((height, width, 3), dtype=np.float32)
    render_kernel(width, height, ro, lookat, fov, sdf_func, output_buffer)
    img_data = (output_buffer * 255).astype(np.uint8)
    img = Image.fromarray(img_data)
    img.save(filename)
    print(f"Saved {filename}")
