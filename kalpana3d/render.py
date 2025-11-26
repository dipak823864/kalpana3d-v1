import numpy as np
from numba import njit, prange
from PIL import Image
from kalpana3d.math_core import vec3, normalize, cross, dot

@njit(fastmath=True)
def get_camera_ray(uv, ro, lookat, fov):
    f = normalize(lookat - ro)
    # World up is usually Y
    world_up = vec3(0.0, 1.0, 0.0)
    r = normalize(cross(world_up, f))
    u = cross(f, r)
    
    # Zoom factor
    zoom = 1.0 / np.tan(np.radians(fov) / 2.0)
    
    c = ro + f * zoom
    i = c + uv[0] * r + uv[1] * u
    rd = normalize(i - ro)
    return rd

@njit(fastmath=True)
def calc_normal(p, sdf_func):
    eps = 0.0001
    # Central difference
    x = sdf_func(p + vec3(eps, 0.0, 0.0)) - sdf_func(p - vec3(eps, 0.0, 0.0))
    y = sdf_func(p + vec3(0.0, eps, 0.0)) - sdf_func(p - vec3(0.0, eps, 0.0))
    z = sdf_func(p + vec3(0.0, 0.0, eps)) - sdf_func(p - vec3(0.0, 0.0, eps))
    return normalize(vec3(x, y, z))

@njit(fastmath=True)
def ray_march(ro, rd, sdf_func):
    dO = 0.0
    for i in range(256):
        p = ro + rd * dO
        dS = sdf_func(p)
        if dS < 0.001:
            return dO
        if dO > 100.0:
            break
        dO += dS
    return 100.0

@njit(fastmath=True, parallel=True)
def render_kernel(width, height, ro, lookat, fov, sdf_func, output_buffer):
    # output_buffer is (height, width, 3)
    
    for y in prange(height):
        for x in range(width):
            # Normalized coordinates
            # Map y to [-1, 1]
            # Map x to [-aspect, aspect]
            
            uv_y = -((y / height) * 2.0 - 1.0) # Flip Y so 0 is top
            uv_x = ((x / width) * 2.0 - 1.0) * (width / height)
            
            uv = np.array([uv_x, uv_y], dtype=np.float32)
            
            rd = get_camera_ray(uv, ro, lookat, fov)
            
            d = ray_march(ro, rd, sdf_func)
            
            col = vec3(0.1, 0.1, 0.15) # Background color
            
            if d < 100.0:
                p = ro + rd * d
                n = calc_normal(p, sdf_func)
                
                # Simple lighting
                light_pos = vec3(2.0, 4.0, 3.0)
                l = normalize(light_pos - p)
                
                diff = max(dot(n, l), np.float32(0.0))
                ambient = np.float32(0.1)
                
                # Material color (white for now)
                mat_col = vec3(1.0, 1.0, 1.0)
                
                final_col = mat_col * (diff + ambient)
                col = final_col
            
            # Clamp
            col[0] = min(max(col[0], 0.0), 1.0)
            col[1] = min(max(col[1], 0.0), 1.0)
            col[2] = min(max(col[2], 0.0), 1.0)
            
            output_buffer[y, x, 0] = col[0]
            output_buffer[y, x, 1] = col[1]
            output_buffer[y, x, 2] = col[2]

def render_image(width, height, ro, lookat, fov, sdf_func, filename):
    output_buffer = np.zeros((height, width, 3), dtype=np.float32)
    
    # Numba will compile render_kernel for the specific sdf_func
    render_kernel(width, height, ro, lookat, fov, sdf_func, output_buffer)
    
    # Convert to uint8
    img_data = (output_buffer * 255).astype(np.uint8)
    img = Image.fromarray(img_data)
    img.save(filename)
    print(f"Saved {filename}")
