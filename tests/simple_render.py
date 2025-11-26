import numpy as np
from numba import njit
from kalpana3d.sdf import sdCapsule
from kalpana3d.render import render_image
from kalpana3d.math_core import vec3

@njit(fastmath=True)
def sdf_func(p):
    a = vec3(0.0, 0.0, 0.0)
    b = vec3(0.0, 1.0, 0.0)
    r = np.float32(0.5)
    return sdCapsule(p, a, b, r)

if __name__ == '__main__':
    ro = vec3(2.0, 2.0, 2.0)
    lookat = vec3(0.0, 0.5, 0.0)
    render_image(512, 512, ro, lookat, 45.0, sdf_func, 'gallery/images/simple_capsule.png')
