import sys
import os
import numpy as np
from numba import njit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d.math_core import vec3
from kalpana3d.sdf import sdSphere, opSmoothUnion
from kalpana3d.noise import fbm
from kalpana3d.render import render_image

def create_scene_sdf(perm):
    @njit(fastmath=True)
    def scene_sdf(p):
        # Two spheres
        s1 = sdSphere(p - vec3(-0.8, 0.0, 0.0), 1.0)
        s2 = sdSphere(p - vec3(0.8, 0.0, 0.0), 0.8)

        # Smooth blend
        d = opSmoothUnion(s1, s2, 0.5)

        # Add noise
        # Use fbm for texture
        # Ensure p is float32
        n = fbm(p * 2.0, 3, perm)

        # Displacement
        d += n * 0.1

        return d
    return scene_sdf

def main():
    width = 640
    height = 480
    ro = vec3(0.0, 0.0, 4.0)
    lookat = vec3(0.0, 0.0, 0.0)
    fov = 60.0
    
    # Generate permutation table
    perm = np.random.permutation(256).astype(np.int32)
    perm = np.concatenate((perm, perm))

    scene_sdf = create_scene_sdf(perm)

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gallery/images/02_organic.png'))
    
    print("Rendering Phase 2: Organic Shapes...")
    render_image(width, height, ro, lookat, fov, scene_sdf, output_path)

if __name__ == "__main__":
    main()
