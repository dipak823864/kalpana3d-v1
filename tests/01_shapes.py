import sys
import os
import numpy as np
from numba import njit

# Ensure we can import kalpana3d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d.math_core import vec3
from kalpana3d.sdf import sdCapsule, sdTorus, opUnion
from kalpana3d.render import render_image

@njit(fastmath=True)
def scene_sdf(p):
    # Capsule
    # Point a: (-1.5, -0.5, 0), b: (-1.5, 0.5, 0), r: 0.5
    capsule = sdCapsule(p, vec3(-1.5, -0.5, 0.0), vec3(-1.5, 0.5, 0.0), 0.5)
    
    # Torus
    # Center at (1.5, 0, 0)
    # We need to translate p to move the shape
    p_torus = p - vec3(1.5, 0.0, 0.0)
    torus = sdTorus(p_torus, 0.8, 0.2)
    
    return opUnion(capsule, torus)

def main():
    width = 640
    height = 480
    ro = vec3(0.0, 0.0, 4.0) # Camera position
    lookat = vec3(0.0, 0.0, 0.0)
    fov = 60.0
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gallery/images/01_shapes.png'))
    
    print("Rendering Phase 1: Shapes...")
    render_image(width, height, ro, lookat, fov, scene_sdf, output_path)

if __name__ == "__main__":
    main()
