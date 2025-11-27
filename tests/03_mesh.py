import sys
import os
import numpy as np
from numba import njit
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d.math_core import vec3
from kalpana3d.sdf import sdSphere, opSmoothUnion
from kalpana3d.noise import fbm
from kalpana3d.mesher import generate_mesh, compute_mesh_counts
from kalpana3d.export import save_obj

# We need to pass perm to fbm, but sdf_func must only take p.
# So we use a closure or a factory.
# However, Numba's closure support is limited when passing functions around unless using objmode or typed funcs.
# For simplicity, let's create a factory function that returns the JIT-ed function, similar to final_demo.py.

def create_sdf_func(perm):
    @njit(fastmath=True)
    def scene_sdf(p):
        s1 = sdSphere(p - vec3(-0.8, 0.0, 0.0), 1.0)
        s2 = sdSphere(p - vec3(0.8, 0.0, 0.0), 0.8)
        d = opSmoothUnion(s1, s2, 0.5)
        n = fbm(p * 2.0, 3, perm)
        d += n * 0.1
        # Apply Lipschitz correction
        return d * np.float32(0.6)
    return scene_sdf

def main():
    perm = np.random.permutation(256).astype(np.int32)
    perm = np.concatenate((perm, perm))
    scene_sdf = create_sdf_func(perm)
    # Bounds
    min_bound = vec3(-2.5, -2.0, -2.0)
    max_bound = vec3(2.5, 2.0, 2.0)
    
    # Resolution
    res = 64
    resolution = vec3(res, res, res)
    
    iso_level = 0.0
    
    print("Phase 3: Meshing...")
    start_time = time.time()
    
    # Pass 1: Count
    # We'll skip the count pass and just allocate a large buffer for simplicity/speed if we want,
    # but the mesher has compute_mesh_counts. Let's use it.
    print("Counting triangles...")
    # Note: compute_mesh_counts might be slow on first run due to compilation
    count = compute_mesh_counts(min_bound, max_bound, resolution, scene_sdf, iso_level)
    print(f"Counted {count} triangles.")
    
    if count == 0:
        print("No triangles generated!")
        return

    # Pass 2: Generate
    print("Generating mesh...")
    vertices = generate_mesh(min_bound, max_bound, resolution, scene_sdf, iso_level, count)
    
    end_time = time.time()
    print(f"Meshing took {end_time - start_time:.2f} seconds.")
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gallery/models/organic_test.obj'))
    save_obj(vertices, output_path)

if __name__ == "__main__":
    main()
