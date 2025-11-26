import sys
import os
import numpy as np
from numba import njit
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d.math_core import vec3
from kalpana3d.sdf import sdRoundCone, opUnion, opSmoothUnion, opTwist
from kalpana3d.noise import fbm
from kalpana3d.render import render_image
from kalpana3d.mesher import generate_mesh, compute_mesh_counts
from kalpana3d.export import export_obj
from kalpana3d.parser import load_scene

def make_tree_sdf(rc_a, rc_b, rc_r1, rc_r2, count):
    # We capture the arrays. Numba should handle this if we call njit inside or return a jitted function.
    # But calling njit inside a function is slow (compiles every time).
    # Better to define a generic implementation and a wrapper.
    
    @njit(fastmath=True)
    def tree_sdf(p):
        # Twist
        # Twist the whole tree around Y axis
        # Amount depends on height (y)
        # Twist(p, 1.0)
        p_twisted = opTwist(p, 0.5) # Gentle twist
        
        d = 1000.0
        
        # Smooth union of branches
        # We can loop and blend.
        # But SmoothUnion is order dependent.
        # For a tree, simple union is often enough, or smooth union with k=0.1
        
        for i in range(count):
            # Apply noise to radius or position?
            # Bark texture: fbm(p)
            
            # Distance to this branch
            d_branch = sdRoundCone(p_twisted, rc_a[i], rc_b[i], rc_r1[i], rc_r2[i])
            
            if i == 0:
                d = d_branch
            else:
                d = opSmoothUnion(d, d_branch, 0.2)
                
        # Bark detail
        # Apply noise to the final distance field
        # Use p_twisted for noise consistency
        n = fbm(p_twisted * 4.0, 3)
        d += n * 0.02 # Small displacement
        
        return d
        
    return tree_sdf

def main():
    yaml_path = 'examples/tree.yaml'
    print(f"Loading {yaml_path}...")
    scene = load_scene(yaml_path)
    
    rc = scene['round_cones']
    if rc['count'] == 0:
        print("No round cones found in tree.yaml!")
        return
        
    # Create SDF
    print("Compiling SDF...")
    sdf_func = make_tree_sdf(rc['a'], rc['b'], rc['r1'], rc['r2'], rc['count'])
    
    # 1. Render Full View
    width = 800
    height = 600
    ro = vec3(0.0, 1.5, 4.0)
    lookat = vec3(0.0, 1.5, 0.0)
    fov = 60.0
    
    img_path_full = 'gallery/images/final_tree_full.png'
    print("Rendering Full View...")
    render_image(width, height, ro, lookat, fov, sdf_func, img_path_full)
    
    # 2. Render Detail View
    ro_detail = vec3(0.5, 1.0, 1.0)
    lookat_detail = vec3(0.0, 1.0, 0.0)
    img_path_detail = 'gallery/images/final_tree_detail.png'
    print("Rendering Detail View...")
    render_image(width, height, ro_detail, lookat_detail, fov, sdf_func, img_path_detail)
    
    # 3. Export Mesh
    print("Generating Mesh...")
    min_bound = vec3(-2.0, -0.5, -2.0)
    max_bound = vec3(2.0, 3.5, 2.0)
    resolution = vec3(128, 128, 128) # High resolution for bark detail
    iso_level = 0.0
    
    start_time = time.time()
    count = compute_mesh_counts(min_bound, max_bound, resolution, sdf_func, iso_level)
    print(f"Counted {count} triangles.")
    
    if count > 0:
        vertices = generate_mesh(min_bound, max_bound, resolution, sdf_func, iso_level, count)
        end_time = time.time()
        print(f"Meshing took {end_time - start_time:.2f} seconds.")
        
        obj_path = 'gallery/models/final_tree.obj'
        export_obj(vertices, obj_path)
    else:
        print("No triangles to export.")

if __name__ == "__main__":
    main()
