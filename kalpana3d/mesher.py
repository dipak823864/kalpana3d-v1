import numpy as np
from numba import njit, prange
from kalpana3d.math_core import vec3, normalize, cross, dot, mix
from kalpana3d.marching_cubes_tables import edge_table, tri_table

@njit(fastmath=True)
def get_grid_value(p, sdf_func):
    return sdf_func(p)

@njit(fastmath=True)
def vertex_interp(iso_level, p1, p2, val1, val2):
    iso = np.float32(iso_level)
    eps = np.float32(0.00001)
    if abs(iso - val1) < eps:
        return p1
    if abs(iso - val2) < eps:
        return p2
    if abs(val1 - val2) < eps:
        return p1
    mu = (iso - val1) / (val2 - val1)
    return mix(p1, p2, mu)

@njit(fastmath=True)
def compute_mesh_counts(min_bound, max_bound, resolution, sdf_func, iso_level):
    # Pass 1: Count vertices
    # resolution is (res_x, res_y, res_z)
    
    step = (max_bound - min_bound) / resolution
    
    count = 0
    
    # We iterate through the grid
    # For Numba parallel, we can try to parallelize the outer loop
    # But reduction (count) is tricky in parallel without race conditions or atomics (which are slow/limited)
    # So we might do this serially or use an array to sum up.
    # Let's do serial for safety first, or use a reduction array.
    
    res_x = int(resolution[0])
    res_y = int(resolution[1])
    res_z = int(resolution[2])
    
    # Pre-allocate grid values? No, too much memory if large.
    # Evaluate on the fly.
    
    # Optimization: We can't evaluate 8 corners for every cell, that's redundant.
    # But for simplicity in Phase 3, let's evaluate per cell.
    # To optimize, we would cache layers.
    
    # Let's just do it per cell for now.
    
    for x in range(res_x):
        for y in range(res_y):
            for z in range(res_z):
                # Calculate position of corner 0
                pos = min_bound + vec3(x, y, z) * step
                
                # We need values at 8 corners
                # 0: x, y, z
                # 1: x+1, y, z
                # ...
                # This is very expensive to re-evaluate.
                # But let's stick to the plan.
                
                # Actually, let's implement a simple caching strategy?
                # No, let's trust the CPU speed for now.
                
                # Determine cube index
                cube_index = 0
                
                # Corner positions relative to cell
                # 0: 0,0,0
                # 1: 1,0,0
                # 2: 1,0,1
                # 3: 0,0,1
                # 4: 0,1,0
                # 5: 1,1,0
                # 6: 1,1,1
                # 7: 0,1,1
                
                # Wait, standard MC indexing:
                # 0: 0,0,0
                # 1: 1,0,0
                # 2: 1,1,0
                # 3: 0,1,0
                # 4: 0,0,1
                # 5: 1,0,1
                # 6: 1,1,1
                # 7: 0,1,1
                # Check Paul Bourke's indexing.
                # 0: x, y, z
                # 1: x+1, y, z
                # 2: x+1, y, z+1  <-- Wait.
                # Bourke:
                # 0: 0,0,0
                # 1: 1,0,0
                # 2: 1,0,1
                # 3: 0,0,1
                # 4: 0,1,0
                # 5: 1,1,0
                # 6: 1,1,1
                # 7: 0,1,1
                
                # My edge table assumes Bourke's.
                
                # Let's evaluate.
                vals = np.empty(8, dtype=np.float32)
                # Bourke's vertex ordering
                # 0: 0,0,0
                # 1: 1,0,0
                # 2: 1,1,0
                # 3: 0,1,0
                # 4: 0,0,1
                # 5: 1,0,1
                # 6: 1,1,1
                # 7: 0,1,1
                vals[0] = sdf_func(pos + vec3(0,0,0)*step)
                vals[1] = sdf_func(pos + vec3(1,0,0)*step)
                vals[2] = sdf_func(pos + vec3(1,1,0)*step)
                vals[3] = sdf_func(pos + vec3(0,1,0)*step)
                vals[4] = sdf_func(pos + vec3(0,0,1)*step)
                vals[5] = sdf_func(pos + vec3(1,0,1)*step)
                vals[6] = sdf_func(pos + vec3(1,1,1)*step)
                vals[7] = sdf_func(pos + vec3(0,1,1)*step)
                
                if vals[0] < iso_level: cube_index |= 1
                if vals[1] < iso_level: cube_index |= 2
                if vals[2] < iso_level: cube_index |= 4
                if vals[3] < iso_level: cube_index |= 8
                if vals[4] < iso_level: cube_index |= 16
                if vals[5] < iso_level: cube_index |= 32
                if vals[6] < iso_level: cube_index |= 64
                if vals[7] < iso_level: cube_index |= 128
                
                # Look up edges
                edges = edge_table[cube_index]
                if edges == 0:
                    continue
                
                # Count triangles
                # Tri table has -1 terminator
                for i in range(0, 16, 3):
                    if tri_table[cube_index, i] == -1:
                        break
                    count += 1
                    
    return count

@njit(fastmath=True)
def generate_mesh(min_bound, max_bound, resolution, sdf_func, iso_level, max_triangles):
    # Pass 2: Generate geometry
    
    # Output arrays
    # Vertices: (max_triangles * 3, 3)
    # We will output unindexed triangles (flat shading ready)
    # Or we can try to weld them later. For now, just raw triangles.
    
    vertices = np.empty((max_triangles * 3, 3), dtype=np.float32)
    normals = np.empty((max_triangles * 3, 3), dtype=np.float32) # Optional
    
    step = (max_bound - min_bound) / resolution
    res_x = int(resolution[0])
    res_y = int(resolution[1])
    res_z = int(resolution[2])
    
    tri_idx = 0
    
    for x in range(res_x):
        for y in range(res_y):
            for z in range(res_z):
                if tri_idx >= max_triangles:
                    break
                
                pos = min_bound + vec3(x, y, z) * step
                
                # Evaluate 8 corners
                # Optimization: In a real engine, we would cache these.
                p = np.empty((8, 3), dtype=np.float32)
                val = np.empty(8, dtype=np.float32)
                
                # Offsets for Bourke's ordering
                offsets = np.array([
                    [0,0,0], [1,0,0], [1,1,0], [0,1,0],
                    [0,0,1], [1,0,1], [1,1,1], [0,1,1]
                ], dtype=np.float32)
                
                cube_index = 0
                for i in range(8):
                    p[i] = pos + offsets[i] * step
                    val[i] = sdf_func(p[i])
                    if val[i] < iso_level:
                        cube_index |= (1 << i)
                
                if edge_table[cube_index] == 0:
                    continue
                
                # Compute vertices on edges
                vert_list = np.empty((12, 3), dtype=np.float32)
                
                if edge_table[cube_index] & 1:
                    vert_list[0] = vertex_interp(iso_level, p[0], p[1], val[0], val[1])
                if edge_table[cube_index] & 2:
                    vert_list[1] = vertex_interp(iso_level, p[1], p[2], val[1], val[2])
                if edge_table[cube_index] & 4:
                    vert_list[2] = vertex_interp(iso_level, p[2], p[3], val[2], val[3])
                if edge_table[cube_index] & 8:
                    vert_list[3] = vertex_interp(iso_level, p[3], p[0], val[3], val[0])
                if edge_table[cube_index] & 16:
                    vert_list[4] = vertex_interp(iso_level, p[4], p[5], val[4], val[5])
                if edge_table[cube_index] & 32:
                    vert_list[5] = vertex_interp(iso_level, p[5], p[6], val[5], val[6])
                if edge_table[cube_index] & 64:
                    vert_list[6] = vertex_interp(iso_level, p[6], p[7], val[6], val[7])
                if edge_table[cube_index] & 128:
                    vert_list[7] = vertex_interp(iso_level, p[7], p[4], val[7], val[4])
                if edge_table[cube_index] & 256:
                    vert_list[8] = vertex_interp(iso_level, p[0], p[4], val[0], val[4])
                if edge_table[cube_index] & 512:
                    vert_list[9] = vertex_interp(iso_level, p[1], p[5], val[1], val[5])
                if edge_table[cube_index] & 1024:
                    vert_list[10] = vertex_interp(iso_level, p[2], p[6], val[2], val[6])
                if edge_table[cube_index] & 2048:
                    vert_list[11] = vertex_interp(iso_level, p[3], p[7], val[3], val[7])
                
                # Create triangles
                for i in range(0, 16, 3):
                    if tri_table[cube_index, i] == -1:
                        break
                    
                    if tri_idx >= max_triangles:
                        break
                        
                    # Get vertices
                    v1 = vert_list[tri_table[cube_index, i]]
                    v2 = vert_list[tri_table[cube_index, i+1]]
                    v3 = vert_list[tri_table[cube_index, i+2]]
                    
                    # Store
                    vertices[tri_idx*3 + 0] = v1
                    vertices[tri_idx*3 + 1] = v2
                    vertices[tri_idx*3 + 2] = v3
                    
                    tri_idx += 1
                    
    return vertices[:tri_idx*3]
