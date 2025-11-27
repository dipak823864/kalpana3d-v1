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
    v1 = np.float32(val1)
    v2 = np.float32(val2)
    eps = np.float32(1e-5)

    if abs(iso - v1) < eps:
        return p1
    if abs(iso - v2) < eps:
        return p2

    diff = v2 - v1
    if abs(diff) < eps:
        return p1

    mu = (iso - v1) / diff
    return mix(p1, p2, mu)

@njit(fastmath=True)
def compute_mesh_counts(min_bound, max_bound, resolution, sdf_func, iso_level):
    step = (max_bound - min_bound) / resolution
    res_x = int(resolution[0])
    res_y = int(resolution[1])
    res_z = int(resolution[2])
    
    count = 0
    
    # Standard Paul Bourke Offsets (XZ-plane first)
    # 0:(0,0,0), 1:(1,0,0), 2:(1,0,1), 3:(0,0,1)
    # 4:(0,1,0), 5:(1,1,0), 6:(1,1,1), 7:(0,1,1)
    offsets = np.array([
        [0,0,0], [1,0,0], [1,0,1], [0,0,1],
        [0,1,0], [1,1,0], [1,1,1], [0,1,1]
    ], dtype=np.float32)

    for x in range(res_x):
        for y in range(res_y):
            for z in range(res_z):
                pos = min_bound + vec3(x, y, z) * step
                
                cube_index = 0
                for i in range(8):
                    p_corner = pos + offsets[i] * step
                    val = sdf_func(p_corner)
                    if val < iso_level:
                        cube_index |= (1 << i)
                
                if edge_table[cube_index] == 0:
                    continue
                
                for i in range(0, 16, 3):
                    if tri_table[cube_index, i] == -1:
                        break
                    count += 1
                    
    return count

@njit(parallel=True) # fastmath removed as requested for safety
def generate_mesh(min_bound, max_bound, resolution, sdf_func, iso_level, max_triangles):
    vertices = np.empty((max_triangles * 3, 3), dtype=np.float32)
    
    step = (max_bound - min_bound) / resolution
    res_x = int(resolution[0])
    res_y = int(resolution[1])
    res_z = int(resolution[2])
    
    tri_idx = 0
    
    # Standard Paul Bourke Offsets (XZ-plane first)
    offsets = np.array([
        [0,0,0], [1,0,0], [1,0,1], [0,0,1],
        [0,1,0], [1,1,0], [1,1,1], [0,1,1]
    ], dtype=np.float32)

    for x in range(res_x):
        for y in range(res_y):
            for z in range(res_z):
                if tri_idx >= max_triangles:
                    break
                
                pos = min_bound + vec3(x, y, z) * step
                
                p = np.empty((8, 3), dtype=np.float32)
                val = np.empty(8, dtype=np.float32)
                
                cube_index = 0
                for i in range(8):
                    p[i] = pos + offsets[i] * step
                    val[i] = sdf_func(p[i])
                    if val[i] < iso_level:
                        cube_index |= (1 << i)
                
                if edge_table[cube_index] == 0:
                    continue
                
                vert_list = np.empty((12, 3), dtype=np.float32)
                
                # Edge connections based on the Paul Bourke standard topology
                # This mapping assumes the edge numbering:
                # 0: v0-v1, 1: v1-v2, 2: v2-v3, 3: v3-v0
                # 4: v4-v5, 5: v5-v6, 6: v6-v7, 7: v7-v4
                # 8: v0-v4, 9: v1-v5, 10: v2-v6, 11: v3-v7

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
                
                for i in range(0, 16, 3):
                    if tri_table[cube_index, i] == -1:
                        break
                    
                    if tri_idx >= max_triangles:
                        break
                        
                    v1 = vert_list[tri_table[cube_index, i]]
                    v2 = vert_list[tri_table[cube_index, i+1]]
                    v3 = vert_list[tri_table[cube_index, i+2]]
                    
                    vertices[tri_idx*3 + 0] = v1
                    vertices[tri_idx*3 + 1] = v2
                    vertices[tri_idx*3 + 2] = v3
                    
                    tri_idx += 1
                    
    return vertices[:tri_idx*3]
