import numpy as np

def save_obj(vertices, filename):
    """
    Exports a list of vertices to an OBJ file.
    vertices: np.array of shape (N, 3)
    """
    with open(filename, 'w') as f:
        f.write("# Kalpana3D OBJ Export\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # Write faces
        # Since we are exporting raw triangles (unindexed), every 3 vertices form a face
        num_triangles = len(vertices) // 3
        for i in range(num_triangles):
            # OBJ indices are 1-based
            idx = i * 3 + 1
            f.write(f"f {idx} {idx+1} {idx+2}\n")
            
    print(f"Exported {filename} ({num_triangles} triangles)")
