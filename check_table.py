import numpy as np
from kalpana3d.marching_cubes_tables import tri_table

print(f"Shape: {tri_table.shape}")
print(f"Max value: {np.max(tri_table)}")
print(f"Min value: {np.min(tri_table)}")

# Check if any value is >= 12 (except -1)
mask = (tri_table != -1) & (tri_table >= 12)
if np.any(mask):
    print("Found invalid edge indices!")
    print(tri_table[mask])
else:
    print("All indices valid.")
