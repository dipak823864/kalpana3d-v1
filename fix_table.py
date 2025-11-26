import re
import numpy as np

def fix_table():
    with open('kalpana3d/marching_cubes_tables.py', 'r') as f:
        content = f.read()
    
    # Extract the tri_table part
    # It starts with "tri_table = np.array([" and ends with "], dtype=np.int32)"
    start_marker = "tri_table = np.array(["
    end_marker = "], dtype=np.int32)"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Could not find tri_table start")
        return
        
    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        print("Could not find tri_table end")
        return
        
    array_content = content[start_idx + len(start_marker):end_idx]
    
    # Parse the rows
    # Each row is [...]
    # We can use regex to find [...]
    rows = []
    matches = re.findall(r'\[(.*?)\]', array_content)
    
    max_len = 0
    parsed_rows = []
    for m in matches:
        # Split by comma
        vals = [int(x.strip()) for x in m.split(',') if x.strip()]
        # Remove -1s to get actual data
        vals = [v for v in vals if v != -1]
        parsed_rows.append(vals)
        if len(vals) > max_len:
            max_len = len(vals)
            
    print(f"Max length found: {max_len}")
    
    # Pad to max_len + 1 (for terminator) or just max_len if we don't use terminator?
    # My code checks for -1.
    # So we need at least max_len + 1 if the longest row doesn't have a terminator.
    # But wait, if the longest row is 18, and we make it 18, and it's full of data, we won't hit -1.
    # My mesher loop: `for i in range(0, 16, 3):`
    # It assumes max 16!
    # If I have 18, I need to update the mesher loop too.
    
    target_len = max(16, max_len + 1)
    # Let's make it 19 to be safe (multiple of 3? No, just enough).
    # 18 is divisible by 3.
    # If we have 18 indices, that's 6 triangles.
    # So we need to iterate up to 18.
    
    target_len = 19 # 18 + 1 for safety
    
    new_rows = []
    for row in parsed_rows:
        pad_count = target_len - len(row)
        new_row = row + [-1] * pad_count
        new_rows.append(new_row)
        
    # Reconstruct the file
    new_array_str = "tri_table = np.array([\n"
    for row in new_rows:
        new_array_str += "    [" + ", ".join(map(str, row)) + "],\n"
    new_array_str += "], dtype=np.int32)"
    
    new_content = content[:start_idx] + new_array_str + content[end_idx + len(end_marker):]
    
    with open('kalpana3d/marching_cubes_tables.py', 'w') as f:
        f.write(new_content)
        
    print("Fixed marching_cubes_tables.py")
    
    # Also need to update mesher.py to handle more than 16 indices
    return max_len

if __name__ == "__main__":
    fix_table()
