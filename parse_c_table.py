import re
import numpy as np

def parse_c_table():
    with open('c_code.txt', 'r') as f:
        content = f.read()
        
    # Extract content inside {{ ... }}
    # It starts with {{ and ends with }};
    start = content.find('{{')
    end = content.rfind('}}')
    
    if start == -1 or end == -1:
        print("Could not find table boundaries")
        return
        
    inner = content[start+1:end+1] # Include outer braces for regex?
    # Actually, content is `{{...}, {...}, ...}`
    # I want to split by `}, {`
    
    # Remove outer `{` and `}`
    inner = inner.strip('{}')
    
    # Split by `}, {`
    rows_str = inner.split('}, {')
    
    rows = []
    for r in rows_str:
        # Clean up
        r = r.replace('{', '').replace('}', '')
        vals = [int(x.strip()) for x in r.split(',') if x.strip()]
        rows.append(vals)
        
    print(f"Parsed {len(rows)} rows.")
    
    # Check lengths
    max_len = max(len(r) for r in rows)
    print(f"Max row length: {max_len}")
    
    # Pad to 16 (or max_len)
    target_len = 16
    if max_len > 16:
        print("Warning: Max length > 16!")
        target_len = max_len
        
    final_rows = []
    for r in rows:
        pad = target_len - len(r)
        final_rows.append(r + [-1]*pad)
        
    # Read existing file to keep edge_table
    with open('kalpana3d/marching_cubes_tables.py', 'r') as f:
        old_content = f.read()
        
    # Find where tri_table starts
    marker = "tri_table = np.array(["
    idx = old_content.find(marker)
    if idx == -1:
        print("Could not find tri_table in existing file")
        return
        
    # Keep everything before marker
    new_content = old_content[:idx]
    
    # Append new table
    new_content += "tri_table = np.array([\n"
    for r in final_rows:
        new_content += "    [" + ", ".join(map(str, r)) + "],\n"
    new_content += "], dtype=np.int32)\n"
    
    with open('kalpana3d/marching_cubes_tables.py', 'w') as f:
        f.write(new_content)
        
    print("Updated marching_cubes_tables.py")

if __name__ == "__main__":
    parse_c_table()
