import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d.parser import load_scene

def main():
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples/test_scene.yaml'))
    print(f"Loading {yaml_path}...")
    
    scene = load_scene(yaml_path)
    
    print("Parsed Scene Data:")
    for key, val in scene.items():
        if val['count'] > 0:
            print(f"  {key}: {val['count']} items")
            for k, v in val.items():
                if k != 'count':
                    print(f"    {k}: shape {v.shape}, dtype {v.dtype}")
                    print(f"    {v}")
        else:
            print(f"  {key}: 0 items")

if __name__ == "__main__":
    main()
