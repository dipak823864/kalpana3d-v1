import yaml
import numpy as np

def load_scene(filename):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
        
    scene_data = {}
    
    # Common fields to extract for each primitive type
    primitive_fields = {
        'spheres': {'pos': 'p', 'radius': 'r'},
        'capsules': {'a': 'a', 'b': 'b', 'radius': 'r'},
        'boxes': {'pos': 'p', 'dims': 'b'},
        'round_cones': {'a': 'a', 'b': 'b', 'r1': 'r1', 'r2': 'r2'},
        'torus': {'pos': 'p', 'r_main': 'r_main', 'r_tube': 'r_tube'}
    }

    for p_type, fields in primitive_fields.items():
        if p_type in data.get('scene', {}):
            items = data['scene'][p_type]
            count = len(items)
            
            # Prepare data dictionaries
            p_data = {'count': count}
            for key in fields.keys():
                p_data[key] = np.zeros((count, 3) if key in ['pos', 'dims', 'a', 'b'] else count, dtype=np.float32)
            
            p_data['translate'] = np.zeros((count, 3), dtype=np.float32)
            p_data['rotate'] = np.zeros((count, 3), dtype=np.float32)
            p_data['scale'] = np.ones(count, dtype=np.float32) # Default scale is 1

            for i, item in enumerate(items):
                for key, yaml_key in fields.items():
                    if key in ['pos', 'dims', 'a', 'b']:
                        p_data[key][i] = item[yaml_key]
                    else:
                        p_data[key][i] = item[yaml_key]

                # Transformations
                p_data['translate'][i] = item.get('translate', [0,0,0])
                p_data['rotate'][i] = item.get('rotate', [0,0,0])
                p_data['scale'][i] = item.get('scale', 1.0)

            scene_data[p_type] = p_data
        else:
            scene_data[p_type] = {'count': 0}
            
    return scene_data
