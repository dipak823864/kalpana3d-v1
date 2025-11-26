import yaml
import numpy as np

def load_scene(filename):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
        
    scene_data = {}
    
    # Spheres
    if 'spheres' in data['scene']:
        items = data['scene']['spheres']
        count = len(items)
        pos = np.zeros((count, 3), dtype=np.float32)
        radii = np.zeros(count, dtype=np.float32)
        
        for i, item in enumerate(items):
            pos[i] = item['p']
            radii[i] = item['r']
            
        scene_data['spheres'] = {
            'pos': pos,
            'radius': radii,
            'count': count
        }
    else:
        scene_data['spheres'] = {'count': 0}
        
    # Capsules
    if 'capsules' in data['scene']:
        items = data['scene']['capsules']
        count = len(items)
        a = np.zeros((count, 3), dtype=np.float32)
        b = np.zeros((count, 3), dtype=np.float32)
        radii = np.zeros(count, dtype=np.float32)
        
        for i, item in enumerate(items):
            a[i] = item['a']
            b[i] = item['b']
            radii[i] = item['r']
            
        scene_data['capsules'] = {
            'a': a,
            'b': b,
            'radius': radii,
            'count': count
        }
    else:
        scene_data['capsules'] = {'count': 0}
        
    # Boxes
    if 'boxes' in data['scene']:
        items = data['scene']['boxes']
        count = len(items)
        pos = np.zeros((count, 3), dtype=np.float32)
        dims = np.zeros((count, 3), dtype=np.float32)
        
        for i, item in enumerate(items):
            pos[i] = item['p']
            dims[i] = item['b']
            
        scene_data['boxes'] = {
            'pos': pos,
            'dims': dims,
            'count': count
        }
    else:
        scene_data['boxes'] = {'count': 0}
        
    # Round Cones (Tapered Capsules)
    if 'round_cones' in data['scene']:
        items = data['scene']['round_cones']
        count = len(items)
        a = np.zeros((count, 3), dtype=np.float32)
        b = np.zeros((count, 3), dtype=np.float32)
        r1 = np.zeros(count, dtype=np.float32)
        r2 = np.zeros(count, dtype=np.float32)
        
        for i, item in enumerate(items):
            a[i] = item['a']
            b[i] = item['b']
            r1[i] = item['r1']
            r2[i] = item['r2']
            
        scene_data['round_cones'] = {
            'a': a,
            'b': b,
            'r1': r1,
            'r2': r2,
            'count': count
        }
    else:
        scene_data['round_cones'] = {'count': 0}

    # Torus
    if 'torus' in data['scene']:
        items = data['scene']['torus']
        count = len(items)
        pos = np.zeros((count, 3), dtype=np.float32)
        r_main = np.zeros(count, dtype=np.float32)
        r_tube = np.zeros(count, dtype=np.float32)
        
        for i, item in enumerate(items):
            pos[i] = item['p']
            r_main[i] = item['r_main']
            r_tube[i] = item['r_tube']
            
        scene_data['torus'] = {
            'pos': pos,
            'r_main': r_main,
            'r_tube': r_tube,
            'count': count
        }
    else:
        scene_data['torus'] = {'count': 0}
        
    return scene_data
