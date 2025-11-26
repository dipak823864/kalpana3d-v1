import numpy as np
from numba import njit
from kalpana3d.parser import load_scene
from kalpana3d.sdf import sdCapsule, sdBox, opSmoothUnion
from kalpana3d.noise import domain_warp, fbm
from kalpana3d.mesher import compute_mesh_counts, generate_mesh
from kalpana3d.export import save_obj
from kalpana3d.math_core import vec3

def create_sdf_function(capsules_data, boxes_data, settings, perm):
    # Extract numpy arrays from dicts
    capsules_a = capsules_data.get('a', np.empty((0,3), dtype=np.float32))
    capsules_b = capsules_data.get('b', np.empty((0,3), dtype=np.float32))
    capsules_radius = capsules_data.get('radius', np.empty(0, dtype=np.float32))
    capsules_translate = capsules_data.get('translate', np.empty((0,3), dtype=np.float32))
    capsules_count = capsules_data.get('count', 0)

    boxes_pos = boxes_data.get('pos', np.empty((0,3), dtype=np.float32))
    boxes_dims = boxes_data.get('dims', np.empty((0,3), dtype=np.float32))
    boxes_translate = boxes_data.get('translate', np.empty((0,3), dtype=np.float32))
    boxes_count = boxes_data.get('count', 0)

    noise_octaves = settings['noise_octaves']
    noise_amplitude = np.float32(settings['noise_amplitude'])
    smooth_union_k = np.float32(settings['smooth_union_k'])

    @njit(fastmath=True)
    def sdf_func(p):
        p_warped = domain_warp(p, fbm, noise_octaves, noise_amplitude, perm)

        dist = np.float32(1e9)

        # Process capsules
        for i in range(capsules_count):
            a = capsules_a[i] + capsules_translate[i]
            b = capsules_b[i] + capsules_translate[i]
            r = capsules_radius[i]
            d = sdCapsule(p_warped, a, b, r)
            dist = opSmoothUnion(dist, d, smooth_union_k)

        # Process boxes
        for i in range(boxes_count):
            pos = boxes_pos[i] + boxes_translate[i]
            dims = boxes_dims[i]
            d = sdBox(p_warped - pos, dims)
            dist = opSmoothUnion(dist, d, smooth_union_k)

        return dist

    return sdf_func

if __name__ == '__main__':
    scene_data = load_scene('examples/twisted_creature.yaml')
    with open('examples/twisted_creature.yaml', 'r') as f:
        import yaml
        yaml_data = yaml.safe_load(f)
        settings = yaml_data['settings']

    perm = np.random.permutation(256).astype(np.int32)
    perm = np.concatenate((perm, perm))

    sdf_func = create_sdf_function(scene_data['capsules'], scene_data['boxes'], settings, perm)

    min_bound = vec3(-3, -3, -3)
    max_bound = vec3(3, 3, 3)
    resolution = vec3(128, 128, 128)
    iso_level = 0.0

    max_triangles = compute_mesh_counts(min_bound, max_bound, resolution, sdf_func, iso_level)
    vertices = generate_mesh(min_bound, max_bound, resolution, sdf_func, iso_level, max_triangles)

    save_obj(vertices, 'gallery/models/twisted_creature.obj')
