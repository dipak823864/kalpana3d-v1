import numpy as np
from numba import njit
from kalpana3d.math_core import vec3, dot, mix

# Simplex noise implementation based on Stefan Gustavson's paper
# Adapted for Numba

@njit(fastmath=True)
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

@njit(fastmath=True)
def grad(hash_val, x, y, z):
    h = hash_val & 15
    u = x if h < 8 else y
    # Fix: Avoid tuple membership check 'h in (12, 14)' for Numba compatibility
    v = y if h < 4 else (x if (h == 12 or h == 14) else z)
    
    res = 0.0
    if h & 1:
        res += u
    else:
        res -= u

    if h & 2:
        res += v
    else:
        res -= v

    return res

@njit
def snoise(p, perm):
    X = int(np.floor(p[0])) & 255
    Y = int(np.floor(p[1])) & 255
    Z = int(np.floor(p[2])) & 255
    
    px = p[0] - np.floor(p[0])
    py = p[1] - np.floor(p[1])
    pz = p[2] - np.floor(p[2])
    
    u = fade(px)
    v = fade(py)
    w = fade(pz)

    A = perm[X] + Y
    AA = perm[A] + Z
    AB = perm[A + 1] + Z
    B = perm[X + 1] + Y
    BA = perm[B] + Z
    BB = perm[B + 1] + Z

    g1 = grad(perm[AA], px, py, pz)
    g2 = grad(perm[BA], px - 1, py, pz)
    g3 = grad(perm[AB], px, py - 1, pz)
    g4 = grad(perm[BB], px - 1, py - 1, pz)
    g5 = grad(perm[AA + 1], px, py, pz - 1)
    g6 = grad(perm[BA + 1], px - 1, py, pz - 1)
    g7 = grad(perm[AB + 1], px, py - 1, pz - 1)
    g8 = grad(perm[BB + 1], px - 1, py - 1, pz - 1)

    return mix(w, mix(v, mix(u, g1, g2), mix(u, g3, g4)),
                  mix(v, mix(u, g5, g6), mix(u, g7, g8)))

@njit(fastmath=True)
def fbm(p, octaves, perm):
    p = p.astype(np.float32)
    v = np.float32(0.0)
    a = np.float32(0.5)
    shift = vec3(100.0, 100.0, 100.0) # vec3 creates float32
    for i in range(octaves):
        v += a * snoise(p, perm)
        p = p * np.float32(2.0) + shift
        a *= np.float32(0.5)
    return v

@njit(fastmath=True)
def domain_warp(p, noise_func, octaves, amplitude, perm):
    q = vec3(fbm(p, octaves, perm),
             fbm(p + vec3(5.2, 1.3, 8.3), octaves, perm),
             fbm(p + vec3(4.2, 6.3, 1.3), octaves, perm))
    return p + q * amplitude
