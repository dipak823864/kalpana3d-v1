import numpy as np
from numba import njit

@njit(fastmath=True)
def vec3(x, y, z):
    a = np.empty(3, dtype=np.float32)
    a[0] = np.float32(x)
    a[1] = np.float32(y)
    a[2] = np.float32(z)
    return a

@njit(fastmath=True)
def dot(a, b):
    return np.float32(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

@njit(fastmath=True)
def length(v):
    return np.float32(np.sqrt(dot(v, v)))

@njit(fastmath=True)
def normalize(v):
    l = length(v)
    if l < 1e-8:
        return vec3(0.0, 0.0, 0.0)
    return v / l

@njit(fastmath=True)
def cross(a, b):
    res = np.empty(3, dtype=np.float32)
    res[0] = a[1]*b[2] - a[2]*b[1]
    res[1] = a[2]*b[0] - a[0]*b[2]
    res[2] = a[0]*b[1] - a[1]*b[0]
    return res

@njit(fastmath=True)
def mix(a, b, t):
    t_f32 = np.float32(t)
    return a * (np.float32(1.0) - t_f32) + b * t_f32
