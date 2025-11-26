import numpy as np
from numba import njit
from kalpana3d.math_core import dot, length, vec3, mix

@njit(fastmath=True)
def sdSphere(p, r):
    return length(p) - r

@njit(fastmath=True)
def sdBox(p, b):
    q = np.abs(p) - b
    max_q_x = max(q[0], 0.0)
    max_q_y = max(q[1], 0.0)
    max_q_z = max(q[2], 0.0)
    len_max_q = np.sqrt(max_q_x*max_q_x + max_q_y*max_q_y + max_q_z*max_q_z)
    inner = min(max(q[0], max(q[1], q[2])), 0.0)
    return len_max_q + inner

@njit(fastmath=True)
def sdCylinder(p, h, r):
    d_x = length(np.array([p[0], p[2]], dtype=np.float32)) - r
    d_y = np.abs(p[1]) - h
    d_x_clamped = max(d_x, 0.0)
    d_y_clamped = max(d_y, 0.0)
    dist_exterior = np.sqrt(d_x_clamped*d_x_clamped + d_y_clamped*d_y_clamped)
    dist_interior = min(max(d_x, d_y), 0.0)
    return dist_exterior + dist_interior

@njit(fastmath=True)
def sdCapsule(p, a, b, r):
    pa = p - a
    ba = b - a
    h = dot(pa, ba) / dot(ba, ba)
    h = min(max(h, 0.0), 1.0)
    return length(pa - ba * h) - r

@njit(fastmath=True)
def sdRoundCone(p, a, b, r1, r2):
    # a, b: start and end points
    # r1, r2: start and end radii
    ba = b - a
    l2 = dot(ba, ba)
    rr = r1 - r2
    a2 = l2 - rr*rr
    il2 = 1.0 / l2
    
    pa = p - a
    y = dot(pa, ba)
    z = y - l2
    
    # x = length(pa*l2 - ba*y)
    # This vector calculation might be unstable if l2 is small?
    # pa*l2 - ba*y = pa*dot(ba,ba) - ba*dot(pa,ba) = cross(cross(ba, pa), ba) ?
    # Let's stick to the formula
    
    v = pa * l2 - ba * y
    x2 = dot(v, v)
    y2 = y*y*l2
    z2 = z*z*l2
    
    k = np.sign(rr)*rr*rr*x2
    
    if np.sign(z)*a2*z2 > k:
        return np.sqrt(x2 + z2) * il2 - r2
    
    if np.sign(y)*a2*y2 < k:
        return np.sqrt(x2 + y2) * il2 - r1
        
    return (np.sqrt(x2*a2*il2) + y*rr) * il2 - r1

@njit(fastmath=True)
def sdTorus(p, r_main, r_tube):
    q_x = length(np.array([p[0], p[2]], dtype=np.float32)) - r_main
    q_y = p[1]
    return np.sqrt(q_x*q_x + q_y*q_y) - r_tube

@njit(fastmath=True)
def opUnion(d1, d2):
    return min(d1, d2)

@njit(fastmath=True)
def opSubtraction(d1, d2):
    return max(-d1, d2)

@njit(fastmath=True)
def opIntersection(d1, d2):
    return max(d1, d2)

@njit(fastmath=True)
def opSmoothUnion(d1, d2, k):
    h = max(k - abs(d1 - d2), 0.0) / k
    return min(d1, d2) - h*h*k*(1.0/4.0)

# Space Folding / Modifiers

@njit(fastmath=True)
def opTwist(p, k):
    c = np.cos(k * p[1])
    s = np.sin(k * p[1])
    # Rotate xz
    # [c -s]
    # [s  c]
    qx = c * p[0] - s * p[2]
    qz = s * p[0] + c * p[2]
    return vec3(qx, p[1], qz)

@njit(fastmath=True)
def opBend(p, k):
    c = np.cos(k * p[0])
    s = np.sin(k * p[0])
    # Rotate xy
    qx = c * p[0] - s * p[1]
    qy = s * p[0] + c * p[1]
    return vec3(qx, qy, p[2])
    
# Taper is usually handled by the primitive (RoundCone), but generic taper:
@njit(fastmath=True)
def opTaper(p, k):
    # Scale xz by y
    # This is not distance preserving!
    # But for small k it's okay.
    # Or use RoundCone for exact distance.
    return p # Placeholder, prefer RoundCone
