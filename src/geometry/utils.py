import numpy as np


def triangle_normal(p0, p1, p2):
    """Compute the normal of a triangle."""
    return normalize(np.cross(p1 - p0, p2 - p0))

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def dot(v1, v2):
    """Compute dot product between two vectors."""
    return np.dot(v1, v2)

def cross(v1, v2):
    """Compute cross product between two vectors."""
    return np.cross(v1, v2)

def length(v):
    """Compute the length of a vector."""
    return np.linalg.norm(v)

def area_triangle(p0,p1,p2):
    v1 = p1 - p0
    v2 = p2 - p0
    
    # The magnitude of the cross product is twice the area of the triangle
    return length(cross(v1, v2)) / 2


