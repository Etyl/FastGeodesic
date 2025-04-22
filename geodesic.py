import numpy as np
from collections import defaultdict

from mesh import Mesh, SurfacePoint

def build_face_adjacency(mesh):
    """Returns a dict mapping (min_idx, max_idx) -> list of face indices sharing that edge."""
    edge_to_faces = defaultdict(list)
    for face_index, face in enumerate(mesh.faces):
        edges = [(face[i], face[(i+1)%3]) for i in range(3)]
        for v1, v2 in edges:
            edge = tuple(sorted((v1, v2)))
            edge_to_faces[edge].append(face_index)
    return edge_to_faces

def compute_barycentric(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def trace_straightest_geodesic(mesh, surface_point, direction, epsilon=1e-6):
    """
    Trace a straightest geodesic from a surface point in a given direction.
    
    :param mesh: Mesh object.
    :param surface_point: SurfacePoint (must be on a face).
    :param direction: 3D numpy array direction vector.
    :return: list of SurfacePoint representing the geodesic path.
    """
    if surface_point.location_type != 'face':
        raise ValueError("SurfacePoint must be on a face.")

    max_distance = np.linalg.norm(direction)
    if max_distance < epsilon:
        return [surface_point]
    
    direction = direction / max_distance  # Normalize for step tracking
    path = [surface_point]

    face_adjacency = build_face_adjacency(mesh)
    current_point = surface_point
    traveled = 0.0

    while traveled < max_distance:
        face_idx = current_point.index
        v_idx = mesh.faces[face_idx]
        v = [np.array(mesh.get_vertex(i)) for i in v_idx]

        # Local basis: edge1 and edge2 from first vertex
        e1 = v[1] - v[0]
        e2 = v[2] - v[0]
        normal = np.cross(e1, e2)
        normal = normal / np.linalg.norm(normal)
        local_x = e1 / np.linalg.norm(e1)
        local_y = np.cross(normal, local_x)

        # Project direction onto triangle plane
        proj_dir = direction - np.dot(direction, normal) * normal
        if np.linalg.norm(proj_dir) < epsilon:
            break  # Direction is perpendicular to triangle plane
        proj_dir /= np.linalg.norm(proj_dir)

        # Compute intersection with triangle edges in 2D barycentric space
        p = sum(b * v[i] for i, b in enumerate(current_point.barycentric_coords))
        bary = np.array(current_point.barycentric_coords)

        # Convert to 2D local coordinates
        origin = v[0]
        rel_p = p - origin
        rel_d = proj_dir

        # We want to move as far as possible within the triangle until we hit an edge
        # Compute step to each barycentric boundary (0)
        times = []
        for i in range(3):
            if rel_d.dot(np.cross(v[(i+1)%3] - v[i], normal)) > 0:  # Moving toward the edge
                if bary[i] > epsilon:
                    t = bary[i] / abs(np.dot(rel_d, v[i] - p))
                    times.append((t, i))
        
        if not times:
            break

        t_min, edge_hit = min(times, key=lambda x: x[0])
        step = min(t_min, (max_distance - traveled))
        next_p = p + proj_dir * step

        # Convert next_p to barycentric coords
        u, v_, w = compute_barycentric(next_p, *v)
        bary_coords = (u, v_, w)

        if step < epsilon:
            break

        traveled += step
        current_point = SurfacePoint(mesh, 'face', face_idx, bary_coords)
        path.append(current_point)

        # Transition to neighboring face if hitting edge
        if step == t_min:
            edge = tuple(sorted((v_idx[edge_hit], v_idx[(edge_hit+1)%3])))
            neighbors = face_adjacency[edge]
            next_face = [f for f in neighbors if f != face_idx]
            if not next_face:
                break  # Boundary edge
            current_point.index = next_face[0]
    
    return path
