import numpy as np
from typing import List,Optional

from geometry.mesh import Mesh, MeshPoint
from geometry.utils import dot, cross, length, normalize, triangle_normal

# TODO: add docstrings to all functions
# TODO: add type hints to all functions
# TODO: fix when start is on edge or vertex

EPS = 1e-6

class GeodesicPath:
    def __init__(self):
        self.start : Optional[MeshPoint] = None
        self.end : Optional[MeshPoint] = None
        self.path : List[np.ndarray] = []
        self.dirs : List[np.ndarray] = []


def tri_bary_coords(p0, p1, p2, p):
    """Compute barycentric coordinates of p in the triangle (p0, p1, p2)."""
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0
    
    d00 = dot(v0, v0)
    d01 = dot(v0, v1)
    d11 = dot(v1, v1)
    d20 = dot(v2, v0)
    d21 = dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < EPS:
        return np.array([1.0, 0.0, 0.0])
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
  
    return np.array([u, v, w])

def point_is_edge(point: MeshPoint):
    """Check if a mesh point is on an edge and return the edge index."""
    uv = point.uv
    
    if abs(uv[0]) < EPS:
        return True, 2
    if abs(uv[1]) < EPS:
        return True, 0
    if abs(1 - uv[0] - uv[1]) < EPS:
        return True, 1
    
    return False, -1

def point_is_vert(point: MeshPoint):
    """Check if a mesh point is on a vertex and return the vertex index."""
    uv = point.uv
    
    if abs(uv[0]) < EPS and abs(uv[1]) < EPS:
        return True, 0
    if abs(uv[0] - 1.0) < EPS and abs(uv[1]) < EPS:
        return True, 1
    if abs(uv[0]) < EPS and abs(uv[1] - 1.0) < EPS:
        return True, 2
    
    return False, -1

def bary_is_edge(bary):
    """Check if barycentric coordinates are on an edge and return the edge index."""
    
    if abs(bary[0]) < EPS:
        return True, 1
    if abs(bary[1]) < EPS:
        return True, 2
    if abs(bary[2]) < EPS:
        return True, 0
    
    return False, -1

def bary_is_vert(bary):
    """Check if barycentric coordinates are on a vertex and return the vertex index."""
    
    if abs(bary[0] - 1.0) < EPS:
        return True, 0
    if abs(bary[1] - 1.0) < EPS:
        return True, 1
    if abs(bary[2] - 1.0) < EPS:
        return True, 2
    
    return False, -1

def project_vec(v, normal):
    """Project a vector onto a plane defined by its normal."""
    return v - dot(v, normal) * normal


def closest_point_parameter_coplanar(P1, d1, P2, d2):
    """
    Finds the closest points on two lines in 3D.
    """
    # Ensure numpy arrays
    P1, d1 = np.array(P1, dtype=float), np.array(d1, dtype=float)
    P2, d2 = np.array(P2, dtype=float), np.array(d2, dtype=float)

    # Normalize direction vectors
    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)

    # Compute intermediate values
    r = P1 - P2
    a = np.dot(d1_norm, d1_norm)
    b = np.dot(d1_norm, d2_norm)
    c = np.dot(d2_norm, d2_norm)
    d = np.dot(d1_norm, r)
    e = np.dot(d2_norm, r)

    denominator = a * c - b * b

    if abs(denominator) < EPS:
        return -1,-1

    # Solve for parameters
    t1 = (b * e - c * d) / denominator
    t2 = (a * e - b * d) / denominator

    return t1,t2


def trace_in_triangles(positions, triangles, dir_3d, curr_bary, curr_tri, next_pos, next_bary,max_len):
    """Trace a straight line within triangles."""
    # Get the triangle vertices
    p0 = positions[triangles[curr_tri][0]]
    p1 = positions[triangles[curr_tri][1]]
    p2 = positions[triangles[curr_tri][2]]
        
    # Get the current position
    curr_pos = curr_bary[0] * p0 + curr_bary[1] * p1 + curr_bary[2] * p2
    
    # Find the intersection with the triangle edges
    intersections = []
    
    # Check each edge
    edges = [(0, 1), (1, 2), (2, 0)]
    for edge_idx, (i, j) in enumerate(edges):
        vi = triangles[curr_tri][i]
        vj = triangles[curr_tri][j]
        
        p_i = positions[vi]
        p_j = positions[vj]
        
        edge_dir = p_j - p_i
        normal = cross(dir_3d, edge_dir)
        
        if length(normal) < EPS:
            continue
        
        # Find the intersection parameter t
        t,_ = closest_point_parameter_coplanar(curr_pos, dir_3d, p_i, p_j-p_i)
        
        if t <= EPS:
            continue
        
        intersection = curr_pos + t * dir_3d
        
        # Check if the intersection is on the edge
        edge_param = dot(intersection - p_i, edge_dir) / dot(edge_dir, edge_dir)
        
        if edge_param < 0 or edge_param > 1:
            continue
        
        # Valid intersection
        intersections.append((t, intersection, edge_idx, edge_param))
    
    if not intersections:
        # No intersection, use the direction
        next_pos[:] = curr_pos + dir_3d
        next_bary[:] = tri_bary_coords(p0, p1, p2, next_pos)
        # TODO stop geodesic
        return
    
    # Sort intersections by distance
    intersections.sort(key=lambda x: x[0])
    
    # Get the closest intersection
    _, intersection, edge_idx, edge_param = intersections[0]

    if length(curr_pos-intersection)>max_len:
        # No intersection, use the direction
        next_pos[:] = curr_pos + max_len*dir_3d
        next_bary[:] = tri_bary_coords(p0, p1, p2, next_pos)
        return

    next_pos[:] = intersection

    # Compute barycentric coordinates at the intersection
    i, j = edges[edge_idx]
    next_bary[:] = np.zeros(3)
    next_bary[i] = 1 - edge_param
    next_bary[j] = edge_param


def common_edge(triangles, tri1, tri2):
    """Find the common edge between two triangles."""
    # Get the vertices of both triangles
    verts1 = set(triangles[tri1,:])
    verts2 = set(triangles[tri2,:])
    
    # Find the common vertices
    common_verts = verts1.intersection(verts2)
    diff_verts = list(verts1-common_verts), list(verts2-common_verts)
    
    if len(common_verts) != 2:
        return np.array([-1, -1]), []
    
    return np.array(list(common_verts)), diff_verts

def check_point(point:MeshPoint):
    """Check if a mesh point is valid."""
    assert point.uv[0] >= -EPS and point.uv[1] >= -EPS
    assert point.uv[0] + point.uv[1] <= 1 + 2*EPS

def signed_angle(A, B, N):
    """
    Compute the signed angle between two vectors A and B with respect to a normal vector N.
    """
    N = N / length(N)
    A = A - dot(A,N)*N
    B = B - dot(B,N)*N
    if length(A) < EPS or length(B) < EPS:
        return 0.0
    A = A / length(A)
    B = B / length(B)

    cross_prod = cross(A, B)
    dot_prod = dot(A, B)
    sign = dot(N, cross_prod)
    angle = np.arctan2(sign, dot_prod)
    return angle  # in radians

def compute_parallel_transport(mesh, curr_pos, curr_tri, next_pos, next_tri, dir_3d):
    """
    Compute the parallel transport of a vector from one triangle to another.
    """
    if curr_tri == next_tri:
        return dir_3d
    
    # Find the common edge between triangles
    common_e, other_v = common_edge(mesh.triangles, curr_tri, next_tri)
    
    if common_e[0] == -1:
        return dir_3d  # No common edge found
    
    # Get the normals of both triangles
    p0 = mesh.positions[common_e[0]]
    p1 = mesh.positions[common_e[1]]
    p2_curr = mesh.positions[other_v[0][0]]
    p2_next = mesh.positions[other_v[1][0]]
    
    n1 = triangle_normal(p0,p1,p2_curr)
    n2 = triangle_normal(p0,p1,p2_next)
    
    # Get the edge direction
    edge_dir = mesh.positions[common_e[1]] - mesh.positions[common_e[0]]
    axis = normalize(edge_dir)

    sym_axis = normalize(cross(axis,n1))
    dir_3d = dir_3d - 2*(dir_3d@sym_axis)*sym_axis
    
    # Compute the rotation angle
    angle = signed_angle(n1,n2,axis)

    # Rodrigues' rotation formula
    return dir_3d * np.cos(angle) + cross(axis, dir_3d) * np.sin(angle) + axis * dot(axis, dir_3d) * (1 - np.cos(angle))


def compute_parallel_transport_vertex(mesh:Mesh, curr_pos, curr_tri, vertex_id, dir_3d):
    """
    Compute the parallel transport of a vector at a vertex.
    """
    connected_triangles = mesh.v2t[vertex_id]
    total_angle = 0.0
    for tri_id in connected_triangles:
        vertices = [0,0]
        idx = 0
        for v in mesh.triangles[tri_id]:
            if v != vertex_id:
                vertices[idx] = v
                idx += 1

        n = mesh.triangle_normals[tri_id]

        angle = signed_angle(
            mesh.positions[vertices[0]]-mesh.positions[vertex_id],
            mesh.positions[vertices[1]]-mesh.positions[vertex_id],
            n
        )
        total_angle += abs(angle)
                
    half_angle = total_angle/2
    angle = 0
    
    # get initial angle
    dir_3d = -dir_3d
    v1 = -1
    for v in mesh.triangles[curr_tri]:
        if v!=vertex_id:
            v1 = v
            break

    n = mesh.triangle_normals[tri_id]
    angle += abs(signed_angle(
        dir_3d,
        mesh.positions[v1]-mesh.positions[vertex_id],
        n
    ))

    while angle < half_angle:
        # Get next triangle
        v2 = (set(mesh.triangles[curr_tri])-{v1,vertex_id}).pop()
        local_edge_idx = -1
        for idx,v in enumerate(mesh.triangles[curr_tri]):
            if v != v1 and v != vertex_id:
                local_edge_idx = (idx+1)%3
                break
        next_tri = mesh.adjacencies[curr_tri][local_edge_idx]
        v2 = (set(mesh.triangles[next_tri])-{v1,vertex_id}).pop()
        
        p0 = mesh.positions[vertex_id]
        p1 = mesh.positions[v1]
        p2 = mesh.positions[v2]
        n = triangle_normal(p0,p1,p2)

        tri_angle = abs(signed_angle(
            mesh.positions[v1]-mesh.positions[vertex_id],
            mesh.positions[v2]-mesh.positions[vertex_id],
            n
        ))

        if angle+tri_angle >= half_angle-EPS:
            angle_diff = half_angle-angle # TODO angle orientation
            axis = n
            edge = mesh.positions[v1]-mesh.positions[vertex_id]
            new_dir = edge * np.cos(angle_diff) + cross(axis, edge) * np.sin(angle_diff) + axis * dot(axis, edge) * (1 - np.cos(angle_diff))
            return new_dir, next_tri
        
        curr_tri = next_tri
        v1 = v2
        angle += tri_angle

    return np.zeros(3),0


def bary_to_uv(bary):
    """Convert barycentric coordinates to UV coordinates."""
    return np.array([bary[1], bary[2]])

def straightest_geodesic(mesh:Mesh, start:MeshPoint, dir:np.ndarray) -> GeodesicPath:
    """
    Compute the straightest geodesic path on a mesh.
    
    Args:
        mesh: The bezier mesh
        start: The starting point on the mesh
        dir: The initial 3D direction
    
    Returns:
        A geodesic path
    """
    # Get the triangle normal
    tid_normal = mesh.triangle_normals[start.face]

    len_path = 0.0
    path_len = length(dir)
    
    # Project the direction onto the triangle plane
    dir = project_vec(dir, tid_normal)

    geodesic = GeodesicPath()
    geodesic.start = start
    geodesic.path.append(start.interpolate(mesh))
    geodesic.dirs.append(normalize(dir))
    
    next_bary = np.zeros(3)
    curr_bary = np.array([1 - start.uv[0] - start.uv[1], start.uv[0], start.uv[1]])
    curr_pos = start.interpolate(mesh)
    next_pos = np.zeros(3)
    curr_point = start
    tid_normal = np.zeros(3)
    curr_tri = start.face
    next_tri = -1
    
    while len_path < path_len:
        # Get the triangle normal
        tid_normal = mesh.triangle_normals[curr_tri]
        
        # Project the direction onto the triangle plane
        proj_dir = project_vec(dir, tid_normal)            
        if length(proj_dir) < EPS:
            # Direction is perpendicular to the triangle, cannot proceed
            break
        
        proj_dir = normalize(proj_dir)
        
        # Trace the ray in the current triangle
        trace_in_triangles(mesh.positions, mesh.triangles, proj_dir, curr_bary, curr_tri, next_pos, next_bary, path_len-len_path)
        
        # Check if the point is on an edge or vertex
        is_edge_bary, edge_idx = bary_is_edge(next_bary)
        is_vert_bary, vert_idx = bary_is_vert(next_bary)
        
        # Update the path
        len_path += length(next_pos - curr_pos)
        geodesic.path.append(next_pos.copy())
        if len(geodesic.path) == 22:
            x = 0
        
        # Determine the next triangle
        if is_vert_bary:
            # Point is on a vertex
            v_idx = mesh.triangles[curr_tri][vert_idx]
            
            # Transport the direction to the next triangle
            dir, next_tri = compute_parallel_transport_vertex(mesh, curr_pos, curr_tri, v_idx, proj_dir)
            geodesic.dirs.append(dir)

            # Compute barycentric coordinates in the adjacent triangle
            p0_adj = mesh.positions[mesh.triangles[next_tri][0]]
            p1_adj = mesh.positions[mesh.triangles[next_tri][1]]
            p2_adj = mesh.positions[mesh.triangles[next_tri][2]]
            
            next_bary = tri_bary_coords(p0_adj, p1_adj, p2_adj, next_pos)
            
            # Update current state
            curr_tri = next_tri
            curr_pos = next_pos.copy()
            curr_bary = next_bary.copy()
            
            # Create a new mesh point for current position
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary))
            
        elif is_edge_bary:
            # Point is on an edge
            # Find the adjacent triangle across this edge
            edge_local_idx = edge_idx
            adj_tri = mesh.adjacencies[curr_tri][edge_local_idx]
            if adj_tri == -1:
                adj_tri = curr_tri

            # Compute barycentric coordinates in the adjacent triangle
            p0_adj = mesh.positions[mesh.triangles[adj_tri][0]]
            p1_adj = mesh.positions[mesh.triangles[adj_tri][1]]
            p2_adj = mesh.positions[mesh.triangles[adj_tri][2]]
            
            if adj_tri == -1:
                next_bary = tri_bary_coords(p0_adj, p1_adj, p2_adj, next_pos)
                curr_point = MeshPoint(adj_tri, bary_to_uv(next_bary))
                geodesic.dirs.append(dir) # TODO same for vertex
                break
            
            # Get the common edge vertices
            e,_ = common_edge(mesh.triangles, curr_tri, adj_tri)
            
            if e[0] == -1:
                # No common edge found
                next_bary = tri_bary_coords(p0_adj, p1_adj, p2_adj, next_pos)                
                curr_point = MeshPoint(adj_tri, bary_to_uv(next_bary))
                break
            
            # Transport the direction to the adjacent triangle
            dir = compute_parallel_transport(mesh, curr_pos, curr_tri, next_pos, adj_tri, proj_dir)
            geodesic.dirs.append(dir)
            
            next_bary = tri_bary_coords(p0_adj, p1_adj, p2_adj, next_pos)
            
            # Update current state
            curr_tri = adj_tri
            curr_pos = next_pos.copy()
            curr_bary = next_bary.copy()
            
            # Create a new mesh point for current position
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary))
            
        else:
            # Point is in the interior of the triangle
            # Update current state
            curr_tri = curr_tri  # No change in triangle
            curr_pos = next_pos.copy()
            curr_bary = next_bary.copy()
            
            # Create a new mesh point for current position
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary))
            
            # Continue in the same direction
            geodesic.dirs.append(proj_dir)
    
    # Set the end point
    geodesic.end = curr_point
    
    return geodesic

