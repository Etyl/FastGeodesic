import numpy as np
from typing import List,Optional,Tuple

from fastgeodesic.geometry.mesh import Mesh, MeshPoint
from fastgeodesic.geometry.utils import dot, cross, length, normalize, triangle_normal
from fastgeodesic.constants import EPS


class GeodesicPath:
    def __init__(self):
        self.start : Optional[MeshPoint] = None
        self.end : Optional[MeshPoint] = None
        self.path : List[np.ndarray] = []
        self.dirs : List[np.ndarray] = []
        self.normals : List[np.ndarray] = []


def tri_bary_coords(p0, p1, p2, p) -> np.ndarray:
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
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    assert u > -EPS and v > -EPS and w > -EPS and u + v + w < 1+3*EPS, f"Invalid barycentric coordinates: {u}, {v}, {w}"
  
    return np.array([u, v, w])

def point_is_edge(point: MeshPoint) -> Tuple[bool,int]:
    """Check if a mesh point is on an edge and return the edge index."""
    uv = point.uv
    
    if abs(uv[0]) < EPS:
        return True, 2
    if abs(uv[1]) < EPS:
        return True, 0
    if abs(1 - uv[0] - uv[1]) < EPS:
        return True, 1
    
    return False, -1

def point_is_vert(point: MeshPoint) -> Tuple[bool,int]:
    """Check if a mesh point is on a vertex and return the vertex index."""
    uv = point.uv
    
    if abs(uv[0]) < EPS and abs(uv[1]) < EPS:
        return True, 0
    if abs(uv[0] - 1.0) < EPS and abs(uv[1]) < EPS:
        return True, 1
    if abs(uv[0]) < EPS and abs(uv[1] - 1.0) < EPS:
        return True, 2
    
    return False, -1

def bary_is_edge(bary) -> Tuple[bool,int]:
    """Check if barycentric coordinates are on an edge and return the edge index."""
    
    if abs(bary[0]) < EPS:
        return True, 1
    if abs(bary[1]) < EPS:
        return True, 2
    if abs(bary[2]) < EPS:
        return True, 0
    
    return False, -1

def bary_is_vert(bary) -> Tuple[bool,int]:
    """Check if barycentric coordinates are on a vertex and return the vertex index."""
    
    if abs(bary[0] - 1.0) < EPS:
        return True, 0
    if abs(bary[1] - 1.0) < EPS:
        return True, 1
    if abs(bary[2] - 1.0) < EPS:
        return True, 2
    
    return False, -1

def project_vec(v, normal) -> np.ndarray:
    """Project a vector onto a plane defined by its normal."""
    normal = normalize(normal)
    return v - dot(v, normal) * normal


def closest_point_parameter_coplanar(P1, d1, P2, d2) -> Tuple[float, float]:
    """
    Finds the closest points on two lines in 3D.
    Returns the parameters t1 and t2 for the closest points on the lines.
    d1 is supposed to be normalized
    """
    d2_length = length(d2)
    d2 = normalize(d2)
    A = np.array([d1, -d2])
    M = A @ A.T
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < EPS:
        # d1 and d2 colinear
        # check if P1 and P2 are on same line
        if abs(normalize(P2-P1)@d1) < EPS:
            if length(P2-P1)<EPS:
                P = P2+d2*d2_length
                s = 1
            else:
                P = P2
                s = 0
            return ((P-P1)@d1), s
        else:
            return -1, -1
    
    M_inv = np.array([
        [M[1, 1], -M[0, 1]], 
        [-M[1, 0], M[0, 0]]
    ]) / det

    A_inv = A.T @ M_inv
    res = (P2 - P1) @ A_inv

    return res[0], res[1]/d2_length


def trace_in_triangles(mesh: Mesh, dir_3d:np.ndarray, curr_point:MeshPoint, curr_tri:int, max_len:float) -> Tuple[np.ndarray,np.ndarray]:
    """Trace a straight line within triangles."""
    # Get the current position
    curr_pos = curr_point.interpolate(mesh)
    
    # Find the intersection with the triangle edges
    intersections = []
    
    # Check each edge
    edges = [(0, 1), (1, 2), (2, 0)]
    for edge_idx, (i, j) in enumerate(edges):
        vi = mesh.triangles[curr_tri][i]
        vj = mesh.triangles[curr_tri][j]
        
        p_i = mesh.positions[vi]
        p_j = mesh.positions[vj]
        
        edge_dir = p_j - p_i
        normal = cross(dir_3d, edge_dir)
        
        if length(normal) < EPS:
            continue
        
        # Find the intersection parameter t
        t,edge_param = closest_point_parameter_coplanar(curr_pos, dir_3d, p_i, edge_dir)
        
        if t < -EPS or edge_param < -EPS or edge_param > 1+EPS:
            continue
        
        # Valid intersection
        intersection = curr_pos + t * dir_3d
        intersections.append((t, intersection, edge_idx, edge_param))
    
    if not intersections:
        # No intersection
        return curr_point, curr_point.get_barycentric_coords()
    
    # Sort intersections by distance
    intersections.sort(key=lambda x: x[0])

    idx = 0
    if len(intersections) > 1:
        if len(intersections) > 2 and intersections[1][0] < EPS:
            idx = 2
        elif intersections[0][0] < EPS:
            idx = 1

    _, intersection, edge_idx, edge_param = intersections[idx]

    if length(curr_pos-intersection)>max_len:
        # Intersection is too far, move for max_len in the direction
        next_pos = curr_pos + max_len*dir_3d
        p0 = mesh.positions[mesh.triangles[curr_tri][0]]
        p1 = mesh.positions[mesh.triangles[curr_tri][1]]
        p2 = mesh.positions[mesh.triangles[curr_tri][2]]
        next_bary = tri_bary_coords(p0, p1, p2, next_pos)
        return next_pos, next_bary

    next_pos = intersection

    # Compute barycentric coordinates at the intersection
    i, j = edges[edge_idx]
    next_bary = np.zeros(3)
    next_bary[i] = 1 - edge_param
    next_bary[j] = edge_param

    return next_pos, next_bary


def common_edge(triangles:np.ndarray, tri1:int, tri2:int) -> Tuple[List[int], Optional[int], Optional[int]]:
    """Find the common edge between two triangles."""
    # Get the vertices of both triangles
    verts1 = set(triangles[tri1,:])
    verts2 = set(triangles[tri2,:])
    
    # Find the common vertices
    common_verts = verts1.intersection(verts2)
    diff_vert_1 = verts1-common_verts
    diff_vert_2 = verts2-common_verts
    diff_vert_1 = diff_vert_1.pop() if diff_vert_1 else None
    diff_vert_2 = diff_vert_2.pop() if diff_vert_2 else None

    if len(common_verts) != 2:
        return [], None, None
    
    return list(common_verts), diff_vert_1, diff_vert_2


def signed_angle(A:np.ndarray, B:np.ndarray, N:np.ndarray) -> float:
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


def compute_parallel_transport_edge(mesh:Mesh, curr_tri:int, next_tri:int, dir_3d:np.ndarray, curr_normal:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """
    Compute the parallel transport of a vector from one triangle to another at an edge.
    """
    if curr_tri == next_tri:
        return dir_3d, curr_normal
    
    # Find the common edge between triangles
    common_e, other_v1, other_v2 = common_edge(mesh.triangles, curr_tri, next_tri)
    
    if len(common_e)==0:
        return dir_3d, curr_normal  # No common edge found
    
    # Get the normals of both triangles
    p0 = mesh.positions[common_e[0]]
    p1 = mesh.positions[common_e[1]]
    p2_curr = mesh.positions[other_v1]
    p2_next = mesh.positions[other_v2]
    
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
    dir = dir_3d * np.cos(angle) + cross(axis, dir_3d) * np.sin(angle) + axis * dot(axis, dir_3d) * (1 - np.cos(angle))
    
    normal_sign = dot(n1, curr_normal)
    normal = normal_sign * (-n2)
    return dir, normal


def compute_parallel_transport_vertex(mesh:Mesh, curr_tri:int, vertex_id:int, dir_3d:np.ndarray, curr_normal:np.ndarray) -> Tuple[np.ndarray,int,np.ndarray]:
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
    v2 = (set(mesh.triangles[curr_tri])-{v1,vertex_id}).pop()
    p0 = mesh.positions[vertex_id]
    p1 = mesh.positions[v1]
    p2 = mesh.positions[v2]
    n = triangle_normal(p0,p1,p2)
    if dot(n, curr_normal) > 0:
        normal_sign = 1
    else:
        normal_sign = -1
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
        if next_tri==-1:
            angle_diff = half_angle-angle
            axis = n
            edge = mesh.positions[v1]-mesh.positions[vertex_id]
            new_dir = edge * np.cos(angle_diff) + cross(axis, edge) * np.sin(angle_diff) + axis * dot(axis, edge) * (1 - np.cos(angle_diff))
            return new_dir, curr_tri, normal_sign*n
        
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
            angle_diff = half_angle-angle
            axis = n
            edge = mesh.positions[v1]-mesh.positions[vertex_id]
            new_dir = edge * np.cos(angle_diff) + cross(axis, edge) * np.sin(angle_diff) + axis * dot(axis, edge) * (1 - np.cos(angle_diff))
            return new_dir, next_tri, normal_sign*n
        
        curr_tri = next_tri
        v1 = v2
        angle += tri_angle

    return np.zeros(3), 0, curr_normal


def bary_to_uv(bary:np.ndarray) -> np.ndarray:
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
    # Project the direction onto the triangle plane
    current_normal = mesh.triangle_normals[start.face]    
    dir = project_vec(dir, current_normal)

    len_path = 0.0
    path_len = length(dir)

    geodesic = GeodesicPath()
    geodesic.start = start
    geodesic.path.append(start.interpolate(mesh))
    geodesic.dirs.append(normalize(dir))
    geodesic.normals.append(current_normal)
    
    next_bary = np.zeros(3)
    curr_bary = np.array([1 - start.uv[0] - start.uv[1], start.uv[0], start.uv[1]])
    curr_pos = start.interpolate(mesh)
    next_pos = np.zeros(3)
    curr_point = start
    tid_normal = np.zeros(3)
    curr_tri = start.face
    next_tri = -1
    
    while len_path < path_len-EPS:
        # Get the triangle normal
        tid_normal = mesh.triangle_normals[curr_tri]
        
        # Project the direction onto the triangle plane
        proj_dir = project_vec(dir, tid_normal)            
        if length(proj_dir) < EPS:
            # Direction is perpendicular to the triangle, cannot proceed
            break
        
        proj_dir = normalize(proj_dir)
        
        # Trace the ray in the current triangle
        next_pos, next_bary = trace_in_triangles(mesh, proj_dir, curr_point, curr_tri, path_len-len_path)
        
        # Check if the point is on an edge or vertex
        is_edge_bary, edge_idx = bary_is_edge(next_bary)
        is_vert_bary, vert_idx = bary_is_vert(next_bary)
        
        # Update the path
        len_path += length(next_pos - curr_pos)
        geodesic.path.append(next_pos.copy())
        
        # Determine the next triangle
        if is_vert_bary:
            # Point is on a vertex
            v_idx = mesh.triangles[curr_tri][vert_idx]
            
            # Transport the direction to the next triangle
            dir, next_tri, current_normal = compute_parallel_transport_vertex(mesh, curr_tri, v_idx, proj_dir, current_normal)
            geodesic.dirs.append(dir)
            geodesic.normals.append(current_normal)

            # Compute barycentric coordinates in the adjacent triangle
            p0_adj = mesh.positions[mesh.triangles[next_tri][0]]
            p1_adj = mesh.positions[mesh.triangles[next_tri][1]]
            p2_adj = mesh.positions[mesh.triangles[next_tri][2]]
            next_bary = np.zeros(3)
            dist = [(p0_adj-next_pos)@(p0_adj-next_pos), (p1_adj-next_pos)@(p1_adj-next_pos), (p2_adj-next_pos)@(p2_adj-next_pos)]
            next_bary[np.argmin(dist)] = 1
                        
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
            next_bary = tri_bary_coords(p0_adj, p1_adj, p2_adj, next_pos)
            
            if adj_tri == -1:
                curr_point = MeshPoint(adj_tri, bary_to_uv(next_bary))
                geodesic.dirs.append(dir)
                break
            
            # Get the common edge vertices
            e,_,_ = common_edge(mesh.triangles, curr_tri, adj_tri)
            
            if len(e)==0:
                # No common edge found
                curr_point = MeshPoint(adj_tri, bary_to_uv(next_bary))
                break
            
            # Transport the direction to the adjacent triangle
            dir, current_normal = compute_parallel_transport_edge(mesh, curr_tri, adj_tri, proj_dir, current_normal)
            geodesic.dirs.append(dir)
            geodesic.normals.append(current_normal)
                        
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
            dir = proj_dir
            geodesic.dirs.append(dir)
            geodesic.normals.append(current_normal)
    
    # Set the end point
    geodesic.end = curr_point
    
    return geodesic

