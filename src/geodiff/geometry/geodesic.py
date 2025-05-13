import numpy as np
import torch
from typing import Tuple, List, Optional
from multiprocessing import Pool, cpu_count
import copy

from geodiff.geometry.geodesic_utils import straightest_geodesic, GeodesicPath, triangle_normal
from geodiff.geometry.mesh import MeshPoint, Mesh
from geodiff.constants import EPS


def get_triangle_normal(mesh: Mesh, face_id: int) -> np.ndarray:
    """
    Get the normal vector of a triangle in the mesh.
    """
    p0 = mesh.positions[mesh.triangles[face_id][0]]
    p1 = mesh.positions[mesh.triangles[face_id][1]]
    p2 = mesh.positions[mesh.triangles[face_id][2]]
    return triangle_normal(p0, p1, p2)


def project_to_plane(dir, normal) -> torch.Tensor:
    return dir - normal * torch.dot(dir, normal)


def get_tensor_from_path(mesh: Mesh, path: GeodesicPath, start: MeshPoint, dir: torch.Tensor) -> torch.Tensor:
    """
    Convert a GeodesicPath endpoint to a tensor representation.
    """
    start_dir = path.dirs[0]
    start_dir = start_dir / np.linalg.norm(start_dir)
    start_normal = path.normals[0]

    end_point = path.end
    end_dir = path.dirs[-1]
    end_dir = end_dir / np.linalg.norm(end_dir)
    end_normal = path.normals[-1]

    start_cross = np.cross(start_dir, start_normal)
    end_cross = np.cross(end_dir, end_normal)

    # create matrix from orthonormal basis
    start_M = np.array([start_dir, start_normal, start_cross])
    end_M = np.array([end_dir, end_normal, end_cross])

    # rotation matrix from basis to basis
    R = start_M.T @ end_M

    R = torch.tensor(R, dtype=torch.float64)
    fixed_start = start.interpolate(mesh, tensor=True).detach()
    start_point = start.interpolate(mesh, tensor=True)

    target_end = (start.interpolate(mesh,tensor=True).detach()+project_to_plane(dir.detach(),torch.tensor(start_normal))) @ R
    T = end_point.interpolate(mesh,tensor=True).detach() - target_end

    tensor_point = (project_to_plane(start_point-fixed_start+dir, torch.tensor(start_normal))+fixed_start) @ R + T

    return tensor_point


def tri_bary_coords(p0, p1, p2, p) -> np.ndarray:
    """Compute barycentric coordinates of p in the triangle (p0, p1, p2)."""
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0
    
    d00 = torch.dot(v0, v0)
    d01 = torch.dot(v0, v1)
    d11 = torch.dot(v1, v1)
    d20 = torch.dot(v2, v0)
    d21 = torch.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < EPS:
        return np.array([1.0, 0.0, 0.0])
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    assert u > -EPS and v > -EPS and w > -EPS and u + v + w < 1+3*EPS, f"Invalid barycentric coordinates: {u}, {v}, {w}"
  
    return torch.tensor([u, v, w])


def get_meshpoint_3d(mesh:Mesh, face:int, point:torch.Tensor) -> MeshPoint:
    triangle = mesh.triangles[face]
    p0 = torch.tensor(mesh.positions[triangle[0]])
    p1 = torch.tensor(mesh.positions[triangle[1]])
    p2 = torch.tensor(mesh.positions[triangle[2]])
    bary = tri_bary_coords(p0,p1,p2,point)
    return MeshPoint(face, bary[1:])
    

# TODO toggle gradient
def trace_geodesics(mesh: Mesh, starts: List[MeshPoint], dirs: List[torch.tensor], gradient:bool=True) -> Tuple[List[MeshPoint], torch.Tensor]: 
    """
    Computes the geodesic path using finite differences for gradient computation.
    """
    paths:List[GeodesicPath] = []
    for start, dir in zip(starts, dirs):
        path = straightest_geodesic(mesh, start.detach(), dir.detach().numpy())
        paths.append(path)
    
    mesh_points = []
    tensor_points = torch.zeros(len(paths), 3, dtype=torch.float64)
    for k, (path, dir, start) in enumerate(zip(paths, dirs, starts)):
        tensor_points[k] = get_tensor_from_path(mesh, path, start, dir)
        mesh_points.append(get_meshpoint_3d(mesh, path.end.face, tensor_points[k]))

    return mesh_points, tensor_points
    