import numpy as np
import torch
from typing import Tuple, List
import os

from geometry.trace_geodesic import straightest_geodesic, GeodesicPath, triangle_normal
from geometry.mesh import MeshPoint, Mesh
from dataloader.mesh_loader import load_mesh_from_obj
from ui import visualize_mesh_and_path
from constants import DATA_DIR


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

def diff_straighest_geodesic(mesh: Mesh, start: MeshPoint, dir: torch.tensor) -> Tuple[GeodesicPath, torch.Tensor]: 
    """
    Compute the geodesic path using finite differences for gradient computation.
    """
    path: GeodesicPath = straightest_geodesic(mesh, start.detach(), dir.detach().numpy())

    start_dir = path.dirs[0]
    start_dir = start_dir / np.linalg.norm(start_dir)
    start_normal = get_triangle_normal(mesh, start.face)

    end_point = path.end
    end_dir = path.dirs[-1]
    end_dir = end_dir / np.linalg.norm(end_dir)
    end_normal = get_triangle_normal(mesh, end_point.face)

    start_cross = np.cross(start_dir, start_normal)
    end_cross = np.cross(end_dir, end_normal)

    # create matrix from orthonormal basis
    start_M = np.array([start_dir, start_normal, start_cross])
    end_M = np.array([end_dir, end_normal, end_cross])

    # rotation matrix from basis to basis
    R = start_M.T @ end_M

    R = torch.tensor(R, dtype=torch.float32)
    fixed_start = start.interpolate(mesh, tensor=True).detach()
    start_point = start.interpolate(mesh, tensor=True)

    target_end = (start.interpolate(mesh,tensor=True).detach()+project_to_plane(dir.detach(),torch.tensor(start_normal))) @ R
    T = end_point.interpolate(mesh,tensor=True).detach() - target_end

    tensor_point = (project_to_plane(start_point-fixed_start+dir, torch.tensor(start_normal))+fixed_start) @ R + T

    return path, tensor_point   

