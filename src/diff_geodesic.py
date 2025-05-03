import numpy as np
import torch
from typing import Tuple, List
from multiprocessing import Pool, cpu_count
import copy

from geometry.trace_geodesic import straightest_geodesic, GeodesicPath, triangle_normal
from geometry.mesh import MeshPoint, Mesh


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


def diff_straighest_geodesic(mesh: Mesh, start: MeshPoint, dir: torch.Tensor) -> Tuple[GeodesicPath, torch.Tensor]: 
    """
    Compute the geodesic path using finite differences for gradient computation.
    """
    path: GeodesicPath = straightest_geodesic(mesh, start.detach(), dir.detach().numpy())
    tensor_point = get_tensor_from_path(mesh, path, start, dir)
    
    return path, tensor_point   


def batch_straightest_geodesic(mesh: Mesh, starts: List[MeshPoint], dirs: List[torch.tensor]) -> Tuple[List[GeodesicPath], List[torch.Tensor]]:
    """
    Compute the geodesic path using finite differences for gradient computation.
    """
    paths = []

    for k, (start, dir) in enumerate(zip(starts, dirs)):
        path = straightest_geodesic(mesh, start, dir)
        paths.append(path)

    return paths

def batch_diff_straighest_geodesic(mesh: Mesh, starts: List[MeshPoint], dirs: List[torch.tensor], cpus = None) -> Tuple[List[GeodesicPath], List[torch.Tensor]]: 
    """
    Compute the geodesic path using finite differences for gradient computation, uses multiprocessing.
    """

    if cpus is None:
        paths = []
        for start, dir in zip(starts, dirs):
            path = straightest_geodesic(mesh, start.detach(), dir.detach().numpy())
            paths.append(path)
    else: 
        cpus = min(cpus, len(starts))

        # Split the start and dir lists into chunks for multiprocessing
        chunk_size = len(starts) // cpus
        dir_chunks = [
            [dir.detach().numpy() for dir in dirs[i:min(len(dirs),i+chunk_size)]] 
            for i in range(0, len(dirs), chunk_size)
        ]
        start_chunks = [
            [start.detach() for start in starts[i:min(len(starts),i+chunk_size)]] 
            for i in range(0, len(starts), chunk_size)
        ]

        # Create a pool of workers
        with Pool(cpus) as pool:
            total_paths = pool.starmap(batch_straightest_geodesic, [(mesh, start_chunk, dir_chunk) for start_chunk, dir_chunk in zip(start_chunks, dir_chunks)])
        paths = [item for sublist in total_paths for item in sublist]
    
    tensor_points = torch.zeros(len(paths), 3, dtype=torch.float64)
    for k, (path, dir, start) in enumerate(zip(paths, dirs, starts)):
        tensor_points[k] = get_tensor_from_path(mesh, path, start, dir)

    return paths, tensor_points
    