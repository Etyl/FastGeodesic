import numpy as np
import torch

from trace_geodesic import straightest_geodesic, GeodesicPath, triangle_normal
from mesh import MeshPoint, Mesh
from mesh_loader import load_mesh_from_obj
from ui import visualize_mesh_and_path


def get_triangle_normal(mesh: Mesh, face_id: int) -> np.ndarray:
    """
    Get the normal vector of a triangle in the mesh.
    """
    p0 = mesh.positions[mesh.triangles[face_id][0]]
    p1 = mesh.positions[mesh.triangles[face_id][1]]
    p2 = mesh.positions[mesh.triangles[face_id][2]]
    return triangle_normal(p0, p1, p2)

def diff_straighest_geodesic(mesh: Mesh, start: MeshPoint, dir: torch.tensor):
    """
    Compute the geodesic path using finite differences for gradient computation.
    """
    start_dir = dir.detach().numpy()
    start_normal = get_triangle_normal(mesh, start.face)
    
    path: GeodesicPath = straightest_geodesic(mesh, start, start_dir)

    start_dir = start_dir / np.linalg.norm(start_dir)

    end_point = path.end
    end_dir = path.dirs[-1]
    end_dir = end_dir / np.linalg.norm(end_dir)
    end_normal = get_triangle_normal(mesh, end_point.face)

    start_cross = np.cross(start_dir, start_normal)
    end_cross = np.cross(end_dir, end_normal)

    # create matrix from orthonormal basis
    start_M = np.array([start_dir, start_normal, start_cross])
    end_M = np.array([end_dir, end_normal, end_cross])

    # rortation matrix from basis to basis
    R = end_M @ start_M.T

    target_end = R@(start.interpolate(mesh)+dir.detach().numpy())
    T = end_point.interpolate(mesh) - target_end

    R = torch.tensor(R, dtype=torch.float32)
    T = torch.tensor(T, dtype=torch.float32)

    return path, R @ (start.interpolate(mesh, tensor=True) + dir) + T    


if __name__ == "__main__":
    mesh = load_mesh_from_obj("./data/cat_head.obj")
    start = MeshPoint(0, np.array([0.3, 0.2]))
    dir = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    path,_ = diff_straighest_geodesic(mesh, start, dir)
    visualize_mesh_and_path(mesh, [path])
