import torch
import numpy as np
from torchviz import make_dot
import os
from typing import List, Tuple
import tqdm

from diff_geodesic import batch_diff_straighest_geodesic, get_triangle_normal
from dataloader.mesh_loader import load_mesh_from_obj, create_triangle, create_tetrahedron
from geometry.mesh import MeshPoint, Mesh
from ui import visualize_mesh_and_path, visualize_mesh_and_points
from geometry.trace_geodesic import GeodesicPath
from constants import DATA_DIR
from geometry.sampling import uniform_sampling


# TODO add unit tests (use pot pourri)

def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def tri_bary_coords(p0, p1, p2, p):
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
    if abs(denom) < 1e-6:
        return torch.tensor([1.0, 0.0, 0.0])
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return torch.stack([u, v, w]) # todo fix this

def bary_to_uv(bary):
    """Convert barycentric coordinates to UV coordinates."""
    return bary[1:]

def point_to_uv(mesh: Mesh, point: torch.Tensor, face:int) -> MeshPoint:
    """Convert a point in 3D space to UV coordinates on the mesh."""
    p0 = torch.tensor(mesh.positions[mesh.triangles[face][0]])
    p1 = torch.tensor(mesh.positions[mesh.triangles[face][1]])
    p2 = torch.tensor(mesh.positions[mesh.triangles[face][2]])
    
    bary = tri_bary_coords(p0, p1, p2, point)
    return MeshPoint(face, bary_to_uv(bary))

class DirNN(torch.nn.Module):
    def __init__(self, mesh: Mesh, hidden_dim=128, cpus=1):
        super(DirNN, self).__init__()
        self.cpus = cpus
        self.mesh = mesh
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(6,hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,3),
        ).type(torch.float64)

    def forward(self, mesh_points:List[MeshPoint]):
        mesh_points = [mesh_point.detach() for mesh_point in mesh_points]
        normals = torch.tensor(np.array([get_triangle_normal(self.mesh, mesh_point.face) for mesh_point in mesh_points]),dtype=torch.float64)
        points = torch.tensor(np.array([mesh_point.interpolate(self.mesh) for mesh_point in mesh_points]),dtype=torch.float64)
        x = torch.cat([points, normals], dim=1)
        dirs = self.mlp(x)
        geodesics, new_points = batch_diff_straighest_geodesic(self.mesh, mesh_points, dirs, cpus=self.cpus)
        new_mesh_points = [point_to_uv(self.mesh, new_point, geodesic.end.face) for geodesic, new_point in zip(geodesics, new_points)]
        return new_points, new_mesh_points


def score(x) -> torch.Tensor:
    return torch.mean(1+x[:,1])


def main(n_points=100, iterations=50, cpus=None):
    # mesh = create_tetrahedron()
    mesh = load_mesh_from_obj(os.path.join(DATA_DIR, "cat_head.obj"))
    dir_nn = DirNN(mesh, cpus=cpus)

    optimizer = torch.optim.Adam(dir_nn.parameters(), lr=0.01)

    start_mesh_points = uniform_sampling(mesh, n_points=n_points, tensor=True)

    progress_bar = tqdm.tqdm(range(iterations), desc="Training")

    for i in progress_bar:
        points, mesh_points = dir_nn(start_mesh_points)

        score_value = score(points)
        
        # backpropagate the score to the point
        optimizer.zero_grad()
        (-score_value).backward()
        optimizer.step()

        progress_bar.set_postfix({"Score": score_value.item()})
     
    start_points = np.array([mesh_point.interpolate(mesh) for mesh_point in start_mesh_points])
    end_points = points.detach().numpy()
    visualize_mesh_and_points(mesh, start_points, end_points)

if __name__ == "__main__":
    set_seed(102)
    main()