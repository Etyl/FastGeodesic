import torch
import numpy as np

from diff_geodesic import diff_straighest_geodesic, get_triangle_normal
from mesh_loader import load_mesh_from_obj
from mesh import MeshPoint, Mesh
from ui import visualize_mesh_and_path
from trace_geodesic import GeodesicPath


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
    def __init__(self, mesh: Mesh):
        super(DirNN, self).__init__()
        self.mesh = mesh
        self.layer = torch.nn.Linear(6,3)

    def forward(self, mesh_point:MeshPoint):
        normal = torch.tensor(get_triangle_normal(self.mesh, mesh_point.face),dtype=torch.float32)
        x = torch.cat([mesh_point.interpolate(self.mesh, tensor=True), normal])
        dir = self.layer(x)
        geodesic, new_point = diff_straighest_geodesic(self.mesh, mesh_point, dir)
        return new_point, point_to_uv(self.mesh, new_point, geodesic.end.face)
    
def score(x) -> torch.Tensor:
    return 1+x[1]


def main():
    mesh = load_mesh_from_obj("./data/cat_head.obj")

    dir_nn = DirNN(mesh)

    optimizer = torch.optim.SGD(dir_nn.parameters(), lr=0.001)
    geodesic = GeodesicPath()
    geodesic.start = MeshPoint(face=0, uv=torch.tensor([0.2, 0.2], dtype=torch.float32))
    geodesic.path = [geodesic.start.interpolate(mesh)]

    for i in range(1000):
        mesh_point = MeshPoint(face=0, uv=torch.tensor([0.2, 0.2], dtype=torch.float32))
        point = mesh_point.interpolate(mesh)
        for _ in range(1):
            point, mesh_point = dir_nn(mesh_point.detach())

            score_value = score(point)
            
            # backpropagate the score to the point
            optimizer.zero_grad()
            (-score_value).backward()
            optimizer.step()
        geodesic.path.append(point.detach().numpy())
 
        print(f"Iteration {i}: Score = {score_value.item()}")        
    geodesic.end = mesh_point
    visualize_mesh_and_path(mesh, [geodesic])


if __name__ == "__main__":
    main()