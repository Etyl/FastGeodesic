import torch
import numpy as np
import os
from typing import List
import tqdm
import matplotlib.pyplot as plt

from fastgeodesic.geometry.geodesic import trace_geodesics, get_triangle_normal
from fastgeodesic.dataloader.mesh_loader import load_mesh_from_obj
from fastgeodesic.geometry.mesh import MeshPoint, Mesh
from ui import plot_loss, visualize_mesh_and_points
from fastgeodesic.constants import DATA_DIR
from fastgeodesic.geometry.sampling import uniform_sampling


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class DirNN(torch.nn.Module):
    def __init__(self, mesh: Mesh, hidden_dim=128):
        super(DirNN, self).__init__()
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
        new_mesh_points, new_points = trace_geodesics(self.mesh, mesh_points, dirs)
        return new_points, new_mesh_points


def score(x) -> torch.Tensor:
    return torch.mean(x[:,1])


def main(n_points=100, epochs=50, iterations=3):
    # mesh = create_tetrahedron()
    mesh = load_mesh_from_obj(os.path.join(DATA_DIR, "cat_head.obj"))
    dir_nn = DirNN(mesh)

    optimizer = torch.optim.Adam(dir_nn.parameters(), lr=0.01)
    
    start_mesh_points = uniform_sampling(mesh, n_points=n_points, tensor=True)
    total_loss = []

    progress_bar = tqdm.tqdm(range(epochs), desc="Training")

    for _ in progress_bar:
        mesh_points = [mesh_point.detach() for mesh_point in start_mesh_points]
        
        for _ in range(iterations):
            points, mesh_points = dir_nn(mesh_points)

        score_value = score(points)
        
        # backpropagate the score to the point
        optimizer.zero_grad()
        (-score_value).backward()
        optimizer.step()

        progress_bar.set_postfix({"Score": score_value.item()})
        total_loss.append(-score_value.item())
     
    # Visualize the mesh and points
    with torch.no_grad():
        mesh_points = [mesh_point.detach() for mesh_point in start_mesh_points]
            
        for k in range(iterations):
            points = np.array([mesh_point.interpolate(mesh) for mesh_point in mesh_points])
            visualize_mesh_and_points(mesh, points, f"Points iteration {k}")
            points, mesh_points = dir_nn(mesh_points)
             
        end_points = points.detach().numpy()
        visualize_mesh_and_points(mesh, end_points, "End Points")
    
    plot_loss(total_loss)

    plt.show()

if __name__ == "__main__":
    set_seed(102)
    main()