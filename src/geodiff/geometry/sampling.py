import numpy as np
import torch
from typing import List

from geodiff.geometry.mesh import Mesh, MeshPoint
from geodiff.geometry.utils import area_triangle


def uniform_sampling(mesh: Mesh, n_points: int, tensor=False) -> List[MeshPoint]:
    """
    Sample points uniformly from the mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh to sample from.
    n_points : int
        The number of points to sample.

    Returns
    -------
    List[MeshPoint]
        The sampled points.
    """
    weights = np.zeros(len(mesh.triangles))
    for triangle in range(len(mesh.triangles)):
        weights[triangle] += area_triangle(
            mesh.positions[mesh.triangles[triangle][0]],
            mesh.positions[mesh.triangles[triangle][1]],
            mesh.positions[mesh.triangles[triangle][2]]
        )
    
    samples = []
    weights = weights / weights.sum()

    for k in range(n_points):
        triangle_id = np.random.choice(len(mesh.triangles), p=weights)
        if tensor:
            barycentric_coords = torch.zeros(2, dtype=torch.float64)
        else:
            barycentric_coords = np.zeros(2, dtype=np.float64)
        barycentric_coords[0] = np.random.random()
        barycentric_coords[1] = (1-barycentric_coords[0])*np.random.random()
        samples.append(MeshPoint(triangle_id, barycentric_coords))

    return samples