import numpy as np
import torch
from typing import List

from geodiff.geometry.mesh import Mesh, MeshPoint


def uniform_sampling(mesh: Mesh, n_points: int, tensor: bool = False) -> List[MeshPoint]:
    """
    Sample points uniformly from the mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh to sample from.
    n_points : int
        The number of points to sample.
    tensor : bool
        Whether to use torch tensors for barycentric coordinates.

    Returns
    -------
    List[MeshPoint]
        The sampled points.
    """
    tri = mesh.triangles
    pos = mesh.positions
    
    p0 = pos[tri[:, 0]]
    p1 = pos[tri[:, 1]]
    p2 = pos[tri[:, 2]]

    # Compute triangle areas
    v0 = p1 - p0
    v1 = p2 - p0
    cross = np.cross(v0, v1)
    weights = 0.5 * np.linalg.norm(cross, axis=1)
    weights /= weights.sum()

    samples = []
    for _ in range(n_points):
        triangle_id = np.random.choice(len(tri), p=weights)

        # Generate random uniform barycentric coordinates
        u = np.random.random()
        v = (1 - u) * np.random.random()
        bc = torch.zeros(2, dtype=torch.float64) if tensor else np.zeros(2)
        bc[0], bc[1] = u, v

        samples.append(MeshPoint(triangle_id, bc))

    return samples
