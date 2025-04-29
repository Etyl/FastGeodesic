import numpy as np

from mesh import Mesh, MeshPoint


def area_triangle(p0,p1,p2):
    v1 = p1 - p0
    v2 = p2 - p0
    
    # Calculate the cross product of the two vectors
    cross_product = np.cross(v1, v2)
    
    # The magnitude of the cross product is twice the area of the triangle
    return np.linalg.norm(cross_product) / 2


def uniform_sampling(mesh: Mesh, n_points: int):
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
    np.ndarray
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
        barycentric_coords = np.zeros(2)
        barycentric_coords[0] = np.random.random()
        barycentric_coords[1] = (1-barycentric_coords[0])*np.random.random()
        samples.append(MeshPoint(triangle_id, barycentric_coords))

    return samples