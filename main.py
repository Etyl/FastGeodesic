import numpy as np
import time

from trace_geodesic import straightest_geodesic
from mesh import MeshPoint
from mesh_loader import create_tetrahedron, load_mesh_from_obj
from ui import visualize_mesh_and_path
from sampling import uniform_sampling


def main():
    # mesh = create_tetrahedron()
    mesh = load_mesh_from_obj("./data/cat_head.obj")

    n_points = 100

    samples = uniform_sampling(mesh, n_points)
    directions = [
        np.random.normal(size=3) for _ in range(n_points)
    ]

    print("Tracing Geodesics...")
    paths = []
    t0 = time.perf_counter()

    for start,dir in zip(samples,directions):
        paths.append(straightest_geodesic(mesh, start, dir))

    t = time.perf_counter() - t0
    print("Time elapsed:")
    print(f"total time: {t}")
    print(f"average: {t/n_points}")

    visualize_mesh_and_path(mesh, paths)

if __name__ == "__main__":
    main()