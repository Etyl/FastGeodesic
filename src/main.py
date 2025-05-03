import numpy as np
import time
import os
from tqdm import tqdm

from geometry.trace_geodesic import straightest_geodesic
from geometry.mesh import MeshPoint, Mesh
from dataloader.mesh_loader import create_tetrahedron, load_mesh_from_obj
from ui import visualize_mesh_and_path
from geometry.sampling import uniform_sampling
from constants import DATA_DIR

os.environ['PYTHONUNBUFFERED'] = '1'

def main():
    # mesh = create_tetrahedron()
    mesh = load_mesh_from_obj(os.path.join(DATA_DIR, "cat_head.obj"))

    n_points = 10000

    samples = uniform_sampling(mesh, n_points)
    directions = [
        np.random.normal(size=3) for _ in range(n_points)
    ]

    paths = []
    t0 = time.perf_counter()

    bar = tqdm(range(n_points), desc="Geodesic Tracing")
    for idx in bar:
        start = samples[idx]
        dir = directions[idx]
        path = straightest_geodesic(mesh, start, dir)
        # paths.append(path)

    t = time.perf_counter() - t0
    print(f"Total time elapsed: {t}")
    print(f"Average per trace: {t/n_points}")

    # visualize_mesh_and_path(mesh, paths)

if __name__ == "__main__":
    main()