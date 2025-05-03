import numpy as np
import time
import os
from tqdm import tqdm

from geometry.trace_geodesic import straightest_geodesic
from dataloader.mesh_loader import load_mesh_from_obj
from ui import visualize_mesh_and_path
from geometry.sampling import uniform_sampling
from constants import DATA_DIR

os.environ['PYTHONUNBUFFERED'] = '1'

def main(n_points=10000, visualize=False):
    mesh = load_mesh_from_obj(os.path.join(DATA_DIR, "cat_head.obj"))

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
        if visualize:
            paths.append(path)

    t = time.perf_counter() - t0
    print(f"Total time elapsed: {t}")
    print(f"Average per trace: {t/n_points}")

    if visualize:
        visualize_mesh_and_path(mesh, paths)

if __name__ == "__main__":
    main()