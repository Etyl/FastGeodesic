import os
import numpy as np

from fastgeodesic.dataloader.mesh_loader import load_mesh_from_obj
from fastgeodesic.constants import DATA_DIR

def write_test():
    mesh = load_mesh_from_obj(os.path.join(DATA_DIR, "cat_head.obj"))
    with open(os.path.join(DATA_DIR, 'test_trace_geodesic.txt'), 'w') as f:
        f.write(f"{50}\n")
        for _ in range(50):
            u = np.random.random()
            v = (1-u)*np.random.random()
            f.write(f"{np.random.randint(0, mesh.triangles.shape[0])} {u} {v}\n")

        for _ in range(50):
            dir = np.random.random(3)
            f.write(f"{dir[0]} {dir[1]} {dir[2]}\n")
  
if __name__ == "__main__":
    write_test()