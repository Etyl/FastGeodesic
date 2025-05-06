import potpourri3d as pp3d
import numpy as np
import os
import matplotlib.pyplot as plt
import time


from geometry.trace_geodesic import GeodesicPath, straightest_geodesic
from geometry.mesh import MeshPoint, Mesh
from constants import DATA_DIR, EPS
from dataloader.mesh_loader import load_mesh_from_obj
from ui import visualize_mesh_and_path
from geometry.utils import length


def get_pp3d_geodesic(V, F, start: MeshPoint, direction: np.ndarray) -> GeodesicPath:
    tracer = pp3d.GeodesicTracer(V,F)
    pp3d_trace = tracer.trace_geodesic_from_face(start.face, start.get_barycentric_coords(), direction)
    pp3d_geodesic = GeodesicPath()
    pp3d_geodesic.path = pp3d_trace
    pp3d_geodesic.start = start
    return pp3d_geodesic


def test_trace_geodesic():
    start_points = []
    start_directions = []
    
    test_path = os.path.join(DATA_DIR, 'test_trace_geodesic.txt')
    assert os.path.exists(test_path), "Test file does not exist."

    with open(test_path, 'r') as f:
        n = int(f.readline().strip())
        
        for _ in range(n):
            line = f.readline().strip().split()
            face = int(line[0])
            u = float(line[1])
            v = float(line[2])
            start_points.append(MeshPoint(face, np.array([u, v]))) 
        
        for _ in range(n):
            line = f.readline().strip().split()
            dir = np.array([float(line[0]), float(line[1]), float(line[2])])
            start_directions.append(dir)

    mesh_path = os.path.join(DATA_DIR, 'cat_head.obj')
    assert os.path.exists(mesh_path), "Mesh file does not exist."
    mesh = load_mesh_from_obj(mesh_path)

    V,F = pp3d.read_mesh(mesh_path)

    t0 = time.perf_counter()
    for start_point, start_direction in zip(start_points, start_directions):   
        pp3d_geodesic = get_pp3d_geodesic(V, F, start_point, start_direction)
    t_pp3d = time.perf_counter() - t0    
    
    t0 = time.perf_counter()
    for start_point, start_direction in zip(start_points, start_directions):   
        path = straightest_geodesic(mesh, start_point, start_direction)
    t_our = time.perf_counter() - t0

    print(f"PP3D time: {t_pp3d:.4f}s")
    print(f"Our method time: {t_our:.4f}s")


if __name__ == "__main__":
    test_trace_geodesic()
