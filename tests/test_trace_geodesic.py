import potpourri3d as pp3d
import numpy as np
import os
import matplotlib.pyplot as plt


from geodiff.geometry.geodesic_utils import GeodesicPath, straightest_geodesic
from geodiff.geometry.mesh import MeshPoint, Mesh
from geodiff.constants import DATA_DIR, EPS
from geodiff.dataloader.mesh_loader import load_mesh_from_file
from geodiff.geometry.utils import length


def get_pp3d_geodesic(mesh_path:str, start: MeshPoint, direction: np.ndarray) -> GeodesicPath:
    V,F = pp3d.read_mesh(mesh_path)
    tracer = pp3d.GeodesicTracer(V,F)
    pp3d_trace = tracer.trace_geodesic_from_face(start.face, start.get_barycentric_coords(), direction)
    pp3d_geodesic = GeodesicPath(
        start = start,
        end = MeshPoint(),
        path = pp3d_trace,
    )
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
    mesh = load_mesh_from_file(mesh_path)

    total_path = []
    for start_point, start_direction in zip(start_points, start_directions):   
        pp3d_geodesic = get_pp3d_geodesic(mesh_path, start_point, start_direction)
        path = straightest_geodesic(mesh, start_point, start_direction)

        total_path.append(path)
        total_path.append(pp3d_geodesic)

        assert len(path.path) == len(pp3d_geodesic.path), "Path lengths do not match."
        for i in range(len(path.path)):
            assert length(path.path[i]-pp3d_geodesic.path[i]) < EPS, f"Path points do not match at index {i}."


