import numpy as np

from trace_geodesic import straightest_geodesic
from mesh import MeshPoint
from mesh_loader import create_tetrahedron, load_mesh_from_obj
from ui import visualize_mesh_and_path


def main():
    # mesh = create_tetrahedron()
    mesh = load_mesh_from_obj("./data/cat_head.obj")
    
    # Define a starting point on a face (barycentric coordinates on face 0)
    start = MeshPoint(0, np.array([0.5, 0.5]))  # On face 0 with UV coordinates (0.3, 0.3)
    
    # Define a direction
    direction = 4*np.array([-1, 0, 0])  # Direction
    
    print("Computing geodesic path...")
    
    # Compute the geodesic path
    path = straightest_geodesic(mesh, start, direction)
    
    # Visualize the mesh and path
    print("\nVisualizing mesh and path...")
    visualize_mesh_and_path(mesh, path)

if __name__ == "__main__":
    main()