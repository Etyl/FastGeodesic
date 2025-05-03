import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List

from geometry.mesh import MeshPoint, Mesh
from geometry.trace_geodesic import GeodesicPath


def plot_path(path: GeodesicPath, mesh: Mesh, ax, arrow_scale=0.2):
    path_points = np.array(path.path)
    ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 'o-', 
            color='magenta', linewidth=2, markersize=5)
    
    # 4. Plot direction vectors at each path point
    if hasattr(path, 'dirs') and len(path.dirs) > 0:
        dirs = np.array(path.dirs)
        
        # Find an appropriate scale for the arrows
        if len(path_points) > 1:
            # Calculate average segment length
            segments = path_points[1:] - path_points[:-1]
            avg_segment_length = np.mean(np.sqrt(np.sum(segments**2, axis=1)))
            arrow_length = avg_segment_length * arrow_scale
        else:
            # Default if only one point
            arrow_length = 0.1
        
        # Make all direction vectors the same length for clearer visualization
        normalized_dirs = np.array([dir / np.linalg.norm(dir) * arrow_length for dir in dirs])
        
        # Draw arrows for each point
        for i in range(len(path_points)):
            if i < len(normalized_dirs):  # Ensure we have a direction vector
                start = path_points[i]
                end = start + normalized_dirs[i]
                
                # Draw arrow
                ax.quiver(
                    start[0], start[1], start[2],           # starting point
                    normalized_dirs[i][0], normalized_dirs[i][1], normalized_dirs[i][2],  # direction
                    color='orange', alpha=1.0, arrow_length_ratio=0.3,
                    linewidth=2
                )
    
    # 5. Add start and end points with different markers
    start_pos = path.start.interpolate(mesh)
    end_pos = path.end.interpolate(mesh)
    
    ax.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]], color='lime', s=200, marker='*')
    ax.scatter([end_pos[0]], [end_pos[1]], [end_pos[2]], color='red', s=200, marker='X')
    


def visualize_mesh_and_path(mesh:Mesh, paths:List[GeodesicPath], arrow_scale=0.2):
    """
    Visualize the mesh and geodesic path in 3D, including direction vectors at each point.
    
    Args:
        mesh: The mesh data
        path: The geodesic path
        arrow_scale: Scale factor for the direction arrows
    """    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot the mesh
    faces = []
    colors = []
    
    # Different colors for each face
    face_colors = [
        (0.8, 0.1, 0.1, 0.5),  # Red
        (0.1, 0.8, 0.1, 0.5),  # Green
        (0.1, 0.1, 0.8, 0.5),  # Blue
        (0.8, 0.8, 0.1, 0.5)   # Yellow
    ]
    
    for i, triangle in enumerate(mesh.triangles):
        # Get vertices
        vertices = [mesh.positions[idx] for idx in triangle]
        faces.append(vertices)
        colors.append((*face_colors[i % len(face_colors)][:3], 0.3))  # More transparent for other faces
    
    # Create 3D polygons
    poly = Poly3DCollection(faces, linewidths=0.3, edgecolors='black', alpha=0.5)
    ax.add_collection3d(poly)
    
    # 2. Plot vertices
    vertices = np.array(mesh.positions)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='black', s=10, label='Vertices')
    
    # 3. Plot the path
    for path in paths:
        plot_path(path, mesh, ax, arrow_scale)
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mesh with Geodesic Path and Direction Vectors')
    
    # Add legend
    ax.legend()
    
    # Set aspect ratio to be equal
    max_range = np.max([
        np.max(vertices[:, 0]) - np.min(vertices[:, 0]),
        np.max(vertices[:, 1]) - np.min(vertices[:, 1]),
        np.max(vertices[:, 2]) - np.min(vertices[:, 2])
    ])
    
    mid_x = (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) * 0.5
    mid_y = (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) * 0.5
    mid_z = (np.max(vertices[:, 2]) + np.min(vertices[:, 2])) * 0.5
    
    ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
    ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
    ax.set_zlim(mid_z - max_range * 0.6, mid_z + max_range * 0.6)
    
    plt.tight_layout()
    plt.show()


def visualize_mesh_and_points(mesh: Mesh, starts: np.ndarray, ends: np.ndarray):
    """Visualize the mesh and start and end points in 3D."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot the mesh
    faces = []
    colors = []
    
    # Different colors for each face
    face_colors = [
        (0.8, 0.1, 0.1, 0.5),  # Red
        (0.1, 0.8, 0.1, 0.5),  # Green
        (0.1, 0.1, 0.8, 0.5),  # Blue
        (0.8, 0.8, 0.1, 0.5)   # Yellow
    ]
    
    for i, triangle in enumerate(mesh.triangles):
        # Get vertices
        vertices = [mesh.positions[idx] for idx in triangle]
        faces.append(vertices)
        colors.append((*face_colors[i % len(face_colors)][:3], 0.3))  # More transparent for other faces
    
    # Create 3D polygons
    poly = Poly3DCollection(faces, linewidths=0.3, edgecolors='black', alpha=0.5)
    ax.add_collection3d(poly)
    
    # 2. Plot vertices
    vertices = np.array(mesh.positions)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='black', s=10, label='Vertices')
    
    # Plot start and end points
    ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2], color='lime', s=200, marker='*', label='Start Points')
    ax.scatter(ends[:, 0], ends[:, 1], ends[:, 2], color='red', s=200, marker='X', label='End Points')
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mesh with Start and End Points')
    
    # Add legend
    ax.legend()
    
    # Set aspect ratio to be equal
    max_range = np.max([
        np.max(vertices[:, 0]) - np.min(vertices[:, 0]),
        np.max(vertices[:, 1]) - np.min(vertices[:, 1]),
        np.max(vertices[:, 2]) - np.min(vertices[:, 2])
    ])

    mid_x = (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) * 0.5
    mid_y = (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) * 0.5
    mid_z = (np.max(vertices[:, 2]) + np.min(vertices[:, 2])) * 0.5

    ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
    ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
    ax.set_zlim(mid_z - max_range * 0.6, mid_z + max_range * 0.6)

    plt.tight_layout()
    plt.show()