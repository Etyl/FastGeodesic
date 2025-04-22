import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mesh import Mesh, SurfacePoint
from geodesic import trace_straightest_geodesic

# Mesh and SurfacePoint classes go here...
# (reuse the Mesh, SurfacePoint, build_face_adjacency, compute_barycentric,
#  and trace_straightest_geodesic functions from earlier)

# === Step 1: Create a Tetrahedron Mesh ===
mesh = Mesh()
v0 = mesh.add_vertex(0, 0, 0)
v1 = mesh.add_vertex(1, 0, 0)
v2 = mesh.add_vertex(0, 1, 0)
v3 = mesh.add_vertex(0, 0, 1)

mesh.add_face(v0, v1, v2)  # base
mesh.add_face(v0, v1, v3)
mesh.add_face(v1, v2, v3)
mesh.add_face(v2, v0, v3)

# === Step 2: Create a SurfacePoint on the base triangle ===
start = SurfacePoint(mesh, 'face', 0, barycentric_coords=(1/3, 1/3, 1/3))

# === Step 3: Define a direction vector ===
direction = np.array([0.3, 0.3, 0])  # Arbitrary direction

# === Step 4: Trace the Geodesic ===
path = trace_straightest_geodesic(mesh, start, direction)

# === Step 5: Visualization ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw faces
for face in mesh.faces:
    tri = np.array([mesh.get_vertex(i) for i in face])
    tri = np.vstack([tri, tri[0]])  # close the triangle
    ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], color='gray', alpha=0.5)

# Draw geodesic path
path_points = np.array([p.get_position() for p in path])
ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color='red', linewidth=2, label='Geodesic')

# Draw vertices
verts = np.array(mesh.vertices)
ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color='black')

# Annotate starting point
ax.scatter(*start.get_position(), color='blue', s=100, label='Start')

ax.set_title("Tetrahedron with Geodesic")
ax.legend()
ax.set_box_aspect([1, 1, 1])


# plt.savefig("tetrahedron_geodesic.png")
plt.show()