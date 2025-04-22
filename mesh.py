class Mesh:
    def __init__(self):
        self.vertices = []  # List of tuples: (x, y, z)
        self.faces = []     # List of tuples: (v1_idx, v2_idx, v3_idx)

    def add_vertex(self, x, y, z):
        """Add a vertex to the mesh."""
        self.vertices.append((x, y, z))
        return len(self.vertices) - 1  # Return index of the added vertex

    def add_face(self, v1, v2, v3):
        """Add a face to the mesh. v1, v2, and v3 are indices of the vertices."""
        if all(0 <= v < len(self.vertices) for v in (v1, v2, v3)):
            self.faces.append((v1, v2, v3))
        else:
            raise IndexError("Vertex index out of range")

    def get_vertex(self, index):
        """Get the coordinates of a vertex by its index."""
        return self.vertices[index]

    def get_face_vertices(self, face_index):
        """Return the coordinates of the three vertices making up a face."""
        v_indices = self.faces[face_index]
        return [self.vertices[i] for i in v_indices]

    def __repr__(self):
        return f"<Mesh with {len(self.vertices)} vertices and {len(self.faces)} faces>"



class SurfacePoint:
    def __init__(self, mesh, location_type, index, barycentric_coords=None):
        """
        :param mesh: The Mesh object this surface point belongs to.
        :param location_type: One of 'vertex', 'edge', or 'face'.
        :param index: Index of the vertex, edge (as tuple), or face, depending on type.
        :param barycentric_coords: Coordinates (u, v, w) if location is on a face or edge.
        """
        assert location_type in {'vertex', 'edge', 'face'}
        self.mesh = mesh
        self.location_type = location_type
        self.index = index
        self.barycentric_coords = barycentric_coords  # Needed for face or edge

    def get_position(self):
        """Return the 3D coordinates of the surface point."""
        if self.location_type == 'vertex':
            return self.mesh.get_vertex(self.index)

        elif self.location_type == 'edge':
            v1_idx, v2_idx = self.index
            v1 = self.mesh.get_vertex(v1_idx)
            v2 = self.mesh.get_vertex(v2_idx)
            t = self.barycentric_coords[0]  # Linear interpolation between v1 and v2
            return tuple((1 - t) * a + t * b for a, b in zip(v1, v2))

        elif self.location_type == 'face':
            v1_idx, v2_idx, v3_idx = self.mesh.faces[self.index]
            u, v, w = self.barycentric_coords
            v1 = self.mesh.get_vertex(v1_idx)
            v2 = self.mesh.get_vertex(v2_idx)
            v3 = self.mesh.get_vertex(v3_idx)
            return tuple(
                u * a + v * b + w * c
                for a, b, c in zip(v1, v2, v3)
            )

    def __repr__(self):
        return f"<SurfacePoint on {self.location_type} {self.index}>"
