implementing trace_geodesic;

struct Mesh {
    StructuredBuffer<double3> positions;
    StructuredBuffer<int3> triangles;
    StructuredBuffer<int3> adjacencies;
    StructuredBuffer<double3> triangle_normals;
    StructuredBuffer<StructuredBuffer<int>> v2t;

    __init(
        StructuredBuffer<double3> positions,
        StructuredBuffer<int3> triangles,
        StructuredBuffer<int3> adjacencies,
        StructuredBuffer<double3> triangle_normals,
        StructuredBuffer<StructuredBuffer<int>> v2t,
    ) {
        this.positions = positions;
        this.triangles = triangles;
        this.adjacencies = adjacencies;
        this.triangle_normals = triangle_normals;
        this.v2t = v2t;
    }
}

struct MeshPoint {
    int face;
    double2 uv;

    __init(int face, double2 uv) {
        this.face = face;
        this.uv = uv;
    }

    double3 interpolate(Mesh mesh) {
        double3 p0 = mesh.positions[mesh.triangles[face][0]];
        double3 p1 = mesh.positions[mesh.triangles[face][1]];
        double3 p2 = mesh.positions[mesh.triangles[face][2]];
        double3 pos = (1 - this.uv[0] - this.uv[1]) * p0 + this.uv[0] * p1 + this.uv[1] * p2;
        return pos;
    }

    double3 get_barycentric_coords() {
        return { 1.0 - this.uv[0] - this.uv[1], this.uv[0], this.uv[1] };
    }
}