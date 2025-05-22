#include <vector>
#include <array>
#include <cmath>
#include <tuple>

const double EPS = 1e-6;

struct Vec3 {
    double x, y, z;
    Vec3 operator-(const Vec3& rhs) const { return {x - rhs.x, y - rhs.y, z - rhs.z}; }
    Vec3 operator+(const Vec3& rhs) const { return {x + rhs.x, y + rhs.y, z + rhs.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    double dot(const Vec3& rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }
    double norm() const { return std::sqrt(x*x + y*y + z*z); }
};

Vec3 normalize(const Vec3& v) {
    double n = v.norm();
    return {v.x / n, v.y / n, v.z / n};
}

Vec3 project_vec(const Vec3& v, const Vec3& normal) {
    return v - normal * (v.dot(normal));
}

// Replace with real mesh structures
struct Mesh {
    std::vector<std::array<int, 3>> triangles;
    std::vector<Vec3> triangle_normals;
    std::vector<Vec3> positions;
    std::vector<std::array<int, 3>> adjacencies;
};

struct MeshPoint {
    int face;
    std::array<double, 2> uv;

    Vec3 interpolate(const Mesh& mesh) const {
        auto tri = mesh.triangles[face];
        double u = uv[0], v = uv[1];
        double w = 1.0 - u - v;
        return mesh.positions[tri[0]] * w + mesh.positions[tri[1]] * u + mesh.positions[tri[2]] * v;
    }
};

std::tuple<MeshPoint, Vec3, Vec3> straightest_geodesic(const Mesh& mesh, const MeshPoint& start, Vec3 dir) {
    Vec3 current_normal = mesh.triangle_normals[start.face];
    dir = project_vec(dir, current_normal);

    double len_path = 0.0;
    double path_len = dir.norm();

    std::vector<Vec3> path = { start.interpolate(mesh) };
    std::vector<Vec3> dirs = { normalize(dir) };
    std::vector<Vec3> normals = { current_normal };

    std::array<double, 3> next_bary = {0, 0, 0};
    std::array<double, 3> curr_bary = {1 - start.uv[0] - start.uv[1], start.uv[0], start.uv[1]};
    Vec3 curr_pos = start.interpolate(mesh);
    Vec3 next_pos = {0, 0, 0};
    MeshPoint curr_point = start;
    Vec3 tid_normal = {0, 0, 0};
    int curr_tri = start.face;
    int next_tri = -1;

    while (len_path < path_len - EPS) {
        tid_normal = mesh.triangle_normals[curr_tri];
        Vec3 proj_dir = project_vec(dir, tid_normal);

        if (proj_dir.norm() < EPS) break;
        proj_dir = normalize(proj_dir);

        std::tie(next_pos, next_bary) = trace_in_triangles(mesh, proj_dir, curr_point, curr_tri, path_len - len_path);

        bool is_edge_bary;
        int edge_idx;
        std::tie(is_edge_bary, edge_idx) = bary_is_edge(next_bary);

        bool is_vert_bary;
        int vert_idx;
        std::tie(is_vert_bary, vert_idx) = bary_is_vert(next_bary);

        len_path += (next_pos - curr_pos).norm();
        path.push_back(next_pos);

        if (is_vert_bary) {
            int v_idx = mesh.triangles[curr_tri][vert_idx];
            std::tie(dir, next_tri, current_normal) = compute_parallel_transport_vertex(mesh, curr_tri, v_idx, proj_dir, current_normal);
            dirs.push_back(dir);
            normals.push_back(current_normal);

            const auto& tri = mesh.triangles[next_tri];
            const Vec3& p0 = mesh.positions[tri[0]];
            const Vec3& p1 = mesh.positions[tri[1]];
            const Vec3& p2 = mesh.positions[tri[2]];
            std::array<double, 3> dist = {
                (p0 - next_pos).dot(p0 - next_pos),
                (p1 - next_pos).dot(p1 - next_pos),
                (p2 - next_pos).dot(p2 - next_pos)
            };
            next_bary = {0, 0, 0};
            next_bary[std::distance(dist.begin(), std::min_element(dist.begin(), dist.end()))] = 1;

            curr_tri = next_tri;
            curr_pos = next_pos;
            curr_bary = next_bary;
            curr_point = MeshPoint{curr_tri, bary_to_uv(curr_bary)};
        } else if (is_edge_bary) {
            int adj_tri = mesh.adjacencies[curr_tri][edge_idx];
            if (adj_tri == -1) adj_tri = curr_tri;

            const auto& tri = mesh.triangles[adj_tri];
            next_bary = tri_bary_coords(
                mesh.positions[tri[0]],
                mesh.positions[tri[1]],
                mesh.positions[tri[2]],
                next_pos
            );

            if (adj_tri == -1) {
                curr_point = MeshPoint{adj_tri, bary_to_uv(next_bary)};
                dirs.push_back(dir);
                break;
            }

            auto [e, _, __] = common_edge(mesh.triangles, curr_tri, adj_tri);
            if (e.empty()) {
                curr_point = MeshPoint{adj_tri, bary_to_uv(next_bary)};
                break;
            }

            std::tie(dir, current_normal) = compute_parallel_transport_edge(mesh, curr_tri, adj_tri, proj_dir, current_normal);
            dirs.push_back(dir);
            normals.push_back(current_normal);

            curr_tri = adj_tri;
            curr_pos = next_pos;
            curr_bary = next_bary;
            curr_point = MeshPoint{curr_tri, bary_to_uv(curr_bary)};
        } else {
            curr_tri = curr_tri;
            curr_pos = next_pos;
            curr_bary = next_bary;
            curr_point = MeshPoint{curr_tri, bary_to_uv(curr_bary)};
            dir = proj_dir;
            dirs.push_back(dir);
            normals.push_back(current_normal);
        }
    }

    return {curr_point, dir, current_normal};
}
