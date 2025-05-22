#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace geodesic_cpp {

#define EPS 1e-5f


__device__ __host__ inline float3 operator*(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ __host__ inline float3 operator*(float s, const float3& v) {
    return v * s;  // reuse the above definition
}

__device__ __host__ inline float3 operator/(const float3& v, float s) {
    return make_float3(v.x / s, v.y / s, v.z / s);
}


__device__ __host__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float get_component(const float3& v, int i) {
    if (i == 0) return v.x;
    else if (i == 1) return v.y;
    else /* i == 2 */ return v.z;
}

__device__ __host__ inline int get_component(const int3& v, int i) {
    if (i == 0) return v.x;
    else if (i == 1) return v.y;
    else /* i == 2 */ return v.z;
}

__device__ __host__ inline void set_float3_component(float3 &v, int idx, float value) {
    if (idx == 0) v.x = value;
    else if (idx == 1) v.y = value;
    else if (idx == 2) v.z = value;
}

__device__ float3 cross(const float3 a, const float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float dot(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float length(const float3 a) {
    return sqrtf(dot(a, a));
}

__device__ float3 normalize(const float3 a) {
    float len = length(a);
    if (len < EPS) return make_float3(0.0f, 0.0f, 0.0f);
    return make_float3(a.x / len, a.y / len, a.z / len);
}


__device__ float3 project_vec(const float3 v, const float3 normal) {
    float3 n = normalize(normal);
    float dp = dot(v, n);
    return make_float3(v.x - dp * n.x, v.y - dp * n.y, v.z - dp * n.z);
}


struct MeshPoint {
    int face;       // Triangle index in the mesh
    float2 uv;      // uv coordinates

    __device__ __host__
    MeshPoint(int face_id, float2 uv_coords) : face(face_id), uv(uv_coords) {}

    __device__ float3 interpolate(const Mesh& mesh) const {
        float3 p0 = mesh.positions[mesh.triangles[face*3]];
        float3 p1 = mesh.positions[mesh.triangles[face*3+1]];
        float3 p2 = mesh.positions[mesh.triangles[face*3+2]];
        return p0 * (1.0f-uv.x-uv.y) + p1 * uv.x + p2 * uv.y;
    }

    __device__ float3 get_barycentric_coords() const {
        return {
            1.0f - uv.x - uv.y,
            uv.x,
            uv.y
        };
    }
};

struct Mesh {
    const float3* positions;
    const int* triangles;
    const int* adjacencies;
    const float3* triangle_normals;
    const int* v2t;

    int num_vertices;
    int num_triangles;
    int max_v2t;
};


__device__ float3 triangle_normal(const float3 p0, const float3 p1, const float3 p2) {
    return normalize(cross(p1 - p0, p2 - p0));
}


__device__ float3 tri_bary_coords(
    const float3 p0, const float3 p1, const float3 p2, const float3 p
) {
    float3 v0 = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    float3 v1 = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
    float3 v2 = make_float3(p.x - p0.x, p.y - p0.y, p.z - p0.z);

    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);

    float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) < EPS) {
        return {1.0f,0.0f,0.0f};
    }

    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    return {u, v, w};
}

__device__ void bary_is_edge(const float3 bary, bool* is_edge, int* edge_idx_out) {
    if (fabsf(bary.x) < EPS) {
        *edge_idx_out = 1;
        *is_edge = true;
    }
    if (fabsf(bary.y) < EPS) {
        *edge_idx_out = 2;
        *is_edge = true;
    }
    if (fabsf(bary.z) < EPS) {
        *edge_idx_out = 0;
        *is_edge = true;
    }
    *edge_idx_out = -1;
    *is_edge = false;
}

__device__ void bary_is_vert(const float3 bary, bool* is_vert, int* vert_idx_out) {
    if (fabsf(bary.x - 1.0f) < EPS) {
        *vert_idx_out = 0;
        *is_vert = true;
    }
    if (fabsf(bary.y - 1.0f) < EPS) {
        *vert_idx_out = 1;
        *is_vert = true;
    }
    if (fabsf(bary.z - 1.0f) < EPS) {
        *vert_idx_out = 2;
        *is_vert = true;
    }
    *vert_idx_out = -1;
    *is_vert = false;
}


__device__ float2 closest_point_parameter_coplanar(
    const float3& P1, const float3& d1,
    const float3& P2, float3 d2
) {
    float d2_length = length(d2);
    d2 = normalize(d2);

    float3 A0 = d1;
    float3 A1 = -1*d2;

    float M00 = dot(A0, A0);
    float M01 = dot(A0, A1);
    float M10 = dot(A1, A0);
    float M11 = dot(A1, A1);

    float det = M00 * M11 - M01 * M10;

    if (fabs(det) < EPS) {
        float3 diff = P2 - P1;

        if (length(diff) < EPS) {
            float3 P = P2 + d2 * d2_length;
            float s = 1.0f;
            return make_float2(dot(P - P1, d1), s);
        }
        else if (fabs(dot(normalize(diff), d1)) > 1.0f - EPS) {
            float3 P = P2;
            float s = 0.0f;
            return make_float2(dot(P - P1, d1), s);
        }
        else {
            return make_float2(-1.0f, -1.0f);
        }
    }

    float invDet = 1.0f / det;

    float2 res;
    float3 rhs = P2 - P1;

    float b0 = dot(rhs, A0);
    float b1 = dot(rhs, A1);

    res.x = invDet * (M11 * b0 - M01 * b1);
    res.y = invDet * (-M10 * b0 + M00 * b1) / d2_length;

    return res;
}

__device__ void trace_in_triangles(
    const Mesh& mesh, float3 dir_3d,
    const MeshPoint& curr_point, int curr_tri,
    float max_len, float3& next_pos, float3& next_bary
) {
    float3 curr_pos = curr_point.interpolate(mesh);

    float best_t = 1e20f;
    float3 best_intersection = make_float3(0, 0, 0);
    int best_edge = -1;
    float best_edge_param = -1.0f;

    int edges[3][2] = { {0,1}, {1,2}, {2,0} };

    for (int e = 0; e < 3; ++e) {
        int i = mesh.triangles[curr_tri*3+edges[e][0]];
        int j = mesh.triangles[curr_tri*3+edges[e][1]];

        float3 p_i = mesh.positions[i];
        float3 p_j = mesh.positions[j];
        float3 edge_dir = p_j - p_i;

        float3 n = cross(dir_3d, edge_dir);
        if (length(n) < EPS) continue;

        float2 res = closest_point_parameter_coplanar(curr_pos, dir_3d, p_i, edge_dir);
        float t = res.x;
        float edge_param = res.y;

        if (t < -EPS || edge_param < -EPS || edge_param > 1.0f + EPS) continue;

        float3 intersection = curr_pos + dir_3d * t;

        if (t < best_t) {
            best_t = t;
            best_intersection = intersection;
            best_edge = e;
            best_edge_param = edge_param;
        }
    }

    if (best_edge == -1) {
        next_pos = curr_pos;
        next_bary = curr_point.get_barycentric_coords();
        return;
    }

    if (length(curr_pos - best_intersection) > max_len) {
        next_pos = curr_pos + max_len * dir_3d;

        int vi0 = mesh.triangles[curr_tri*3];
        int vi1 = mesh.triangles[curr_tri*3+1];
        int vi2 = mesh.triangles[curr_tri*3+2];

        next_bary = tri_bary_coords(
            mesh.positions[vi0],
            mesh.positions[vi1],
            mesh.positions[vi2],
            next_pos
        );
        return;
    }

    next_pos = best_intersection;

    next_bary = make_float3(0, 0, 0);
    int i = edges[best_edge][0];
    int j = edges[best_edge][1];
    set_float3_component(next_bary, i, 1.0f - best_edge_param);
    set_float3_component(next_bary, j, best_edge_param);
}



__device__ void common_edge(
    const int* triangles, int tri1, int tri2,
    int common_verts[2], int& diff_vert_1, int& diff_vert_2, bool& valid
) {
    int verts1[3] = {triangles[tri1*3], triangles[tri1*3+1], triangles[tri1*3+2]};
    int verts2[3] = {triangles[tri2*3], triangles[tri2*3+1], triangles[tri2*3+2]};

    // Find common vertices
    int common_count = 0;
    bool common_flags1[3] = {false, false, false};
    bool common_flags2[3] = {false, false, false};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (verts1[i] == verts2[j]) {
                if (common_count < 2) {
                    common_verts[common_count] = verts1[i];
                }
                common_flags1[i] = true;
                common_flags2[j] = true;
                common_count++;
            }
        }
    }

    if (common_count != 2) {
        valid = false;
        diff_vert_1 = -1;
        diff_vert_2 = -1;
        return;
    }

    // Find unique verts (not common)
    diff_vert_1 = -1;
    diff_vert_2 = -1;

    for (int i = 0; i < 3; i++) {
        if (!common_flags1[i]) diff_vert_1 = verts1[i];
        if (!common_flags2[i]) diff_vert_2 = verts2[i];
    }

    valid = true;
}


__device__ float signed_angle(float3 A, float3 B, float3 N) {
    N = normalize(N);

    A = A - dot(A, N) * N;
    B = B - dot(B, N) * N;

    float lenA = length(A);
    float lenB = length(B);

    if (lenA < EPS || lenB < EPS) return 0.0f;

    A = A / lenA;
    B = B / lenB;

    float3 cross_prod = cross(A, B);
    float dot_prod = dot(A, B);
    float sign = dot(N, cross_prod);

    float angle = atan2f(sign, dot_prod); // result in radians
    return angle;
}



__device__ void compute_parallel_transport_edge(
    const Mesh& mesh,
    int curr_tri,
    int next_tri,
    float3 dir_3d,
    float3 curr_normal,
    float3& out_dir,
    float3& out_normal
) {
    if (curr_tri == next_tri) {
        out_dir = dir_3d;
        out_normal = curr_normal;
        return;
    }

    int common_e[2];
    int other_v1, other_v2;
    bool valid_edge;
    common_edge(mesh.triangles, curr_tri, next_tri, common_e, other_v1, other_v2, valid_edge);

    if (!valid_edge) {
        out_dir = dir_3d;
        out_normal = curr_normal;
        return;
    }

    float3 p0 = mesh.positions[common_e[0]];
    float3 p1 = mesh.positions[common_e[1]];
    float3 p2_curr = mesh.positions[other_v1];
    float3 p2_next = mesh.positions[other_v2];

    float3 n1 = triangle_normal(p0, p1, p2_curr);
    float3 n2 = triangle_normal(p0, p1, p2_next);

    float3 edge_dir = p1 - p0;
    float3 axis = normalize(edge_dir);

    float3 sym_axis = normalize(cross(axis, n1));
    // Reflect dir_3d about sym_axis: dir_3d - 2 * (dir_3d · sym_axis) * sym_axis
    dir_3d = dir_3d - 2.0f * dot(dir_3d, sym_axis) * sym_axis;

    float angle = signed_angle(n1, n2, axis);

    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Rodrigues' rotation formula: rotate dir_3d around axis by angle
    float3 rotated_dir =
        dir_3d * cos_a +
        cross(axis, dir_3d) * sin_a +
        axis * dot(axis, dir_3d) * (1.0f - cos_a);

    float normal_sign = dot(n1, curr_normal);
    float3 normal = normal_sign * (-1 * n2);

    out_dir = rotated_dir;
    out_normal = normal;
}


__device__ void compute_parallel_transport_vertex(
    const Mesh& mesh,
    int curr_tri,
    int vertex_id,
    float3 dir_3d,
    float3 curr_normal,
    float3& out_dir,
    int& out_tri,
    float3& out_normal
) {
    int len_connected_triangles = mesh.v2t[vertex_id * mesh.max_v2t];
    // v2t layout: [count, tri_id1, tri_id2, ...]
    
    float total_angle = 0.0f;

    // Sum absolute angles around the vertex
    for (int idx = 1; idx <= len_connected_triangles; idx++) {
        int tri_id = mesh.v2t[vertex_id * mesh.max_v2t + idx];
        
        // Find the two other vertices in this triangle that are not vertex_id
        int verts[2];
        int vcount = 0;
        for (int i = 0; i < 3; i++) {
            int v = mesh.triangles[tri_id * 3 + i];
            if (v != vertex_id) {
                verts[vcount++] = v;
            }
        }

        float3 vec0 = mesh.positions[verts[0]] - mesh.positions[vertex_id];
        float3 vec1 = mesh.positions[verts[1]] - mesh.positions[vertex_id];
        float3 n = mesh.triangle_normals[tri_id];

        float angle = fabsf(signed_angle(vec0, vec1, n));
        total_angle += angle;
    }

    float half_angle = total_angle * 0.5f;
    float angle = 0.0f;

    // Invert dir_3d like in Python code
    dir_3d = make_float3(-dir_3d.x, -dir_3d.y, -dir_3d.z);

    // Find first v1 vertex in current triangle different from vertex_id
    int v1 = -1;
    for (int i = 0; i < 3; i++) {
        int v = mesh.triangles[curr_tri * 3 + i];
        if (v != vertex_id) {
            v1 = v;
            break;
        }
    }

    // Find v2 vertex different from vertex_id and v1
    int v2 = -1;
    for (int i = 0; i < 3; i++) {
        int v = mesh.triangles[curr_tri * 3 + i];
        if (v != vertex_id && v != v1) {
            v2 = v;
            break;
        }
    }

    float3 p0 = mesh.positions[vertex_id];
    float3 p1 = mesh.positions[v1];
    float3 p2 = mesh.positions[v2];

    float3 n = triangle_normal(p0, p1, p2);

    int normal_sign = (dot(n, curr_normal) > 0) ? 1 : -1;

    angle += fabsf(signed_angle(dir_3d, p1 - p0, n));

    while (angle < half_angle) {
        // Find local_edge_idx: the index in curr_tri of the vertex != v1 and != vertex_id, then +1 % 3
        int local_edge_idx = -1;
        for (int i = 0; i < 3; i++) {
            int v = mesh.triangles[curr_tri * 3 + i];
            if (v != v1 && v != vertex_id) {
                local_edge_idx = (i + 1) % 3;
                break;
            }
        }

        int next_tri = mesh.adjacencies[curr_tri * 3 + local_edge_idx];
        if (next_tri == -1) {
            // No adjacent triangle, rotate and return
            float angle_diff = half_angle - angle;
            float3 axis = n;
            float3 edge = p1 - p0;
            float3 new_dir =
                edge * cosf(angle_diff) +
                cross(axis, edge) * sinf(angle_diff) +
                axis * dot(axis, edge) * (1.0f - cosf(angle_diff));

            out_dir = new_dir;
            out_tri = curr_tri;
            out_normal = normal_sign * (-1*n);
            return;
        }

        // Find v2 in next_tri different from v1 and vertex_id
        v2 = -1;
        for (int i = 0; i < 3; i++) {
            int v = mesh.triangles[next_tri * 3 + i];
            if (v != v1 && v != vertex_id) {
                v2 = v;
                break;
            }
        }

        p0 = mesh.positions[vertex_id];
        p1 = mesh.positions[v1];
        p2 = mesh.positions[v2];

        n = triangle_normal(p0, p1, p2);

        float tri_angle = fabsf(signed_angle(p1 - p0, p2 - p0, n));

        if (angle + tri_angle >= half_angle - EPS) {
            float angle_diff = half_angle - angle;
            float3 axis = n;
            float3 edge = p1 - p0;
            float3 new_dir =
                edge * cosf(angle_diff) +
                cross(axis, edge) * sinf(angle_diff) +
                axis * dot(axis, edge) * (1.0f - cosf(angle_diff));

            out_dir = new_dir;
            out_tri = next_tri;
            out_normal = normal_sign * (-1*n);
            return;
        }

        curr_tri = next_tri;
        v1 = v2;
        angle += tri_angle;
    }

    // Fallback return zero vector, 0, and curr_normal
    out_dir = make_float3(0.0f, 0.0f, 0.0f);
    out_tri = 0;
    out_normal = curr_normal;
}


__device__ float2 bary_to_uv(const float3 bary) {
    return make_float2(bary.y, bary.z);
}

__global__ std::tuple<MeshPoint,float3,float3> straightest_geodesic(
    const Mesh& mesh,
    const MeshPoint& start,
    float3 dir
) {
    float3 current_normal = mesh.triangle_normals[start.face];
    dir = project_vec(dir, current_normal);

    float len_path = 0.0f;
    float path_len = length(dir);

    float3 curr_pos = start.interpolate(mesh);
    float3 curr_normal = current_normal;
    float3 next_pos;
    float3 dir_proj = normalize(dir);

    float3 curr_bary = { 1.0f - start.uv.x - start.uv.y, start.uv.x, start.uv.y };
    float3 next_bary = { 0.0f, 0.0f, 0.0f };

    MeshPoint curr_point = start;
    int curr_tri = start.face;
    int next_tri = -1;

    while (len_path < path_len - EPS) {
        float3 tri_normal = mesh.triangle_normals[curr_tri];
        dir_proj = project_vec(dir, tri_normal);

        if (length(dir_proj) < EPS)
            break;

        dir_proj = normalize(dir_proj);

        // trace_in_triangles must compute next_pos and next_bary
        trace_in_triangles(mesh, dir_proj, curr_point, curr_tri, path_len - len_path, next_pos, next_bary);

        bool is_edge_bary;
        int edge_idx;
        bary_is_edge(next_bary, &is_edge_bary, &edge_idx);

        bool is_vert_bary;
        int vert_idx;
        bary_is_vert(next_bary, &is_vert_bary, &vert_idx);

        len_path += length(next_pos - curr_pos);

        if (is_vert_bary) {
            int v_idx = mesh.triangles[curr_tri * 3 + vert_idx];
            compute_parallel_transport_vertex(
                mesh, curr_tri, v_idx, dir_proj, curr_normal,
                dir, next_tri, current_normal
            );

            // Determine closest vertex in next_tri
            float3 p0 = mesh.positions[mesh.triangles[next_tri * 3 + 0]];
            float3 p1 = mesh.positions[mesh.triangles[next_tri * 3 + 1]];
            float3 p2 = mesh.positions[mesh.triangles[next_tri * 3 + 2]];
            float3 d0 = next_pos - p0;
            float3 d1 = next_pos - p1;
            float3 d2 = next_pos - p2;
            float dists[3] = {
                dot(d0, d0),
                dot(d1, d1),
                dot(d2, d2)
            };

            // Take argmin of distances
            next_bary.x = next_bary.y = next_bary.z = 0.0f;
            if (dists[0] < dists[1] && dists[0] < dists[2]) {
                next_bary.x = 1.0f;
            }
            else if (dists[1] < dists[0] && dists[1] < dists[2]) {
                next_bary.y = 1.0f;
            }
            else {
                next_bary.z = 1.0f;
            }

            curr_tri = next_tri;
            curr_bary = next_bary;
            curr_pos = next_pos;
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary));
            curr_normal = current_normal;
        }
        else if (is_edge_bary) {
            int adj_tri = mesh.adjacencies[curr_tri * 3 + edge_idx];
            if (adj_tri == -1)
                adj_tri = curr_tri;

            float3 p0 = mesh.positions[mesh.triangles[adj_tri * 3 + 0]];
            float3 p1 = mesh.positions[mesh.triangles[adj_tri * 3 + 1]];
            float3 p2 = mesh.positions[mesh.triangles[adj_tri * 3 + 2]];

            next_bary = tri_bary_coords(p0, p1, p2, next_pos);

            if (adj_tri == -1) {
                curr_point = MeshPoint(adj_tri, bary_to_uv(next_bary));
                break;
            }
            
            int e[2];
            int diff_vert_1, diff_vert_2;
            bool found;
            common_edge(mesh.triangles, curr_tri, adj_tri, e, diff_vert_1, diff_vert_2, found);
            if (!found) {
                curr_point = MeshPoint(adj_tri, bary_to_uv(next_bary));
                break;
            }

            compute_parallel_transport_edge(
                mesh, curr_tri, adj_tri, dir_proj, curr_normal,
                dir, current_normal
            );

            curr_tri = adj_tri;
            curr_bary = next_bary;
            curr_pos = next_pos;
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary));
            curr_normal = current_normal;
        }
        else {
            curr_pos = next_pos;
            curr_bary = next_bary;
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary));
            curr_normal = current_normal;
            dir = dir_proj;
        }
    }
    return std::make_tuple(curr_point, dir, curr_normal);
}
}
