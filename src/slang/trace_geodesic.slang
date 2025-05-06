module trace_geodesic;

__include "utils.slang";
__include "mesh.slang";

const double EPS = 0.00001;

struct GeodesicPath {
    MeshPoint start;
    MeshPoint end;
    AppendStructuredBuffer<double3> path;
    AppendStructuredBuffer<double3> dirs;
    AppendStructuredBuffer<double3> normals;
    __init() {}
}

double3 tri_bary_coords(double3 p0, double3 p1, double3 p2, double3 p) {
    // Compute barycentric coordinates of p in the triangle (p0, p1, p2).
    double3 v0 = p1 - p0;
    double3 v1 = p2 - p0;
    double3 v2 = p - p0;
    
    double d00 = dot(v0, v0);
    double d01 = dot(v0, v1);
    double d11 = dot(v1, v1);
    double d20 = dot(v2, v0);
    double d21 = dot(v2, v1);

    double denom = d00 * d11 - d01 * d01;
    if (abs(denom) < EPS) {
        return { 1.0, 0.0, 0.0 };
    }

    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;

    //assert u > -EPS and v > -EPS and w > -EPS and u + v + w < 1+3*EPS, f"Invalid barycentric coordinates: {u}, {v}, {w}"

    return { u, v, w };
}

Tuple<bool, int> point_is_edge(MeshPoint point) {
    // Check if a mesh point is on an edge and return the edge index.
    if (abs(point.uv[0]) < EPS) return { true, 2 };
    if (abs(point.uv[1]) < EPS) return { true, 0 };
    if (abs(1 - point.uv[0] - point.uv[1]) < EPS) return { true, 1 };
    return { false, -1 };
}

Tuple<bool, int> point_is_vertex(MeshPoint point) {
    // Check if a mesh point is a vertex and return the vertex index.
    if (abs(point.uv[0]) < EPS && abs(point.uv[1]) < EPS) return { true, 0 };
    if (abs(point.uv[1]) > 1 - EPS) return { true, 1 };
    if (abs(point.uv[2]) > 1 - EPS) return { true, 2 };
    return { false, -1 };
}

Tuple<bool,int> bary_is_edge(double3 bary) {
    // Check if barycentric coordinates are on an edge and return the edge index.
    if (abs(bary[0]) < EPS) return { true, 1 };
    if (abs(bary[1]) < EPS) return { true, 2 };
    if (abs(bary[2]) < EPS) return { true, 0 };
    return { false, -1 };
}

Tuple<bool, int> bary_is_vertex(double3 bary) {
    // Check if barycentric coordinates are on a vertex and return the vertex index.
    if (abs(bary[0] - 1.0) < EPS) return { true, 0 };
    if (abs(bary[1] - 1.0) < EPS) return { true, 1 };
    if (abs(bary[2] - 1.0) < EPS) return { true, 2 };
    return { false, -1 };
}

double3 project_vec(double3 v, double3 normal) {
    // Project a vector onto a plane defined by its normal.
    normal = normalize(normal);
    return v - dot(v, normal) * normal;
}

double2 closest_point_parameter_coplanar(double3 P1, double3 d1, double3 P2, double3 d2) {
    /*
    Finds the closest points on two lines in 3D.
    Returns the parameters t1 and t2 for the closest points on the lines.
    */
    double2x3 A = double2x3(d1, -d2); 
    double2x2 M = mul(A,transpose(A));
    double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (abs(det) < EPS) {
        return { -1, -1 };
    }
    double2x2 M_inv = double2x2(M[1][1], -M[0][1], -M[1][0], M[0][0]) / det;
    double3x2 A_inv = mul(transpose(A),M_inv);
    double2 res = mul(P2 - P1, A_inv);
    return res;
}

double2x3 trace_in_triangles(Mesh mesh, double3 dir_3d, MeshPoint curr_point, int curr_tri, double max_len) {
    // Trace a straight line within triangles.
    // Get the current position
    double3 curr_pos = curr_point.interpolate(mesh);

    // Find the intersection with the triangle edges
    double2x3 intersections[3];
    uint k = 0;
    
    // Check each edge
    int2 edges[3] = { {0, 1}, {1, 2}, {2, 0} };
    for (int edge_idx = 0; edge_idx < 3; edge_idx++) {
        int i = edges[edge_idx][0];
        int j = edges[edge_idx][1];
        
        int vi = mesh.triangles[curr_tri][i];
        int vj = mesh.triangles[curr_tri][j];
        
        double3 p_i = mesh.positions[vi];
        double3 p_j = mesh.positions[vj];
        
        double3 edge_dir = p_j - p_i;
        double3 normal = cross(dir_3d, edge_dir);
        
        if (length(normal) < EPS) {
            continue;
        }

        // Find the intersection parameter t
        double2 res = closest_point_parameter_coplanar(curr_pos, dir_3d, p_i, edge_dir);
        
        if (res[0] < -EPS || res[1] < -EPS || res[1] > 1 + EPS) {
            continue;
        }

        // Valid intersection
        double3 intersection = curr_pos + res[0] * dir_3d;
        double2x3 param_intersection;
        param_intersection[0] = intersection;
        param_intersection[1][0] = res[0];
        param_intersection[1][1] = res[1];
        param_intersection[1][2] = edge_idx;
        intersections[k++] = param_intersection;
    }

    if (k == 0) {
        // No intersection with edges
        return { curr_pos, curr_point.get_barycentric_coords() };
    }

    // TODO check if OK
    // Sort intersections by distance
    for (uint i = 0; i < k - 1; i++) {
        for (uint j = i + 1; j < k; j++) {
            if (length(intersections[i][0] - curr_pos) > length(intersections[j][0] - curr_pos)) {
                double2x3 temp = intersections[i];
                intersections[i] = intersections[j];
                intersections[j] = temp;
            }
        }
    }

    uint idx = 0;
    if (k > 1) {
        if (length(intersections[0][0] - curr_pos) < EPS) {
            // If both intersections are very close, use the second one
            idx = 1;
        } else {
            idx = 0;
        }
    }

    double3 intersection = intersections[idx][0];
    double t = intersections[idx][1][0];
    double edge_param = intersections[idx][1][1];
    int edge_idx = (int)intersections[idx][1][2];

    if (length(curr_pos - intersection) > max_len) {
        // Intersection is too far, move for max_len in the direction
        double3 next_pos = curr_pos + max_len * dir_3d;
        double3 p0 = mesh.positions[mesh.triangles[curr_tri][0]];
        double3 p1 = mesh.positions[mesh.triangles[curr_tri][1]];
        double3 p2 = mesh.positions[mesh.triangles[curr_tri][2]];
        double3 next_bary = tri_bary_coords(p0, p1, p2, next_pos);
        return { next_pos, next_bary };
    }

    double3 next_pos = intersection;

    // Compute barycentric coordinates at the intersection
    int2 edge = edges[edge_idx];
    double3 next_bary = double3(0.0, 0.0, 0.0);
    next_bary[edge[0]] = 1.0 - edge_param;
    next_bary[edge[1]] = edge_param;
    return { next_pos, next_bary };
}

int4 common_edge(Mesh mesh, int tri1, int tri2) {
    // Find the common edge between two triangles.
    int3 t1 = mesh.triangles[tri1];
    int3 t2 = mesh.triangles[tri2];

    // Check for common vertices
    bool2x3 common_verts_total = { { false, false, false }, {false, false, false} };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (t1[i] == t2[j]) {
                common_verts_total[0][i] = true;
                common_verts_total[1][j] = true;
            }
        }
    }

    int2 common_verts;
    int k=0, diff1, diff2;
    for (int i = 0; i < 3; i++) {
        if (common_verts_total[0][i]) {
            common_verts[k++] = t1[i];
        } else {
            diff1 = t1[i];
        }
        if (!common_verts_total[1][i]) {
            diff2 = t2[i];
        }
    }

    if (k != 2) {
        return { -1, -1, -1, -1 };
    }

    return { common_verts[0], common_verts[1], diff1, diff2 };
}


double signed_angle(double3 A, double3 B, double3 N) {
    // Compute the signed angle between two vectors A and B with respect to a normal vector N.
    N = normalize(N);
    A = A - dot(A, N) * N;
    B = B - dot(B, N) * N;
    if (length(A) < EPS || length(B) < EPS) {
        return 0.0;
    }
    A = A / length(A);
    B = B / length(B);

    double3 cross_prod = cross(A, B);
    double dot_prod = dot(A, B);
    double sign = dot(N, cross_prod);
    double angle = atan2(sign, dot_prod);
    return angle;  // in radians
}

// TODO use Tuple

double2x3 compute_parallel_transport_edge(Mesh mesh, int curr_tri, int next_tri, double3 dir_3d, double3 curr_normal) {
    // Compute the parallel transport of a vector from one triangle to another.
    if (curr_tri == next_tri) {
        return { dir_3d, curr_normal };
    }

    // Find the common edge between triangles
    int4 common_e = common_edge(mesh, curr_tri, next_tri);
    
    if (common_e[0] == -1) {
        return { dir_3d, curr_normal };  // No common edge found
    }

    // Get the normals of both triangles
    double3 p0 = mesh.positions[common_e[0]];
    double3 p1 = mesh.positions[common_e[1]];
    double3 p2_curr = mesh.positions[common_e[2]];
    double3 p2_next = mesh.positions[common_e[3]];
    
    double3 n1 = triangle_normal(p0, p1, p2_curr);
    double3 n2 = triangle_normal(p0, p1, p2_next);
    
    // Get the edge direction
    double3 edge_dir = mesh.positions[common_e[1]] - mesh.positions[common_e[0]];
    double3 axis = normalize(edge_dir);

    double3 sym_axis = normalize(cross(axis, n1));
    dir_3d = dir_3d - 2 * (dot(dir_3d, sym_axis)) * sym_axis;
    
    // Compute the rotation angle
    double angle = signed_angle(n1, n2, axis);

    // Rodrigues' rotation formula
    dir_3d = dir_3d * cos(angle) + cross(axis, dir_3d) * sin(angle) + axis * dot(axis, dir_3d) * (1 - cos(angle));
    
    double normal_sign = dot(n1, curr_normal);
    double3 normal = normal_sign * (-n2);
    
    return { dir_3d, normal };
}

Tuple<double3,uint,double3> compute_parallel_transport_vertex(Mesh mesh, int curr_tri, int vertex_id, double3 dir_3d, double3 curr_normal) {
    // Compute the parallel transport of a vector at a vertex.
    StructuredBuffer<uint> connected_triangles = mesh.v2t[vertex_id];
    double total_angle = 0.0;
    for (int k1 = 0; k1 < connected_triangles.getCount(); k1++) {
        uint tri_id = connected_triangles[k1];
        uint2 vertices = { 0, 0 };
        int idx = 0;
        for (int k2 = 0; k2 < 3; k2++) {
            uint v = mesh.triangles[tri_id][k2];
            if (v != vertex_id) {
                vertices[idx++] = v;
            }
        }
        double3 n = mesh.triangle_normals[tri_id];

        double angle = signed_angle(
            mesh.positions[vertices[0]] - mesh.positions[vertex_id],
            mesh.positions[vertices[1]] - mesh.positions[vertex_id],
            n
        );
        total_angle += abs(angle);
    }

    double half_angle = total_angle / 2;
    double angle = 0;

    dir_3d = -dir_3d;
    uint2 v;
    int k = 0;
    for (int k1 = 0; k1 < 3; k1++) {
        uint vertex = mesh.triangles[curr_tri][k1];
        if (vertex != vertex_id) {
            v[k++] = vertex;
        }
    }

    double3 p0 = mesh.positions[vertex_id];
    double3 p1 = mesh.positions[v[0]];
    double3 p2 = mesh.positions[v[1]];
    double3 n = triangle_normal(p0, p1, p2);

    double normal_sign = -1;
    if (dot(n, curr_normal) > 0) {
        normal_sign = 1;
    }

    angle += abs(signed_angle(
        dir_3d,
        mesh.positions[v[0]] - mesh.positions[vertex_id],
        n
    ));

    uint v2, local_edge_index;
    uint v1 = v[0];
    while (angle < half_angle) {
        for (int k = 0; k < 3; k++) {
            uint vertex = mesh.triangles[curr_tri][k];
            if (vertex != vertex_id && vertex != v2) {
                local_edge_index = (k + 1) % 3;
                break;
            }
        }
        uint next_tri = mesh.adjacencies[curr_tri][local_edge_index];
        for (int k = 0; k < 3; k++) {
            uint vertex = mesh.triangles[curr_tri][k];
            if (vertex != vertex_id && vertex != v2) {
                v2 = vertex;
                break;
            }
        }

        double3 p0 = mesh.positions[vertex_id];
        double3 p1 = mesh.positions[v1];
        double3 p2 = mesh.positions[v2];
        double3 n = triangle_normal(p0, p1, p2);

        double tri_angle = abs(signed_angle(
            mesh.positions[v1] - mesh.positions[vertex_id],
            mesh.positions[v2] - mesh.positions[vertex_id],
            n
        ));

        if (angle + tri_angle >= half_angle - EPS) {
            double angle_diff = half_angle - angle;
            double3 axis = n;
            double3 edge = mesh.positions[v1] - mesh.positions[vertex_id];
            double3 new_dir = edge * cos(angle_diff) + cross(axis, edge) * sin(angle_diff) + axis * dot(axis, edge) * (1 - cos(angle_diff));
            return makeTuple(new_dir, next_tri, normal_sign * n);
        }

        curr_tri = next_tri;
        v1 = v2;
        angle += tri_angle;
    }

    return makeTuple( double3(0, 0, 0), uint(0), curr_normal);
}

double2 bary_to_uv(double3 bary) {
    /*
    Convert barycentric coordinates to UV coordinates.
    */
    return { bary[1], bary[2] };
}

GeodesicPath straighest_geodesic(Mesh mesh, MeshPoint start, double3 dir) {
    /*
    Compute the straightest geodesic path on a mesh.

    Args:
        mesh: The bezier mesh
        start: The starting point on the mesh
        dir: The initial 3D direction

    Returns:
        A geodesic path
    */
    double3 current_normal = mesh.triangle_normals[start.face];
    double3 dir = project_vec(dir, current_normal);

    double len_path = 0;
    double path_len = length(dir);

    GeodesicPath geodesic = GeodesicPath();
    geodesic.start = start;
    geodesic.path.Append(start.interpolate(mesh));
    geodesic.dirs.Append(normalize(dir));
    geodesic.normals.Append(current_normal);

    double3 next_bary;
    double3 curr_bary = { 1 - start.uv[0] - start.uv[1], start.uv[0], start.uv[1] };
    double3 curr_pos = start.interpolate(mesh);
    double3 next_pos;
    MeshPoint curr_point = start;
    double3 tid_normal;
    uint curr_tri = start.face;
    uint next_tri;

    while (len_path < path_len - EPS) {
        double3 tid_normal = mesh.triangle_normals[curr_tri];

        // Project the direction onto the triangle plane
        double3 proj_dir = project_vec(dir, tid_normal);
        if (length(proj_dir) < EPS) {
            // Direction is perpendicular to the triangle, cannot proceed
            break;
        }

        proj_dir = normalize(proj_dir);

        double2x3 res = trace_in_triangles(mesh, proj_dir, curr_point, curr_tri, path_len - len_path);
        next_pos = res[0];
        next_bary = res[1];

        // Check if the point is on an edge or vertex
        var res_edge = bary_is_edge(next_bary);
        var res_vert = bary_is_vertex(next_bary);
        bool is_edge_bary=res_edge._0, is_vert_bary=res_vert._0;
        int edge_idx=res_edge._1, vert_idx=res_vert._1;

        // Update the path
        len_path += length(next_pos - curr_pos);
        geodesic.path.Append(next_pos);

        if (is_vert_bary) {
            uint v_idx = mesh.triangles[curr_tri][vert_idx];

            // Transport the direction to the next triangle
            var res = compute_parallel_transport_vertex(mesh, curr_tri, v_idx, proj_dir, current_normal);
            double3 dir = res._0, current_normal = res._2;
            uint next_tri = res._1;
            geodesic.dirs.Append(dir);
            geodesic.normals.Append(current_normal);

            // Compute barycentric coordinates in the adjacent triangle
            double3 p0_adj = mesh.positions[mesh.triangles[next_tri][0]];
            double3 p1_adj = mesh.positions[mesh.triangles[next_tri][1]];
            double3 p2_adj = mesh.positions[mesh.triangles[next_tri][2]];
            next_bary = tri_bary_coords(p0_adj, p1_adj, p2_adj, next_pos);

            // Update current state
            curr_tri = next_tri;
            curr_pos = next_pos;
            curr_bary = next_bary;

            // Create a new mesh point for current position
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary));
        }
        else if (is_edge_bary) {
            // Find the adjacent triangle across this edge
            int edge_local_idx = edge_idx;
            uint adj_tri = mesh.adjacencies[curr_tri][edge_local_idx];
            if (adj_tri == -1) {
                adj_tri = curr_tri;
            }

            // Compute barycentric coordinates in the adjacent triangle
            double3 p0_adj = mesh.positions[mesh.triangles[adj_tri][0]];
            double3 p1_adj = mesh.positions[mesh.triangles[adj_tri][1]];
            double3 p2_adj = mesh.positions[mesh.triangles[adj_tri][2]];
            next_bary = tri_bary_coords(p0_adj, p1_adj, p2_adj, next_pos);

            if (adj_tri == -1) {
                curr_point = MeshPoint(adj_tri, bary_to_uv(next_bary));
                geodesic.dirs.Append(dir);
                break;
            }

            // Get the common edge vertices
            var res = common_edge(mesh, curr_tri, adj_tri);
            int2 e = { res[0], res[1] };

            if (e[0] == -1) {
                // No common edge found
                curr_point = MeshPoint(adj_tri, bary_to_uv(next_bary));
                break;
            }

            // Transport the direction to the adjacent triangle

            var res_transport = compute_parallel_transport_edge(mesh, curr_tri, adj_tri, proj_dir, current_normal);
            double3 dir = res_transport[0], current_normal = res_transport[1];
            geodesic.dirs.Append(dir);
            geodesic.normals.Append(current_normal);

            // Update current state
            curr_tri = adj_tri;
            curr_pos = next_pos;
            curr_bary = next_bary;

            // Create a new mesh point for current position
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary));
        }
        else {
            //  Point is in the interior of the triangle
            // Update current state
            curr_tri = curr_tri;  // No change in triangle
            curr_pos = next_pos;
            curr_bary = next_bary;

            // Create a new mesh point for current position
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary));
            
            // Continue in the same direction
            dir = proj_dir;
            geodesic.dirs.Append(dir);
            geodesic.normals.Append(current_normal);
        }
    }

    geodesic.end = curr_point;
    return geodesic;
}