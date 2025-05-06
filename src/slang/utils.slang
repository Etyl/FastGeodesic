implementing trace_geodesic;

double3 triangle_normal(double3 p0, double3 p1, double3 p2) {
    // Compute the normal of a triangle.
    return normalize(cross(p1 - p0, p2 - p0));
}

double area_triangle(double3 p0, double3 p1, double3 p2) {
    // Compute the area of a triangle.
    return length(cross(p1 - p0, p2 - p0)) * 0.5;
}