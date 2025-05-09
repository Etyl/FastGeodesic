import numpy as np

from fastgeodesic.geometry.trace_geodesic import *


def test_triangle_normal_basic():
    # Define a triangle in the XY plane
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])

    expected_normal = np.array([0.0, 0.0, 1.0])  # Normal to the XY plane

    result = triangle_normal(p0, p1, p2)
    assert np.allclose(result, expected_normal)

def test_triangle_normal_reverse_order():
    # Changing order of vertices should reverse the normal
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    expected_normal = np.array([0.0, 0.0, -1.0])  # Reversed normal

    result = triangle_normal(p0, p1, p2)
    assert np.allclose(result, expected_normal)

def test_triangle_normal_degenerate():
    # Degenerate triangle (all points on a line or same point)
    p0 = p1 = p2 = np.array([1.0, 1.0, 1.0])
    result = triangle_normal(p0, p1, p2)
    expected = np.array([0.0, 0.0, 0.0])  # No defined normal
    assert np.allclose(result, expected)


def test_barycentric_vertex_points():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.0, 1.0])

    assert np.allclose(tri_bary_coords(p0, p1, p2, p0), [1.0, 0.0, 0.0])
    assert np.allclose(tri_bary_coords(p0, p1, p2, p1), [0.0, 1.0, 0.0])
    assert np.allclose(tri_bary_coords(p0, p1, p2, p2), [0.0, 0.0, 1.0])

def test_barycentric_inside_triangle():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.0, 1.0])
    p = np.array([0.25, 0.25])  # Inside the triangle

    result = tri_bary_coords(p0, p1, p2, p)
    assert np.allclose(result.sum(), 1.0)
    assert all(0 <= val <= 1 for val in result)

def test_barycentric_on_edge():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.0, 1.0])
    p = np.array([0.5, 0.0])  # On edge between p0 and p1

    result = tri_bary_coords(p0, p1, p2, p)
    assert np.allclose(result.sum(), 1.0)
    assert np.isclose(result[2], 0.0)

def test_barycentric_outside_triangle():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.0, 1.0])
    p = np.array([1.0, 1.0])  # Outside triangle

    try:
        result = tri_bary_coords(p0, p1, p2, p)
    except AssertionError as e:
        result = None
    
    assert result is None  # Expecting an error 

def test_barycentric_degenerate_triangle():
    p0 = p1 = p2 = np.array([1.0, 1.0])
    p = np.array([1.0, 1.0])  # Same point

    result = tri_bary_coords(p0, p1, p2, p)
    assert np.allclose(result, [1.0, 0.0, 0.0])  # Fallback case


# Tests for point_is_edge

def test_edge_0():
    point = MeshPoint(0,np.array([0.5, 0.0]))  # On edge opposite vertex 0
    is_edge, edge_idx = point_is_edge(point)
    assert is_edge
    assert edge_idx == 0

def test_edge_1():
    point = MeshPoint(0,np.array([0.5, 0.5]))  # Same point as edge 1 (due to uv[0] == 0)
    is_edge, edge_idx = point_is_edge(point)
    assert is_edge
    assert edge_idx == 1 

def test_edge_2():
    point = MeshPoint(0,np.array([0.0, 0.5]))  # On edge opposite vertex 1
    is_edge, edge_idx = point_is_edge(point)
    assert is_edge
    assert edge_idx == 2

def test_not_on_edge():
    point = MeshPoint(0,np.array([0.3, 0.3]))  # Inside the triangle, not on edge
    is_edge, edge_idx = point_is_edge(point)
    assert not is_edge
    assert edge_idx == -1


# Tests for point_is_vert

def test_vertex_0():
    point = MeshPoint(0,np.array([0.0, 0.0]))
    is_vert, vert_idx = point_is_vert(point)
    assert is_vert
    assert vert_idx == 0

def test_vertex_1():
    point = MeshPoint(0,np.array([1.0, 0.0]))
    is_vert, vert_idx = point_is_vert(point)
    assert is_vert
    assert vert_idx == 1

def test_vertex_2():
    point = MeshPoint(0,np.array([0.0, 1.0]))
    is_vert, vert_idx = point_is_vert(point)
    assert is_vert
    assert vert_idx == 2

def test_not_on_vertex():
    point = MeshPoint(0,np.array([0.2, 0.3]))
    is_vert, vert_idx = point_is_vert(point)
    assert not is_vert
    assert vert_idx == -1


# Tests for bary_is_edge

def test_bary_edge_0_opposite_vertex_0():
    bary = np.array([0.0, 0.5, 0.5])
    is_edge, edge_idx = bary_is_edge(bary)
    assert is_edge
    assert edge_idx == 1  # Edge opposite vertex 0

def test_bary_edge_1_opposite_vertex_1():
    bary = np.array([0.5, 0.0, 0.5])
    is_edge, edge_idx = bary_is_edge(bary)
    assert is_edge
    assert edge_idx == 2  # Edge opposite vertex 1

def test_bary_edge_2_opposite_vertex_2():
    bary = np.array([0.5, 0.5, 0.0])
    is_edge, edge_idx = bary_is_edge(bary)
    assert is_edge
    assert edge_idx == 0  # Edge opposite vertex 2

def test_bary_not_on_edge():
    bary = np.array([0.2, 0.3, 0.5])
    is_edge, edge_idx = bary_is_edge(bary)
    assert not is_edge
    assert edge_idx == -1


# Tests for bary_is_vert

def test_bary_vertex_0():
    bary = np.array([1.0, 0.0, 0.0])
    is_vert, vert_idx = bary_is_vert(bary)
    assert is_vert
    assert vert_idx == 0

def test_bary_vertex_1():
    bary = np.array([0.0, 1.0, 0.0])
    is_vert, vert_idx = bary_is_vert(bary)
    assert is_vert
    assert vert_idx == 1

def test_bary_vertex_2():
    bary = np.array([0.0, 0.0, 1.0])
    is_vert, vert_idx = bary_is_vert(bary)
    assert is_vert
    assert vert_idx == 2

def test_bary_not_on_vertex():
    bary = np.array([0.2, 0.3, 0.5])
    is_vert, vert_idx = bary_is_vert(bary)
    assert not is_vert
    assert vert_idx == -1


# Tests for closest_point_parameter_coplanar

def test_lines_not_parallel():
    P1 = np.array([0.0, 0.0, 0.0])
    d1 = np.array([1.0, 0.0, 0.0])  # Line 1 along the x-axis
    P2 = np.array([1.0, 0.0, 0.0])
    d2 = np.array([0.0, 1.0, 0.0])  # Line 2 along the y-axis

    t1, t2 = closest_point_parameter_coplanar(P1, d1, P2, d2)
    assert np.isclose(t1, 1)
    assert np.isclose(t2, 0)

def test_lines_negative():
    P1 = np.array([1.0, 0.0, 0.0])
    d1 = np.array([1.0, 0.0, 0.0])  # Line 1 along the x-axis
    P2 = np.array([0.0, 0.0, 0.0])
    d2 = np.array([0.0, 1.0, 0.0])  # Line 2 along the y-axis

    t1, t2 = closest_point_parameter_coplanar(P1, d1, P2, d2)
    assert np.isclose(t1, -1)
    assert np.isclose(t2, 0)

def test_lines_parallel():
    P1 = np.array([0.0, 0.0, 0.0])
    d1 = np.array([1.0, 0.0, 0.0])  # Line 1 along the x-axis
    P2 = np.array([0.0, 1.0, 0.0])
    d2 = np.array([1.0, 0.0, 0.0])  # Line 2 parallel to line 1

    # Lines are parallel, should return -1, -1
    t1, t2 = closest_point_parameter_coplanar(P1, d1, P2, d2)
    assert np.isclose(t1, -1)
    assert np.isclose(t2, -1)

def test_lines_coincident():
    P1 = np.array([0.0, 0.0, 0.0])
    d1 = np.array([1.0, 0.0, 0.0])  # Line 1 along the x-axis
    P2 = np.array([0.0, 0.0, 0.0])  # Coincident with line 1
    d2 = np.array([1.0, 0.0, 0.0])  # Coincident direction

    # lines are coincident, should return -1, -1
    t1, t2 = closest_point_parameter_coplanar(P1, d1, P2, d2)
    assert np.isclose(t1, -1)
    assert np.isclose(t2, -1)


# TODO add tests for trace_in_triangles

# Tests for common_edge

def test_common_edge_found():
    triangles = np.array([
        [0, 1, 2],  # Triangle 1 vertices
        [1, 2, 3],  # Triangle 2 vertices
    ])
    
    # Common edge between triangle 0 and 1: vertices 1 and 2
    tri1 = 0
    tri2 = 1
    common, diff1, diff2 = common_edge(triangles, tri1, tri2)
    
    # Expected common edge [1, 2] and remaining vertices
    expected_common = np.array([1, 2])
    
    assert np.array_equal(common, expected_common)
    assert diff1 == 0
    assert diff2 == 3

def test_no_common_edge():
    triangles = np.array([
        [0, 1, 2],  # Triangle 1 vertices
        [3, 4, 5],  # Triangle 2 vertices
    ])
    
    # No common edge between triangle 0 and 1
    tri1 = 0
    tri2 = 1
    common, diff1, diff2 = common_edge(triangles, tri1, tri2)
    
    # Expected result: no common edge
    assert len(common) == 0
    assert diff1 is None
    assert diff2 is None

def test_single_common_vertex():
    triangles = np.array([
        [0, 1, 2],  # Triangle 1 vertices
        [4, 1, 3],  # Triangle 2 vertices (shared vertex 1 and 2, but no full edge)
    ])
    
    # Single common vertex between triangle 0 and 1: vertex 1
    tri1 = 0
    tri2 = 1
    common, diff1, diff2 = common_edge(triangles, tri1, tri2)
    
    # Expected result: no common edge
    assert len(common) == 0
    assert diff1 is None
    assert diff2 is None


# Tests for signed_angle

def test_signed_angle_zero():
    """
    Test that the signed angle is 0 when A and B are the same.
    """
    A = np.array([1.0, 0.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    N = np.array([0.0, 0.0, 1.0])
    angle = signed_angle(A, B, N)
    assert np.isclose(angle, 0.0)

def test_signed_angle_positive_90_degrees():
    """
    Test that the signed angle is +90 degrees (π/2 radians) when B is rotated CCW from A.
    """
    A = np.array([1.0, 0.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])
    N = np.array([0.0, 0.0, 1.0])
    angle = signed_angle(A, B, N)
    assert np.isclose(angle, np.pi / 2)

def test_signed_angle_negative_90_degrees():
    """
    Test that the signed angle is -90 degrees (-π/2 radians) when B is rotated CW from A.
    """
    A = np.array([1.0, 0.0, 0.0])
    B = np.array([0.0, -1.0, 0.0])
    N = np.array([0.0, 0.0, 1.0])
    angle = signed_angle(A, B, N)
    assert np.isclose(angle, -np.pi / 2)

def test_signed_angle_180_degrees():
    """
    Test that the signed angle is ±π when A and B are opposite directions.
    """
    A = np.array([1.0, 0.0, 0.0])
    B = np.array([-1.0, 0.0, 0.0])
    N = np.array([0.0, 0.0, 1.0])
    angle = signed_angle(A, B, N)
    assert np.isclose(abs(angle), np.pi)

def test_signed_angle_off_plane_projection():
    """
    Test that A and B are properly projected to the plane defined by N before angle computation.
    """
    A = np.array([1.0, 0.0, 1.0])  # Not in plane
    B = np.array([0.0, 1.0, 1.0])  # Not in plane
    N = np.array([0.0, 0.0, 1.0])  # XY plane
    angle = signed_angle(A, B, N)
    assert np.isclose(angle, np.pi / 2)

def test_signed_angle_with_near_zero_vectors():
    """
    Test that very small vectors result in an angle of 0 due to normalization threshold.
    """
    A = np.array([1e-10, 0.0, 0.0])
    B = np.array([0.0, 1e-10, 0.0])
    N = np.array([0.0, 0.0, 1.0])
    angle = signed_angle(A, B, N)
    assert np.isclose(angle, 0.0)

# TODO add tests for compute_parallel_transport
# TODO add tests for compute_parallel_transport_vertex