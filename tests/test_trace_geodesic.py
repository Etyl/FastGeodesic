import numpy as np

from trace_geodesic import *


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

    result = tri_bary_coords(p0, p1, p2, p)
    assert not all(0 <= val <= 1 for val in result)

def test_barycentric_degenerate_triangle():
    p0 = p1 = p2 = np.array([1.0, 1.0])
    p = np.array([1.0, 1.0])  # Same point

    result = tri_bary_coords(p0, p1, p2, p)
    assert np.allclose(result, [1.0, 0.0, 0.0])  # Fallback case