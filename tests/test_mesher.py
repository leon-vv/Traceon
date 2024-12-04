import unittest
import ctypes as C

import numpy as np

import traceon.backend as B
import traceon.mesher as M
from traceon.mesher import Mesh


class TestMeshDeduplication(unittest.TestCase):
    def test_no_duplicates(self):
        # Test case with no duplicate points
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        triangles = np.array([
            [0, 1, 2]
        ])
        mesh = Mesh(points, triangles=triangles)
        mesh._deduplicate_points()
        self.assertEqual(len(mesh.points), 3)
        np.testing.assert_array_equal(mesh.points, points)
        np.testing.assert_array_equal(mesh.triangles, triangles)

    def test_with_duplicates(self):
        # Test case with duplicate points
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # Duplicate of the first point
            [0.0, 1.0, 0.0]
        ])
        triangles = np.array([
            [0, 1, 3],
            [2, 1, 3]
        ])
        expected_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        expected_triangles = np.array([
            [0, 1, 2],
            [0, 1, 2]
        ])
        mesh = Mesh(points, triangles=triangles)
        mesh._deduplicate_points()
        self.assertEqual(len(mesh.points), 3)
        np.testing.assert_array_equal(mesh.points, expected_points)
        np.testing.assert_array_equal(mesh.triangles, expected_triangles)

    def test_all_duplicates(self):
        # Test case where all points are duplicates
        points = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        triangles = np.array([
            [0, 1, 2]
        ])
        expected_points = np.array([
            [1.0, 1.0, 1.0]
        ])
        expected_triangles = np.zeros( (0,3), dtype=np.int64) # Degenerate triangle is removed
         
        mesh = Mesh(points, triangles=triangles)
        
        self.assertEqual(len(mesh.points), 1)
        np.testing.assert_array_equal(mesh.points, expected_points)
        np.testing.assert_array_equal(mesh.triangles, expected_triangles)

    def test_close_points(self):
        # Test case with points that are very close to each other
        points = np.array([
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1 + 1e-12],  # Very close to the first point
            [1.0, 0.0, 0.0]
        ])
        triangles = np.array([
            [0, 1, 2]
        ])
        expected_points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 3.0]
        ])
        expected_triangles = np.array([
            [2, 1, 1]
        ])
        mesh = Mesh(points, triangles=triangles)
        
        self.assertEqual(len(mesh.points), 3)
        np.testing.assert_array_almost_equal(mesh.points, expected_points, decimal=5)
        np.testing.assert_array_equal(mesh.triangles, expected_triangles)

    def test_empty_mesh(self):
        mesh = Mesh(np.array([]), triangles=np.array([]))
    
    def test_empty_triangles(self):
        mesh = Mesh(np.random.rand(10, 3), triangles=np.array([]))

    def test_complex_large_mesh(self):
        # Generate a large number of random points
        num_points = 1000
        points = np.random.rand(num_points, 3)

        # Introduce zeros into the points
        points[:100] = 0.0

        # Introduce exact duplicates
        duplicate_indices = np.random.choice(num_points, size=100, replace=False)
        points[duplicate_indices] = points[np.random.choice(num_points, size=100, replace=False)]

        # Introduce points that are very close to each other
        close_points_indices = np.random.choice(num_points, size=100, replace=False)
        points[close_points_indices] += np.random.rand(100, 3) * 1e-12
        
        # Shuffle the points to create a chaotic order
        np.random.shuffle(points)

        # Assign triangles that cover all the points
        num_triangles = num_points // 3
        triangles = np.arange(num_triangles * 3).reshape(num_triangles, 3)
        np.random.shuffle(triangles)
        
        # Make copies of points and triangles before deduplication
        original_points = points.copy()
        original_triangles = triangles.copy()

        # Create a Mesh instance and perform deduplication
        mesh = Mesh()
        mesh.points = points # Do not use constructor, prevent triangle degenerate removal
        mesh.triangles = triangles
        
        # Map the original and deduplicated triangles back to their points
        original_triangle_points = original_points[original_triangles]
        deduplicated_triangle_points = mesh.points[mesh.triangles]

        # Check that the triangle points before and after deduplication are very close
        self.assertEqual(original_triangle_points.shape, deduplicated_triangle_points.shape)
        np.testing.assert_allclose(original_triangle_points, deduplicated_triangle_points, atol=1e-5)


class TestConnectedElements(unittest.TestCase):

    def test_single_element(self):
        elements = np.array([[1, 2]])
        expected_result = [np.array([0])]
        result = M._get_connected_elements(elements)
        self.assertEqual(len(result), len(expected_result))
        np.testing.assert_array_equal(result[0], expected_result[0])

    def test_two_connected_elements(self):
        elements = np.array([[1, 2], [2, 3]])
        expected_result = [np.array([0, 1])]
        result = M._get_connected_elements(elements)
        self.assertEqual(len(result), len(expected_result))
        np.testing.assert_array_equal(result[0], expected_result[0])

    def test_two_disconnected_elements(self):
        elements = np.array([[1, 2], [3, 4]])
        expected_result = [np.array([0]), np.array([1])]
        result = M._get_connected_elements(elements)
        self.assertEqual(len(result), len(expected_result))
        np.testing.assert_array_equal(result[0], expected_result[0])
        np.testing.assert_array_equal(result[1], expected_result[1])

    def test_multiple_connected_elements(self):
        elements = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
        expected_result = [np.array([0, 1, 2]), np.array([3])]
        result = M._get_connected_elements(elements)
        self.assertEqual(len(result), len(expected_result))
        np.testing.assert_array_equal(result[0], expected_result[0])
        np.testing.assert_array_equal(result[1], expected_result[1])

    def test_triangle_elements(self):
        elements = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
        expected_result = [np.array([0, 1]), np.array([2])]
        result = M._get_connected_elements(elements)
        self.assertEqual(len(result), len(expected_result))
        np.testing.assert_array_equal(result[0], expected_result[0])
        np.testing.assert_array_equal(result[1], expected_result[1])




class TestLineOrientation(unittest.TestCase):
    
    def test_normals_same_direction(self):
        points = np.array([
            [0, 0],
            [1, 1], 
            [2, 2] ], dtype=np.float64)

        lines = np.array([
            [0, 1],
            [1, 2]])
        
        normal1 = B.normal_2d(points[lines[0, 0]], points[lines[0, 1]])
        normal2 = B.normal_2d(points[lines[1, 0]], points[lines[1, 1]])
        assert np.allclose(normal1, normal2)
        assert M._line_orientation_equal(0, 1, lines)
    
    def test_normals_opposite_direction(self):
        points = np.array([
            [0, 0],
            [1, 1], 
            [2, 2] ], dtype=np.float64)

        lines = np.array([
            [0, 1],
            [2, 1]])
        
        normal1 = B.normal_2d(points[lines[0, 0]], points[lines[0, 1]])
        normal2 = B.normal_2d(points[lines[1, 0]], points[lines[1, 1]])
        assert np.allclose(normal1, -normal2)
        assert not M._line_orientation_equal(0, 1, lines)

    def test_normals_large_angle(self):
        points = np.array([
            [-1, 0],
            [0, 10], 
            [1, 0] ], dtype=np.float64)

        lines = np.array([
            [0, 1],
            [1, 2]])
        
        normal1 = B.normal_2d(points[lines[0, 0]], points[lines[0, 1]])
        normal2 = B.normal_2d(points[lines[1, 0]], points[lines[1, 1]])
        
        assert np.allclose(normal1[0], -normal2[0])
        assert np.allclose(normal1[1], normal2[1])
        assert M._line_orientation_equal(0, 1, lines)

def triangle_orientation_is_equal(index1, index2, triangles, points):
    triangles_ctypes = triangles.ctypes.data_as(C.c_void_p)
    points_ctypes = points.ctypes.data_as(C.c_void_p)
    
    assert 0 <= index1 < len(triangles)
    assert 0 <= index2 < len(triangles)
    assert triangles.dtype == np.uint64
    assert points.dtype == np.float64
    
    return B.backend_lib.triangle_orientation_is_equal(index1, index2, triangles_ctypes, points_ctypes)

class TestTriangleOrientation(unittest.TestCase):
    def test_identical_triangles(self):
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]], dtype=np.float64)

        triangles = np.array([[0, 1, 2]], dtype=np.uint64)

        result = triangle_orientation_is_equal(0, 0, triangles, points)
        assert result == 0
    
    def test_adjacent_triangles(self):
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0]], dtype=np.float64)

        triangles = np.array([[0, 1, 2], [0, 3, 2]], dtype=np.uint64)

        result = triangle_orientation_is_equal(0, 1, triangles, points)
        assert result == 0
    
    def test_adjacent_triangles2(self):
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0]], dtype=np.float64)

        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint64)

        result = triangle_orientation_is_equal(0, 1, triangles, points)
        assert result == 1
    
    def test_adjacent_triangles3(self):
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0.5],
            [0.5, 0, 0]], dtype=np.float64)
        
        triangles = np.array([[0, 0, 0], [0, 1, 2], [0, 3, 4]], dtype=np.uint64)

        result = triangle_orientation_is_equal(1, 2, triangles, points)
        assert result == 1
    
    def test_disconnected_triangles(self):
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1]], dtype=np.float64)
        
        triangles = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.uint64)

        result = triangle_orientation_is_equal(0, 1, triangles, points)
        assert result == -1



















