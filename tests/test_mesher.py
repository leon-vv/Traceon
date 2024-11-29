import unittest

import numpy as np

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



