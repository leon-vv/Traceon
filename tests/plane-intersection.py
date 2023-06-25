import numpy as np

from traceon.tracing import plane_intersection, xy_plane_intersection

p = np.array([
    [3, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0.]]);

result = plane_intersection(p, np.zeros(3), np.array([4.0,0.0,0.0]))
assert np.allclose(result, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

p = np.array([
    [2, 2, 2, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0.]]);


result = plane_intersection(p, np.zeros(3), np.array([2.0,2.0,2.0]))
assert np.allclose(result, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

result = plane_intersection(p, np.zeros(3), np.array([-1.0,-1.0,-1.0]))
assert np.allclose(result, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

p = np.array([
    [2, 2, 2, 0, 0, 0],
    [2, 2, 1, 0, 1, 0],
    [2, 2, -3, 0, 0, 0.]]);

result = plane_intersection(p, np.array([2.,2.,1.0]), np.array([0.0,0.0,1.0]))
assert np.allclose(result, np.array([2.0, 2.0, 1.0, 0.0, 1.0, 0.0]))

result = plane_intersection(p, np.array([2.,2.,1.0]), np.array([0.0,0.0,-1.0]))
assert result is not None and np.allclose(result, np.array([2.0, 2.0, 1.0, 0.0, 1.0, 0.0]))

p = np.array([
    [0., 0, -3, 0, 0, 0],
    [0., 0, 9, 0, 0, 0]])

result = plane_intersection(p, np.array([0.,1.,0.0]), np.array([1.0,1.0,1.0]))
assert np.allclose(result, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
result = plane_intersection(p, np.array([0.,1.,0.0]), -np.array([1.0,1.0,1.0]))
assert np.allclose(result, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))




p = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0]])
r = xy_plane_intersection(p, 0.5)[0]
assert np.isclose(r, 1.0)

p = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, 0.0]])
r = xy_plane_intersection(p, -0.5)[0]
assert np.isclose(r, 1.0)

p = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, 0.0],
    [2.0, -2.0, 0.0, 0.0]])
r = xy_plane_intersection(p, -1.5)[0]
assert np.isclose(r, 1.5)

p = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [4.0, 2.0, 0.0, 0.0]])
r = xy_plane_intersection(p, 1.75)[0]
assert np.isclose(r, 3.25)

p = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 4.0, 2.0, 0.0, 0.0, 0.0]])
y = xy_plane_intersection(p, 1.75)[1]
assert np.isclose(y, 3.25)

print('Tests run succesfully')






