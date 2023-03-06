import numpy as np

from traceon.tracing import plane_intersection

p = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0]])
r = plane_intersection(p, 0.5)[0]
assert np.isclose(r, 1.0)

p = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, 0.0]])
r = plane_intersection(p, -0.5)[0]
assert np.isclose(r, 1.0)

p = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, 0.0],
    [2.0, -2.0, 0.0, 0.0]])
r = plane_intersection(p, -1.5)[0]
assert np.isclose(r, 1.5)

p = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [4.0, 2.0, 0.0, 0.0]])
r = plane_intersection(p, 1.75)[0]
assert np.isclose(r, 3.25)

print('Tests run succesfully')






