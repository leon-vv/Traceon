import numpy as np
from traceon.backend import position_and_jacobian_radial

line = np.array([
    [0.0, 0.0, 0.0],
    [0.25, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [1.0, 0.0, 0.0]])

_, pos = position_and_jacobian_radial(-1, *line)
assert np.all(np.isclose(pos, line[0, :2]))
_, pos = position_and_jacobian_radial(-1 + 2/3, *line)
assert np.all(np.isclose(pos, line[1, :2]))
_, pos = position_and_jacobian_radial(-1 + 4/3, *line)
assert np.all(np.isclose(pos, line[2, :2]))
_, pos = position_and_jacobian_radial(1, *line)
assert np.all(np.isclose(pos, line[3, :2]))
