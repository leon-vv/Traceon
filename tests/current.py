import numpy as np
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as S
import traceon.plotting as P

with G.Geometry(G.Symmetry.RADIAL) as geom:
    points = [ [3.0, 4.0], [4.0, 4.0], [4.0, 5.0], [3.0, 5.0] ]
    p = geom.add_polygon(points)
    geom.add_physical(p.surface, 'coil')
    mesh = geom.generate_triangle_mesh(True)

#P.plot_mesh(mesh, coil='blue')

exc = E.Excitation(mesh)
exc.add_current(coil=1)

field = S.solve_bem(exc)

z = np.linspace(-10, 10, 250)
f = np.array([field.current_field_at_point(np.array([0., z_])) for z_ in z])

plt.plot(z, f[:, 0], label='Hr')
plt.plot(z, f[:, 1], label='Hz')
plt.legend()
plt.show()

