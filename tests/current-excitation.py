import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os.path as path

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S

with G.Geometry(G.Symmetry.RADIAL) as geom:
    points = [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
    poly = geom.add_polygon(points)
    geom.add_physical(poly, 'coil')
    #geom.set_mesh_size_factor(3)
    mesh = geom.generate_triangle_mesh(True)

P.plot_mesh(mesh)

exc = E.Excitation(mesh)
exc.add_current(coil=5)

field = S.solve_bem(exc)

z = np.linspace(-4, 6, 500)
r = 0.0

mu_0 = 1.25663706212e-03

axial_field = np.array([field.current_field_at_point(np.array([r, z_]))[1] for z_ in z])

file_ = path.join(path.dirname(__file__), 'axial magnetic field.txt')
reference_data = np.loadtxt(file_, delimiter=',')
reference = CubicSpline(reference_data[:, 0], reference_data[:, 1] * mu_0 * 1e-3)

plt.plot(z, reference(z))
plt.plot(z, mu_0/np.pi * axial_field)
Bring = 2.0943951023931957e-3
plt.axhline(Bring, color='green')
plt.show()

plt.figure()
plt.plot(z, np.abs( ( mu_0/np.pi * axial_field)/reference(z) - 1))
plt.yscale('log')
plt.show()


