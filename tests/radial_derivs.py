import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import traceon.backend as backend

rs = 1 # Source point
zs = 0

r = np.linspace(1.1, 2)
pot = [backend.potential_radial_ring(r_, zs, rs, zs) for r_ in r]
plt.plot(r, pot)
deriv = [backend.dr1_potential_radial_ring(r_, zs, rs, zs) for r_ in r]
plt.plot(r, deriv)
plt.plot(r, CubicSpline(r, pot)(r, 1), linestyle='dashed', color='black')
plt.show()


plt.figure()
z = np.linspace(0.1, 1)
pot = [backend.potential_radial_ring(rs, z_, rs, zs) for z_ in z]
plt.plot(z, pot)
deriv = [backend.dz1_potential_radial_ring(rs, z_, rs, zs) for z_ in z]
plt.plot(z, deriv)
plt.plot(z, CubicSpline(z, pot)(z, 1), linestyle='dashed', color='black')
plt.show()


