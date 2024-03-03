import numpy as np
import matplotlib.pyplot as plt

import traceon.backend as B

r_ring = 55
z_ring = 0

z = np.linspace(-5, 5, 250)

# http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html
field_correct = r_ring**2 / (2*((z-z_ring)**2 + r_ring**2)**(3/2))

field_z = np.array([B.current_field_radial_ring(0., z_, r_ring, z_ring)[1] for z_ in z])

plt.plot(z, field_z)
plt.plot(z, field_z, 'k--')
plt.show()
