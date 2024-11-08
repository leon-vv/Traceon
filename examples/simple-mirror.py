import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.plotting as P
import traceon.tracing as T
from traceon.interpolation import FieldRadialAxial


boundary = G.Path.line([0, 0, -1], [2, 0, -1]).line_to([2, 0, 1]).line_to([0.3, 0., 1])
mirror = G.Path.line([0., 0., 0.], [1., 0., 0.])

boundary.name = 'boundary'
mirror.name = 'mirror'

mesh = (boundary + mirror).mesh(mesh_size_factor=45)

# Show the generated mesh, with the given electrode colors.
P.plot_mesh(mesh, ground='green', lens='blue', show_normals=True)

excitation = E.Excitation(mesh, E.Symmetry.RADIAL)

excitation.add_voltage(mirror=-110, boundary=0.0)

field = S.solve_direct(excitation)
field_axial = FieldRadialAxial(field, 0.02, 4, 600)

#NOTE: discrepancy between interpolation and integration near z=0
z = np.linspace(0.02, 4, 600)
pot = [field.potential_at_point(np.array([0.0, z_])) for z_ in z]
pot_axial = [field_axial.potential_at_point(np.array([0.0,z_])) for z_ in z]

plt.title('Potential along axis')
plt.plot(z, pot, label='Surface charge integration')
plt.plot(z, pot_axial, linestyle='dashed', label='Interpolation')
plt.xlabel('z (mm)')
plt.ylabel('Potential (V)')
plt.legend()
plt.show()



tracer = field_axial.get_tracer( [(-1, 1), (-1,1),  (-10, 10)] )

r_start = np.linspace(-1/5, 1/5, 7)

velocity = T.velocity_vec(100, [0, -1])

plt.figure()
plt.title('Electron traces')

for i, r0 in enumerate(r_start):
    print(f'Tracing electron {i+1}/{len(r_start)}...')
    _, positions = tracer(np.array([r0, 5]), velocity)
    # Plot the z position of the electrons vs the r position.
    # C0 produces the default matplotlib color (a shade of blue).
    plt.plot(positions[:, 0], positions[:, 2], color='C0')

plt.xlabel('r (mm)')
plt.ylabel('z (mm)')
plt.show()