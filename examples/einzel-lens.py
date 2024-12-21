import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.plotting as P
import traceon.tracing as T
from traceon.field import FieldRadialAxial

# Dimensions of the einzel lens.
THICKNESS = 0.5
SPACING = 0.5
RADIUS = 0.15

# Start value of z chosen such that the middle of the einzel
# lens is at z = 0mm.
z0 = -THICKNESS - SPACING - THICKNESS/2

boundary = G.Path.line([0., 0., 1.75],  [2.0, 0., 1.75])\
    .extend_with_line([2.0, 0., -1.75]).extend_with_line([0., 0., -1.75])


margin_right = 0.1
extent = 2.0 - margin_right

bottom = G.Path.aperture(THICKNESS, RADIUS, extent, -THICKNESS - SPACING)
middle = G.Path.aperture(THICKNESS, RADIUS, extent)
top = G.Path.aperture(THICKNESS, RADIUS, extent, THICKNESS + SPACING)
    
boundary.name = 'boundary'
bottom.name = 'ground'
middle.name = 'lens'
top.name = 'ground'

mesh = (boundary + bottom + middle + top).mesh(mesh_size_factor=45)
 
excitation = E.Excitation(mesh, E.Symmetry.RADIAL)

# Excite the geometry, put ground at 0V and the lens electrode at 1000V.
excitation.add_voltage(ground=0.0, lens=1800)
excitation.add_electrostatic_boundary('boundary')

# Use the Boundary Element Method (BEM) to calculate the surface charges,
# the surface charges gives rise to a electrostatic field.
field = S.solve_direct(excitation)

# But using an integration over the surface charges to calculate the electron
# trajectories is inherently slow. Instead, use an interpolation technique
# in which we use the derivatives of the potential along the potential axis.
# The complicated mathematics are all abstracted away from the user.
field_axial = FieldRadialAxial(field, -1.5, 1.5, 150)

# Plot the potential along the optical axis to show that the interpolated
# potential is very close to the potential found by an integration over the
# surface charge.
z = np.linspace(-1.5, 1.5, 150)
pot = [field.potential_at_point([0.0, 0.0, z_]) for z_ in z]
pot_axial = [field_axial.potential_at_point([0.0, 0.0, z_]) for z_ in z]



# An instance of the tracer class allows us to easily find the trajectories of 
# electrons. Here we specify that the interpolated field should be used, and that
# the tracing should stop if the x,y value goes outside ±RADIUS/2 or the z value outside ±10 mm.
tracer = field_axial.get_tracer( [(-RADIUS/2, RADIUS/2), (-RADIUS/2,RADIUS/2),  (-10, 10)] )

# Start tracing from z=7mm
r_start = np.linspace(-RADIUS/3, RADIUS/3, 7)

# Initial velocity vector points downwards, with a 
# initial speed corresponding to 1000eV.
velocity = T.velocity_vec(1000, [0, 0, -1])

trajectories = []

for i, r0 in enumerate(r_start):
    print(f'Tracing electron {i+1}/{len(r_start)}...')
    _, positions = tracer(np.array([r0, 0, 5]), velocity)
    trajectories.append(positions)


# Plotting

plt.title('Potential along axis')
plt.plot(z, pot, label='Surface charge integration')
plt.plot(z, pot_axial, linestyle='dashed', label='Interpolation')
plt.xlabel('z (mm)')
plt.ylabel('Potential (V)')
plt.legend()

plt.figure()
plt.title('Electron traces')

for positions in trajectories:
    plt.plot(positions[:, 0], positions[:, 2], color='C0')

plt.xlabel('r (mm)')
plt.ylabel('z (mm)')
plt.show()

P.new_figure()
P.plot_mesh(mesh, ground='green', lens='blue', boundary='purple')
P.plot_trajectories(trajectories, xmin=0, zmin=-1.7, zmax=1.6)
surface = G.Surface.rectangle_xz(0.0, RADIUS*0.9, -1.75, 1.75)
P.plot_equipotential_lines(field_axial, surface, N0=250, N1=75)
P.show()
