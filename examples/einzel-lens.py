import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.plotting as P
import traceon.tracing as T

# Dimensions of the einzel lens.
THICKNESS = 0.5
SPACING = 0.5
RADIUS = 0.15

# Start value of z chosen such that the middle of the einzel
# lens is at z = 0mm.
z0 = -THICKNESS - SPACING - THICKNESS/2

with G.MEMSStack(z0=z0, zmin=-1, zmax=2, size_from_distance=True) as geom:
    
    # Einzel lens consists of three electrodes: 
    # a lens electrode sandwiched between two ground electrodes.
    geom.add_electrode(RADIUS, THICKNESS, 'ground')
    geom.add_spacer(THICKNESS)
    geom.add_electrode(RADIUS, THICKNESS, 'lens')
    geom.add_spacer(THICKNESS)
    geom.add_electrode(RADIUS, THICKNESS, 'ground')
    
    geom.set_mesh_size_factor(80)
    
    # Actually generate the mesh, which takes the boundaries in the
    # geometry and produces many line elements.
    mesh = geom.generate_mesh()

# Show the generated mesh, with the given electrode colors.
P.plot_mesh(mesh, ground='green', lens='blue')

excitation = E.Excitation(mesh)

# Excite the geometry, put ground at 0V and the lens electrode at 1000V.
excitation.add_voltage(ground=0.0, lens=1000)
excitation.add_boundary('boundary')

# Use the Boundary Element Method (BEM) to calculate the surface charges,
# the surface charges gives rise to a electrostatic field.
field = S.solve_bem(excitation)

# But using an integration over the surface charges to calculate the electron
# trajectories is inherently slow. Instead, use an interpolation technique
# in which we use the derivatives of the potential along the potential axis.
# The complicated mathematics are all abstracted away from the user.
field_axial = field.axial_derivative_interpolation(-1, 2, 150)

# Plot the potential along the optical axis to show that the interpolated
# potential is very close to the potential found by an integration over the
# surface charge.
z = np.linspace(-1, 2, 150)
pot = [field.potential_at_point(np.array([0.0, z_])) for z_ in z]
pot_axial = [field_axial.potential_at_point(np.array([0.0, z_])) for z_ in z]

plt.title('Potential along axis')
plt.plot(z, pot, label='Surface charge integration')
plt.plot(z, pot_axial, linestyle='dashed', label='Interpolation')
plt.xlabel('z (mm)')
plt.ylabel('Potential (V)')
plt.legend()
plt.show()

# An instance of the tracer class allows us to easily find the trajectories of 
# electrons. Here we specify that the interpolated field should be used, and that
# the tracing should stop if the r value goes outside ±RADIUS/2 or the z value outside ±10 mm.
tracer = T.Tracer(field_axial, ((-RADIUS/2, RADIUS/2), (-10, 10)) )

# Start tracing from z=7mm
r_start = np.linspace(-RADIUS/5, RADIUS/5, 7)

# Initial velocity vector points downwards, with a 
# initial speed corresponding to 1000eV.
velocity = T.velocity_vec(1000, [0, -1])

plt.figure()
plt.title('Electron traces')

for i, r0 in enumerate(r_start):
    print(f'Tracing electron {i+1}/{len(r_start)}...')
    _, positions = tracer(np.array([r0, 5]), velocity)
    # Plot the z position of the electrons vs the r position.
    # C0 produces the default matplotlib color (a shade of blue).
    plt.plot(positions[:, 0], positions[:, 1], color='C0')

plt.xlabel('r (mm)')
plt.ylabel('z (mm)')
plt.show()





