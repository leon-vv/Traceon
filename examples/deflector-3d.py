import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.plotting as P
import traceon.tracing as T

# Dimensions used for the deflector.
THICKNESS = 0.5
SPACING = 0.5
RADIUS = 0.15
ELECTRODE_WIDTH = 8*RADIUS
LENGTH_DEFLECTORS = 7*RADIUS


z0 = -THICKNESS - SPACING - THICKNESS/2

# Create an aperture electrode.
# This is a 'disk' or round electrode with a hole in the middle.
# The top and bottom electrodes are round grounded electrodes
# which shield the microscope from the deflector fields.
def round_electrode(geom, z0, name):
    
    points = [ [RADIUS, 0.0, z0], [RADIUS, 0.0, z0+THICKNESS],
               [RADIUS+ELECTRODE_WIDTH, 0.0, z0+THICKNESS], [RADIUS+ELECTRODE_WIDTH, 0.0, z0] ]

    points = [geom.add_point(p) for p in points]
    
    l1 = geom.add_line(points[0], points[1])
    l2 = geom.add_line(points[1], points[2])
    l3 = geom.add_line(points[2], points[3])
    l4 = geom.add_line(points[3], points[0])

    revolved = G.revolve_around_optical_axis(geom, [l1, l2, l3, l4])
    geom.add_physical(revolved, 'ground')

# Create a simple electrode consisting of a extruded rectangle.
# A deflector consists of two rectangle electrodes 'facing' each other.
def rectangle_electrode(geom, x, z, name):
    
    points = [ [x, -LENGTH_DEFLECTORS/2, z], [x, -LENGTH_DEFLECTORS/2, z+THICKNESS],
               [x+THICKNESS, -LENGTH_DEFLECTORS/2, z+THICKNESS], [x+THICKNESS, -LENGTH_DEFLECTORS/2, z] ]
    
    poly = geom.add_polygon(points)

    top, extruded, lateral = geom.extrude(poly, [0.0, LENGTH_DEFLECTORS, 0.0])
    geom.add_physical(lateral, name)
    geom.add_physical(top, name)
    geom.add_physical(poly, name)


# Create the actual geometry using the utility functions above.
with G.Geometry(G.Symmetry.THREE_D, size_from_distance=True) as geom:
    round_electrode(geom, z0, 'ground')
     
    rectangle_electrode(geom, RADIUS, z0+THICKNESS+SPACING, 'deflector_positive')    
    rectangle_electrode(geom, -RADIUS-THICKNESS, z0+THICKNESS+SPACING, 'deflector_negative')
    
    round_electrode(geom, z0+THICKNESS+SPACING+THICKNESS+SPACING, 'ground')
    
    # The higher the mesh factor, the more triangles are used. This improves
    # accuracy at the expense of computation time.
    geom.set_mesh_size_factor(250)
    
    mesh = geom.generate_triangle_mesh()

# Show the generated triangle mesh.
P.plot_mesh(mesh, ground='green', deflector_positive='red', deflector_negative='blue')

excitation = E.Excitation(mesh)

# Apply the correct voltages. Here we set one deflector electrode to 5V and
# the other electrode to -5V.
excitation.add_voltage(ground=0.0, deflector_positive=5, deflector_negative=-5)

# Use the Boundary Element Method (BEM) to calculate the surface charges,
# the surface charges gives rise to a electrostatic field.
field = S.solve_bem(excitation)

# But using an integration over the surface charges to calculate the electron
# trajectories is inherently slow. Instead, use an interpolation technique
# in which we use the multipole coefficients of the potential along the potential axis.
# The complicated mathematics are all abstracted away from the user.
field_axial = field.axial_derivative_interpolation(-2, 2, 200)

# An instance of the tracer class allows us to easily find the trajectories of 
# electrons. Here we specify that the interpolated field should be used, and that
# the tracing should stop if the x,y value goes outside ±RADIUS/2 or the z value outside ±7 mm.
tracer = T.Tracer(field_axial, ((-RADIUS/2, RADIUS/2), (-RADIUS/2, RADIUS/2), (-7, 7)))

r_start = np.linspace(-RADIUS/8, RADIUS/8, 5)

# Initial velocity vector points downwards, with a 
# initial speed corresponding to 1000eV.
velocity = T.velocity_vec(1000, [0, 0, -1])

plt.figure()
plt.title('Electron traces')

for i, r0 in enumerate(r_start):
    print(f'Tracing electron {i+1}/{len(r_start)}...')
    _, positions = tracer(np.array([r0, 0.0, 5]), velocity)
    # Plot the z position of the electrons vs the r position.
    # C0 produces the default matplotlib color (a shade of blue).
    plt.plot(positions[:, 0], positions[:, 2], color='C0')

plt.xlabel('x (mm)')
plt.ylabel('z (mm)')
plt.show()








