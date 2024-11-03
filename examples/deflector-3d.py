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
def round_electrode(z0, name):
    
    path = G.Path.line([RADIUS, 0.0, z0], [RADIUS, 0.0, z0+THICKNESS])\
        .line_to([RADIUS+ELECTRODE_WIDTH, 0.0, z0+THICKNESS])\
        .line_to([RADIUS+ELECTRODE_WIDTH, 0.0, z0])\
        .close()

    surf = path.revolve_z()
    surf.name = name
    
    return surf

# Create a simple electrode consisting of a extruded rectangle.
# A deflector consists of two rectangle electrodes 'facing' each other.
def rectangle_electrode(x, z, name):

    p0 = [x, -LENGTH_DEFLECTORS/2, z]
    p1 = [x+THICKNESS, +LENGTH_DEFLECTORS/2, z+THICKNESS]
    
    box = G.Surface.box(p0, p1)
    box.name = name
    
    return box

electrode_top = round_electrode(z0, 'ground')
defl_pos = rectangle_electrode(RADIUS, z0+THICKNESS+SPACING, 'deflector_positive')    
defl_neg = rectangle_electrode(-RADIUS-THICKNESS, z0+THICKNESS+SPACING, 'deflector_negative')
electrode_bottom = round_electrode(z0+THICKNESS+SPACING+THICKNESS+SPACING, 'ground')

electrode_mesh = (electrode_top + electrode_bottom).mesh(mesh_size_factor=24)
defl_mesh = (defl_pos + defl_neg).mesh(mesh_size_factor=3)

mesh = electrode_mesh + defl_mesh

# Show the generated triangle mesh.
P.plot_mesh(mesh, ground='green', deflector_positive='red', deflector_negative='blue', show_normals=True)

excitation = E.Excitation(mesh, E.Symmetry.THREE_D)

# Apply the correct voltages. Here we set one deflector electrode to 5V and
# the other electrode to -5V.
excitation.add_voltage(ground=0.0, deflector_positive=5, deflector_negative=-5)

# Use the Boundary Element Method (BEM) to calculate the surface charges,
# the surface charges gives rise to a electrostatic field.
field = S.solve_direct(excitation)

# An instance of the tracer class allows us to easily find the trajectories of 
# electrons.  Here we specify that the tracing should stop if the x,y values
# go outside ±RADIUS/2 or the z value outside ±7 mm.
tracer = field.get_tracer( [(-RADIUS/2, RADIUS/2), (-RADIUS/2, RADIUS/2), (-7, 7)] )

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








