import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.plotting as P
import traceon.tracing as T

#dimensions
DIAMETER = 0.15
RADIUS = DIAMETER/2
THICKNESS = 0.2
SPACING = 0.5
ELECTRODE_WIDTH = 5*DIAMETER

#voltages
TUNING_VOLTAGE = -1200
MIRROR_VOLTAGE = -5529.8

#displacements x y z plane
ground_elec_displacement = (0.03,0.,0.)
tuning_elec_displacement = (0.,0.04,0.)
mirror_displacement = (0.,0.,0.)


#set middle of the lens to 0
z0 = 0

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

# Create an mirror.
# This is a 'disk' with a hole in the middle that only extends halfway.
def round_mirror(z0, name):
    path = G.Path.line([0.0, 0.0, -THICKNESS+z0], [0.0, 0.0, z0])\
        .line_to([RADIUS, 0.0, z0])\
        .line_to([RADIUS, 0.0, z0+THICKNESS])\
        .line_to([RADIUS+ELECTRODE_WIDTH, 0.0, z0+THICKNESS])\
        .line_to([RADIUS+ELECTRODE_WIDTH, 0.0, z0-THICKNESS])\
        .close()

    surf = path.revolve_z()
    surf.name = name
    return surf

#initialize electrodes
ground_elec = round_electrode(z0+SPACING, 'ground_electrode')
tuning_elec = round_electrode(z0, 'tuning_electrode')
mirror_elec = round_mirror(-SPACING+z0, 'mirror_electrode')

#displace electrodes
ground_electrode = ground_elec.move(*ground_elec_displacement)
tuning_electrode = tuning_elec.move(*tuning_elec_displacement)
mirror = mirror_elec.move(*mirror_displacement)


#create meshes
mirror_mesh = (mirror).mesh(mesh_size_factor=80)
meshes = (ground_electrode + tuning_electrode).mesh(mesh_size_factor=50)

#add meshes
mesh = mirror_mesh+meshes

# Show the generated triangle mesh.
P.plot_mesh(mesh, ground_electrode='green', mirror_electrode='red', tuning_electrode='blue', show_normals=True)

excitation = E.Excitation(mesh, E.Symmetry.THREE_D)

# Apply the correct voltages. Set the ground electrode to zero.
excitation.add_voltage(ground_electrode=0., tuning_electrode=TUNING_VOLTAGE, mirror=MIRROR_VOLTAGE)

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

