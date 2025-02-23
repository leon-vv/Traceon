import time
import numpy as np
import matplotlib.pyplot as plt

import traceon as T

try:
    import traceon_pro.solver as S
except ImportError:
    S = None

PLOTTING = False
MSF = 20

#voltages
TUNING_VOLTAGE = 507.8#535.03
MIRROR_VOLTAGE = -1250

#displacements in (x y z) coords of seperate components
ground_elec_displacement = (0.,0.,0.)
tuning_elec_displacement = (0.,0.,0.)
mirror_displacement = (0.,0.,0.)


rmax = 1.0
margin = 0.3
extent = rmax-0.1

t = 0.15 # thickness
r = 0.075 # radius
st = 0.5  # spacer thickness

boundary = T.Path.line([extent, 0.,0.], [extent, 0., 1.6])
boundary.name = 'boundary'

#create line geometries
mirror = T.Path.aperture(0.15, r, extent, z=t/2)
mirror.name = 'mirror'

mirror_line = T.Path.line([0., 0., 0.], [r, 0., 0.])
mirror_line.name = 'mirror'

lens = T.Path.aperture(0.15, r, extent, z=t + st + t/2)
lens.name = 'lens'

ground = T.Path.aperture(0.15, r, extent, z=t+st+t+st+t/2)
ground.name = 'ground'

#revolve around z axis and displace
mirror = mirror.revolve_z()
mirror_line = mirror_line.revolve_z()
lens = lens.revolve_z()
ground = ground.revolve_z()
boundary = boundary.revolve_z()
#displace electrodes
ground = ground.move(*ground_elec_displacement)
lens = lens.move(*tuning_elec_displacement)
mirror = mirror.move(*mirror_displacement)
mirror_line = mirror_line.move(*mirror_displacement)

#create geometry
geom = mirror+mirror_line+lens+ground+boundary

#make mesh
mesh =  geom.mesh(mesh_size_factor=MSF)

if PLOTTING:
    T.plot_mesh(mesh, ground='green', mirror='red', lens='blue', boundary = 'grey', show_normals=True)
    T.show()

excitation = T.Excitation(mesh, T.Symmetry.THREE_D)

# Apply the correct voltages. Set the ground electrode to zero.
excitation.add_voltage(ground=0., lens=TUNING_VOLTAGE, mirror=MIRROR_VOLTAGE)
excitation.add_electrostatic_boundary('boundary')
# Use the Boundary Element Method (BEM) to calculate the surface charges,
# the surface charges gives rise to a electrostatic field.
assert S is not None, ("The 'traceon_pro' package is not installed or not found. "
        "Traceon Pro is required to solve 3D geometries.\n"
        "For more information, visit: https://www.traceon.org")
field = S.solve_direct(excitation)

tracer = field.get_tracer( [(-r/2, r/2), (-r/2, r/2), (-7, 15.1)] )

#Trace electrons
# Initial velocity vector points downwards, with a 
# initial speed corresponding to 1000eV.
z0 = 15
start_xy_1 = (1e-4,0)
start_xy_2 = (-1e-4,0)
starting_positions = np.array([[*start_xy_1, z0], [*start_xy_2, z0]])
start_vel = T.velocity_vec_xz_plane(1000, angle = 0)

traces = []

for i, p in enumerate(starting_positions):
    print(f'Starting trace {i}...')
    st = time.time()
    _, positions = tracer(p, start_vel)
    traces.append(positions)
    plt.plot(positions[:,0], positions[:,2])
    print(f'Trace {i} took {(time.time()-st)*1000:.1f} ms')

if PLOTTING:
    plt.show()


#find focus
focus_loc = F.focus_position(traces)



print("\n(x,y,z)-displacements: \n")
print(f"ground: {ground_elec_displacement}")
print(f"lens: {tuning_elec_displacement}")
print(f"mirror: {mirror_displacement} \n")

print(f"Focus of perfectly aligned system: {[0,0,15]}")
print(f"Focus of displaced system: {focus_loc}")
