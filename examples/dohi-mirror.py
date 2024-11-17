import time

import numpy as np
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S
import traceon.plotting as P
from traceon.interpolation import FieldRadialAxial



try:
    from traceon_pro.interpolation import Field3DAxial
except ImportError:
    Field3DAxial = None

PLOT_GEOM = False
MSF = 30

#voltages
TUNING_VOLTAGE = 710.0126605741955
MIRROR_VOLTAGE = -1250

#displacements x y z plane of seperate components
ground_elec_displacement = (0.,0.,0.)
tuning_elec_displacement = (0.,0.,0.)
mirror_displacement = (0.0,0.,0.)


rmax = 1.0
margin = 0.3
extent = rmax-0.1

t = 0.15 # thickness
r = 0.075 # radius
st = 0.5  # spacer thickness


#create line geometries
mirror = G.Path.aperture(0.15, r, extent, z=t/2)
mirror.name = 'mirror'

mirror_line = G.Path.line([0., 0., 0.], [r, 0., 0.])
mirror_line.name = 'mirror'

lens = G.Path.aperture(0.15, r, extent, z=t + st + t/2)
lens.name = 'lens'

ground = G.Path.aperture(0.15, r, extent, z=t+st+t+st+t/2)
ground.name = 'ground'

#revolve around z axis and displace
mirror = mirror.revolve_z()
mirror_line = mirror_line.revolve_z()
lens = lens.revolve_z()
ground = ground.revolve_z()

#displace electrodes
ground = ground.move(*ground_elec_displacement)
lens = lens.move(*tuning_elec_displacement)
mirror = mirror.move(*mirror_displacement)
mirror_line = mirror_line.move(*mirror_displacement)

#create geometry
geom = mirror+mirror_line+lens+ground

#make mesh
mesh =  geom.mesh(mesh_size_factor=MSF)

if PLOT_GEOM:
    P.plot_mesh(mesh, ground='green', mirror='red', lens='blue', show_normals=True)

excitation = E.Excitation(mesh, E.Symmetry.THREE_D)

# Apply the correct voltages. Set the ground electrode to zero.
excitation.add_voltage(ground=0., lens=TUNING_VOLTAGE, mirror=MIRROR_VOLTAGE)

# Use the Boundary Element Method (BEM) to calculate the surface charges,
# the surface charges gives rise to a electrostatic field.
field = S.solve_direct(excitation)

tracer = field.get_tracer( [(-r/2, r/2), (-r/2, r/2), (-7, 15.1)] )

angle = 0.5e-8
z0 = 15

# Initial velocity vector points downwards, with a 
# initial speed corresponding to 1000eV.
start_pos = np.array([0.0, 0.0, z0])
start_vel = T.velocity_vec_xz_plane(1000, angle)

print('Starting trace...')
st = time.time()
_, pos_derivs = tracer(start_pos, start_vel)
print(f'Trace took {(time.time()-st)*1000:.1f} ms')
intersection = T.xy_plane_intersection(pos_derivs, z0)

print("\n (x,y,z)-displacements:")
print(f"ground: {ground_elec_displacement}")
print(f"lens: {tuning_elec_displacement}")
print(f"mirror: {mirror_displacement} \n")

print(f"Relative error: {np.abs(intersection[0])}")