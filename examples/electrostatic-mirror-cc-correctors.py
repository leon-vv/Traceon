import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G

import traceon.solver as S
import traceon.excitation as E 
import traceon.plotting as P
import traceon.tracing as T
from traceon_pro.interpolation import Field3DAxial

# Constants
BDY_XMAX = 1.5
BDY_XMIN = 0.5
BDY_ZMIN = -0.2
BDY_ZMAX = 1.8
BDY_THICKNESS = 0.2

EL_XMIN = 0.3
EL_XMAX = 1.0
EL_ZMIN = 0.5
EL_THICKNESS = 0.1

MIR_RADIUS = 0.3
MIRROR_Z_OFFSET = -0.2

MP_NUM_PIECES = 6
MP_IN_XMIN = 0.1
MP_IN_XMAX = 0.3
MP_THICKNESS = 0.1
MP_OUT_XMAX = 0.55
MP_SPACING = 0.05

# Boundary
boundary = (
    G.Path.line([BDY_XMAX, 0, BDY_ZMAX], [BDY_XMIN, 0, BDY_ZMAX])
    .line_to([BDY_XMIN, 0, BDY_ZMAX - BDY_THICKNESS])
    .line_to([BDY_XMAX - BDY_THICKNESS, 0, BDY_ZMAX - BDY_THICKNESS])
    .line_to([BDY_XMAX - BDY_THICKNESS, 0, BDY_ZMIN])
    .revolve_z()
)

# Electrode
electrode = (
    G.Path.rectangle_xz(EL_XMIN, EL_XMAX, EL_ZMIN, EL_ZMIN + EL_THICKNESS)
    .revolve_z()
)

# Mirror
mirror = G.Surface.disk_xy(0.0, 0.0, MIR_RADIUS).move(dz=MIRROR_Z_OFFSET)

# Multipole inner ring
multipole_inner = (
    G.Path.rectangle_xz(MP_IN_XMIN, MP_IN_XMAX, 0, MP_THICKNESS)
    .revolve_z()
)

# Multipole outer pieces
multipole_outer = []

# NOTE: range(0, N) does not work since rotation over 0 angle is not supported
for n in range(1, MP_NUM_PIECES + 1):
    angle = 2 * np.pi / MP_NUM_PIECES
    mpop = (
        G.Path.rectangle_xz(MP_IN_XMAX, MP_OUT_XMAX, 0, MP_THICKNESS)
        .rotate(Rz=n * angle)
        .revolve_z(angle)
        .move(
            dx=MP_SPACING * np.cos((n + 0.5) * angle),
            dy=MP_SPACING * np.sin((n + 0.5) * angle),
        )
    )
    multipole_outer.append(mpop)

    # Add sides to pieces
    for m in [0, 1]:
        mpos = (
            G.Surface.rectangle_xz(MP_IN_XMAX, MP_OUT_XMAX, 0, MP_THICKNESS)
            .rotate(Rz=(n + m) * angle)
            .move(
                dx=MP_SPACING * np.cos((n + 0.5) * angle),
                dy=MP_SPACING * np.sin((n + 0.5) * angle),
            )
        )
        multipole_outer.append(mpos)

# Combine geometries
boundary.name = "boundary"
electrode.name = "electrode"
mirror.name = "mirror"
multipole_inner.name = "multipole_inner"

geometry = boundary + electrode + mirror + multipole_inner

for mpop in multipole_outer:
    mpop.name = "multipole_outer"
    geometry = geometry + mpop

# Create and plot mesh
mesh = geometry.mesh(mesh_size_factor=10)

excitation = E.Excitation(mesh, E.Symmetry.THREE_D)

excitation.add_voltage(electrode=100, mirror=-110)
excitation.add_electrostatic_boundary('boundary')

field = S.solve_direct(excitation)
field_axial = Field3DAxial(field, -0.2, 1.8, 150)

z = np.linspace(-0.2, 1.8, 150)
pot = [field.potential_at_point([0.0, 0.0, z_]) for z_ in z]
pot_axial = [field_axial.potential_at_point([0.0, 0.0, z_]) for z_ in z]


# An instance of the tracer class allows us to easily find the trajectories of 
# electrons. Here we specify that the interpolated field should be used, and that
# the tracing should stop if the x,y value goes outside ±RADIUS/2 or the z value outside ±10 mm.
tracer = field_axial.get_tracer( [(-BDY_XMAX/2, BDY_XMAX/2), (-BDY_XMAX/2, BDY_XMAX/2),  (-0.2, 10)] )

# Start tracing from z=7mm
r_start = np.linspace(-BDY_XMAX/3, BDY_XMAX/3, 7)

# Initial velocity vector points downwards, with a 
# initial speed corresponding to 1000eV.
theta = 1e-4
velocity = T.velocity_vec(100, [np.sin(angle), 0, -np.cos(theta)])

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
plt.show()
