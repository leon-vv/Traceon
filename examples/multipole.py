# Example illustrating the novel boundary sweeping functions (extrude/revolve) introduced in v0.8.0
# at the hand of a simple simulation of a multipole

import numpy as np

import traceon.geometry as G

import traceon.solver as S
import traceon.excitation as E 
import traceon.plotting as P
import traceon.tracing as T


# multipole configuration
ORDER = 2 # A second-order multipole is a quadrupole
THICKNESS = 0.1
IN_RADIUS = 0.3
OUT_RADIUS = 0.5
SPACING = 0.05

# boundary configuration
B_RADIUS = 1.
B_HEIGHT = 2.


# Before v0.8.0
# 1. Initialize a path
# 2. Revolve it
# 3. Add side pieces

multipole_pieces = []

for n in range(1, 2 * ORDER + 1):
    angle = 2 * np.pi / (2 * ORDER)
    piece = (
        G.Path.rectangle_xz(IN_RADIUS, OUT_RADIUS, -THICKNESS/2, THICKNESS/2) 
        .rotate(Rz=n * angle)
        .revolve_z(angle)
        .move(
            dx=SPACING * np.cos((n + 0.5) * angle),
            dy=SPACING * np.sin((n + 0.5) * angle),
        )
    )
    piece.name = 'pos' if n % 2 == 0 else 'neg'
    multipole_pieces.append(piece)

    # We have to add the sides to the pieces of the multipole manually
    for m in [0, 1]:
        piece_side = (
            G.Surface.rectangle_xz(IN_RADIUS, OUT_RADIUS, -THICKNESS/2, THICKNESS/2)
            .rotate(Rz=(n + m) * angle)
            .move(
                dx=SPACING * np.cos((n + 0.5) * angle),
                dy=SPACING * np.sin((n + 0.5) * angle),
            )
        )
        piece_side.name = piece.name
        multipole_pieces.append(piece_side)

multipole = G.SurfaceCollection(multipole_pieces)

# To add a boundary we have to:
# 1. create a path and revolve it
# 2. manual add a disk to close it at the bottom and top
boundary = (
    G.Path.line([1, 0, -1], [1, 0, 1]).revolve_z()
    + G.Surface.disk_xy(0,0,B_RADIUS).move(dz=-B_HEIGHT/2)
    + G.Surface.disk_xy(0,0,B_RADIUS).move(dz=B_HEIGHT/2)
)
boundary.name = 'boundary'

# Uncomment to compare to new implementation
# mesh = (multipole + boundary).mesh(mesh_size=0.08)
# P.plot_mesh(mesh, pos='red', neg='blue', boundary='green')
# P.show()

#### After v.0.8.0 ###
# 1. Initialize a surface.
# 2. Revolve its boundary, cap it off on the other side automatically.

multipole_pieces = []

for n in range(1, 2 * ORDER + 1):
    angle = 2 * np.pi / ( 2 * ORDER)
    piece = (
        G.Surface.rectangle_xz(IN_RADIUS, OUT_RADIUS, -THICKNESS/2, THICKNESS/2) # Initialize a surface
        .rotate(Rz= n* angle)
        .revolve_boundary_z(angle, add_cap=True)
        .move(
            dx=SPACING * np.cos((n + 0.5) * angle),
            dy=SPACING * np.sin((n + 0.5) * angle),
        )
    )
    for surface in piece.surfaces:
        surface.name = 'pos' if n % 2 == 0 else 'neg'
    
    multipole_pieces.extend(piece.surfaces)

multipole = G.SurfaceCollection(multipole_pieces)

# creating a boundary can be done in one line with the extrude_boundary function:
boundary = G.Surface.disk_xy(0, 0, B_RADIUS).move(dz=-B_HEIGHT/2).extrude_boundary([0,0,B_HEIGHT], add_cap=True)
boundary.name='boundary'

# create mesh
mesh = (multipole + boundary).mesh(mesh_size=0.07)

# add excitations
excitation = E.Excitation(mesh, E.Symmetry.THREE_D)
excitation.add_voltage(pos=1, neg=-1, boundary=0)

# calculate field
field = S.solve_direct(excitation)

# plot mesh and equipotential lines
field_xy = G.Surface.disk_xy(0,0,1)
field_xz = G.Surface.rectangle_xz(-1,1,-1, 1) 
field_yz = G.Surface.rectangle_yz(-1,1,-1,1)

# initialize tracer and electron trajectories
tracer = field.get_tracer([(-B_RADIUS/2, B_RADIUS/2), (-B_RADIUS/2,B_RADIUS/2),  (-B_HEIGHT/2, B_HEIGHT/2)])

beam_radius = IN_RADIUS / 3
r_start = np.linspace(-beam_radius, beam_radius, 8)
x, y = np.meshgrid(r_start, r_start)
x_start, y_start = x[x**2 + y**2 <= beam_radius**2], y[x**2 + y**2 <= beam_radius**2]
z_start = B_HEIGHT / 2

trajectories = []
start_velocity = T.velocity_vec(1, [0, 0, -1])

# trace the electrons through the field
for i, (x, y) in enumerate(zip(x_start, y_start)):
    print(f'Tracing electron {i+1}/{len(x_start)}...')
    start_point = np.array([x, y, z_start])
    _, trace = tracer(start_point, start_velocity)
    trajectories.append(trace)

# plotting
P.plot_mesh(mesh, pos='red', neg='blue', boundary='green')
P.plot_equipotential_lines(field, surface=field_xy, N0=100, N1=100)
P.plot_equipotential_lines(field, surface=field_xz, N0=100, N1=100)
P.plot_equipotential_lines(field, surface=field_yz, N0=100, N1=100)
P.plot_trajectories(trajectories, line_width=3)
P.show()