# Simulation of a stimgator using the boundary sweeping functions introduced in v0.8.0.
# Shows how particles of general mass and charge can be traced since v0.8.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_p, e

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E 
import traceon.plotting as P
import traceon.tracing as T

# Multipole configuration
ORDER = 2 # A second-order multipole (quadrupole) acts a stigmator
THICKNESS = 0.1
IN_RADIUS = 0.3
OUT_RADIUS = 0.5
SPACING = 0.05

# Boundary configuration
B_RADIUS = 1.
B_HEIGHT = 1.5

multipole_pieces = []

for n in range(1, 2 * ORDER + 1):
    angle = 2 * np.pi / ( 2 * ORDER)
    piece = (
        G.Surface.rectangle_xz(IN_RADIUS, OUT_RADIUS, -THICKNESS/2, THICKNESS/2) 
        .rotate(Rz= n* angle)
        .revolve_boundary_z(angle, enclose=True) # Revolve the boundary of the side piece and enclose on the other side
        .move(
            dx=SPACING * np.cos((n + 0.5) * angle),
            dy=SPACING * np.sin((n + 0.5) * angle)))

    piece.name = 'positive_electrode' if n % 2 == 0 else 'negative_electrode'
    multipole_pieces.extend(piece.surfaces)

multipole = G.SurfaceCollection(multipole_pieces)

# Creating a boundary can be done in one line with the extrude_boundary function:
boundary = (
    G.Surface.disk_xy(0, 0, B_RADIUS).move(dz=-B_HEIGHT/2).extrude_boundary([0,0,B_HEIGHT], enclose=False) 
    + G.Path.line([IN_RADIUS, 0, B_HEIGHT/2], [B_RADIUS, 0, B_HEIGHT/2]).revolve_z())
boundary.name = 'boundary'

# Create and plot mesh
mesh = (multipole.mesh(mesh_size=0.04) + boundary.mesh(mesh_size=0.1))

P.plot_mesh(mesh, positive_electrode='red', negative_electrode='blue', boundary='green')
P.show()

# Add excitations
excitation = E.Excitation(mesh, E.Symmetry.THREE_D)
excitation.add_voltage(positive_electrode=1, negative_electrode=-1, boundary=0)

# Calculate field
field = S.solve_direct(excitation)

# Plot mesh and equipotential lines
margin = 0.02
x_ext, y_ext, z_ext = B_RADIUS - margin, B_RADIUS - margin, B_HEIGHT/2 - margin
field_xy = G.Surface.disk_xy(0,0, x_ext)
field_xz = G.Surface.rectangle_xz(-x_ext, x_ext, -z_ext, z_ext) 
field_yz = G.Surface.rectangle_yz(-y_ext, y_ext, -z_ext, z_ext) 

# Initialize tracer and particle trajectories.
# We will trace both electrons and alpha particles (fully ionized helium atoms)
tracer = field.get_tracer([(-B_RADIUS/2, B_RADIUS/2), (-B_RADIUS/2,B_RADIUS/2),  (-B_HEIGHT/2, B_HEIGHT/2)])

beam_radius = IN_RADIUS / 3
r_start = np.linspace(-beam_radius, beam_radius, 8)
x, y = np.meshgrid(r_start, r_start)
x_start, y_start = x[x**2 + y**2 <= beam_radius**2], y[x**2 + y**2 <= beam_radius**2]
z_start = B_HEIGHT / 2

e_trajectories = [] # electron trajectories
a_trajectories = [] # alpha particle trajectories

# We give both beams the same energy
e_start_velocity = T.velocity_vec(1, [0, 0, -1])
a_start_velocity = T.velocity_vec(1, [0,0,-1])


# Trace the particles through the field
for i, (x, y) in enumerate(zip(x_start, y_start)):
    print(f'Tracing particles {i+1}/{len(x_start)}...')
    start_point = np.array([x, y, z_start])

    _, e_trace = tracer(start_point, e_start_velocity) # electrons are default
    _, p_trace = tracer(start_point, a_start_velocity, mass=4*m_p, charge=2*e) # alpha-oarticle = two protons + two neutrons

    e_trajectories.append(e_trace)
    a_trajectories.append(p_trace)

# Plotting
P.plot_mesh(mesh, positive_electrode='red', negative_electrode='blue', boundary='green')
P.plot_equipotential_lines(field, surface=field_xy, color_map='coolwarm',  N_isolines=40, N0=200, N1=200)
P.plot_equipotential_lines(field, surface=field_xz, color_map='coolwarm',  N_isolines=0, N0=200, N1=200)
P.plot_equipotential_lines(field, surface=field_yz, color_map='coolwarm',  N_isolines=0, N0=200, N1=200)
P.plot_trajectories(e_trajectories, line_width=3)
P.plot_trajectories(a_trajectories, line_width=3, color='#FF55FF')
P.show()


# Compute the output beam size 
e_x_end = np.array([t[-1,0] for t in e_trajectories])
e_y_end = np.array([t[-1,1] for t in e_trajectories])

p_x_end = np.array([t[-1,0] for t in a_trajectories])
p_y_end = np.array([t[-1,1] for t in a_trajectories])

# Create a figure of the input and output beam sizes
plt.figure(figsize=(8, 8))  
plt.scatter(x_start, y_start, color='black', label='beams in', marker='o')
plt.scatter(e_x_end, e_y_end, label=' electrons out', marker='x', c='#00AA00')
plt.scatter(p_x_end, p_y_end, label=r'$\alpha$-particles out', marker='x', c='#FF55FF')

plt.title("Particle beam distributions", fontsize=12)
plt.xlabel("x", fontsize=10)
plt.ylabel("y", fontsize=10)
plt.axis('equal')

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.show()




