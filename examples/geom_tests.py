import traceon.geometry as G
import traceon.plotting as P
import numpy as np

arc = G.Path.arc_with_angle([0.,0.,0.], 1, arc_angle=3*np.pi/2, start_angle=np.pi/2)

line = G.Path.line([0,0,0], [0,0,1])

line_arc = line.arc_to_with_angle(1, np.pi, [0,-1.,0])

mesh = line_arc.mesh(mesh_size=0.1)
P.plot_mesh(mesh)
P.show()

