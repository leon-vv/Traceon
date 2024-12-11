import traceon.geometry as G
import traceon.plotting as P
import numpy as np

arc = G.Path.arc_with_angle([0.,0.,0.], 2,angle=np.pi/2)
#line = G.Path.line([0.,0.,0.], [2,2,2])
print(arc.starting_point())

mesh = arc.mesh(mesh_size=0.1)
P.plot_mesh(mesh)
P.show()