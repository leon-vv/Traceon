import traceon.geometry as G
import traceon.plotting as P
import numpy as np

path = G.Path.line([1,0,0], [1,0,1])
path_with_arc = path.arc_to2(2., np.pi/2, np.array([0.,2.,0]))

arc2 = G.Path.arc2(1, np.pi, plane_normal=[1,1,1], direction= [0.33333333,  0.33333333, -0.66666667])

mesh = path_with_arc.mesh(mesh_size=0.1)
P.plot_mesh(mesh)
P.show()

