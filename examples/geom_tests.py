import traceon.geometry as G
import traceon.plotting as P
import numpy as np
path = G.Path.line([0,0,0], [1,1,1])
arc = path.arc([1.,0,0], [0.,0,0], [-1.,0,0])
mesh = (arc).mesh(mesh_size=0.1)
P.plot_mesh(mesh)



