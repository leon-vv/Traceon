import traceon.geometry as G
import traceon.plotting as P
import numpy as np

surface = G.Surface.disk_yz(0,0,2)

subregion =surface.get_subregion(0.1, 0.5, 0, surface.path_length2/2).extrude_boundary([1,0,0])

mesh = (surface + subregion).mesh(mesh_size=0.1)

P.plot_mesh(mesh)
P.show()