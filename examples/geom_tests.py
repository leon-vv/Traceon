import traceon.geometry as G
import traceon.plotting as P
import numpy as np

surface = G.Surface.sphere(3)

region = surface.get_region(0, surface.path_length1/2, 0, surface.path_length2/2)

mesh = region.mesh(mesh_size=0.5)
P.plot_mesh(mesh)
P.show()