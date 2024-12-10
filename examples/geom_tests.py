import traceon.geometry as G
import traceon.plotting as P
import numpy as np

surface = G.Surface.sphere(1)

patch =surface.get_patch(surface.path_length1/4, surface.path_length1/2, 0, surface.path_length2/2)



P.plot_mesh(mesh)
P.show()