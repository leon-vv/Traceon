import traceon.geometry as G
import traceon.plotting as P

path = G.Path.line([0,0,0], [1,0,0]).line_to([0.5,1,0]).line_to([0.5,2,0]).line_to([-1,0,0]).close()

surface = path.extrude_to_centroid()

mesh = surface.mesh(mesh_size=0.1)

P.plot_mesh(mesh)
P.show()