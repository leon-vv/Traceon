
import numpy as np
from pygmsh import *

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as S
import traceon.plotting as P

N = 100

with occ.Geometry() as geom:
    r1 = 0.5
    r2 = 1.0

    points = [ [0.0, 0.0], [0, -r1], [0, r1], [0, -r2], [0, r2] ]
    p = [geom.add_point(p, 1/N) for p in points]
     
    l1 = geom.add_circle_arc( p[2], p[0], p[1] )
    l2 = geom.add_circle_arc( p[4], p[0], p[3] )
    
    geom.add_physical([l1], 'inner')
    geom.add_physical([l2], 'outer')
    
    mesh = geom.generate_mesh(dim=1)
    geom = G.Geometry(mesh, N)

exc = E.Excitation(geom)
exc.add_voltage(inner=1)
exc.add_voltage(outer=0)

_, line_points, charges, _ = S.solve_bem(exc)

Q = []

for line, charge in zip(line_points, charges):
    length = np.linalg.norm(line[1] - line[0])
    middle = (line[1] + line[0])/2
    
    Q.append(charge * length * 2*np.pi*middle[0])
    
Q = np.array(Q)
print(np.sum(Q))
print(np.sum(Q[Q>0]))
print(np.sum(Q[Q>0])/4 - 1)

P.show_line_mesh(geom.mesh, inner='blue', outer='green')
P.show()
    




