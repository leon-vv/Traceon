
import numpy as np
from pygmsh import *
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as S
import traceon.plotting as P
import traceon.util as U

N = 100

with occ.Geometry() as geom:
    center = geom.add_point([0.0, 0.0], 1/N)
    
    def add_shell(r, reorient=False):
        points = [[0,-r], [r, 0], [0,r]]
        p = [geom.add_point(p, 1/N) for p in points]

        if not reorient:
            return [geom.add_circle_arc(p[0], center, p[1]), geom.add_circle_arc(p[1], center, p[2])]
        else:
            return [geom.add_circle_arc(p[2], center, p[1]), geom.add_circle_arc(p[1], center, p[0])]
     
    r1 = 0.5
    r2 = 1.0

    r3 = 0.6
    r4 = 0.9
      
    l1 = add_shell(r1)
    d1 = add_shell(r3, reorient=True)
    d2 = add_shell(r4)
    l2 = add_shell(r2)
     
    geom.add_physical(l1, 'inner')
    geom.add_physical(l2, 'outer')
    geom.add_physical([*d1, *d2], 'floating')
     
    mesh = geom.generate_mesh(dim=1)
    geom = G.Geometry(mesh, N)

exc = E.Excitation(geom)
exc.add_voltage(inner=1)
exc.add_voltage(outer=0)
charge_floating = 3.0
exc.add_floating_conductor(floating=charge_floating)

solution = S.solve_bem(exc)
_, line_points, charges, _ = solution
# Bit of hack, figure out a cleaner way to pass this around
_, names = exc.get_active_lines()

Q = {}

for n, v in names.items():
    Q[n] = 0
    
    for line, charge in zip(line_points[v], charges[v]):
        
        length = np.linalg.norm(line[1] - line[0])
        middle = (line[1] + line[0])/2
        
        # Take into account surface area of entire ring
        Q[n] += charge * length*2*np.pi*middle[0]
    

print('-'*25)
print('Total charge on electrodes: ')
for n, c in Q.items():
    print('%s: \t%.3f' % (n, c))

print('Charge requested on floating conductor: %.3f' % charge_floating)

plt.figure()
plt.title('Potential')
r = np.linspace(0.35, 1.15)
pot = [S.potential_at_point(np.array([r_, 0.0]), solution) for r_ in r]
plt.plot(r, pot)

plt.figure()
r = np.linspace(0.35, 1.15)
field = [S.field_at_point(np.array([r_, 0.0]), solution)[0] for r_ in r]
plt.plot(r, field)

P.show_line_mesh(geom.mesh, inner='blue', outer='green', dielectric='purple')
P.show()
    




