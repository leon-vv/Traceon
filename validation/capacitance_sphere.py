import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as S
import traceon.plotting as P

from validation import Validation

r1 = 0.5
r2 = 1.0

r3 = 0.6
r4 = 0.9

K=2

class CapacitanceSphere(Validation):
    
    def __init__(self):
        super().__init__('''Compute the capacitance of two concentric spheres with a layer of dielectric material in between.''')
        self.plot_colors = dict(inner='blue', outer='darkblue', dielectric='green')
    
    def create_mesh(self, MSF, symmetry, higher_order):
         
        def add_shell(radius, name, reverse=False):
            arc = G.Path.arc([0., 0., 0.], [0, 0, -radius], [radius, 0, 0.]).arc_to([0., 0., 0.], [0., 0., radius])
            arc.name = name

            if symmetry.is_3d():
                arc = arc.revolve_z()
                mesh = arc.mesh(mesh_size_factor=MSF)
            else:
                mesh = arc.mesh(mesh_size_factor=MSF, higher_order=higher_order)
            
            return mesh.flip_normals() if reverse else mesh
        
        return add_shell(r1, 'inner') + add_shell(r2, 'outer') + add_shell(r3, 'dielectric', True) + add_shell(r4, 'dielectric')
    
    def get_excitation(self, geom, symmetry):
        exc = E.Excitation(geom, symmetry)
        exc.add_voltage(inner=1)
        exc.add_voltage(outer=0)
        exc.add_dielectric(dielectric=K)
        return exc

    def correct_value_of_interest(self):
        return 4*np.pi/( (1/r1 - 1/r3) + (1/r3 - 1/r4)/K + (1/r4 - 1/r2))
    
    def compute_value_of_interest(self, geometry, field):
        exc = self.get_excitation(geometry, field.symmetry)
         
        x = np.linspace(0.55, 0.95)
        f = [field.field_at_point(np.array([x_, 0.0, 0.0]))[0] for x_ in x]
        
        # Find the charges
        _, names = exc.get_electrostatic_active_elements()
        Q = {n:field.charge_on_elements(i) for n, i in names.items()}
          
        capacitance = (abs(Q['outer']) + abs(Q['inner']))/2
        return capacitance

if __name__ == '__main__':
    CapacitanceSphere().run_validation()

