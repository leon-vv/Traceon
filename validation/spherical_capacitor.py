from math import cos, sqrt

import numpy as np
from scipy.constants import e, m_e

import traceon as T
from validation import Validation

angle = 0.05
r1 = 7.5
r2 = 12.5

class SphericalCapacitor(Validation):

    def __init__(self):
        super().__init__('''Trace electrons through a spherical capacitor. After the electron traces an arc through the capacitor, its intersection
            with the axis is compared with the exact values given in following paper (first benchmark test):
            
            Comparison of FDM, FEM and BEM for electrostatic charged particle optics. D. Cubric , B. Lencova, F.H. Read, J. Zlamal. 1999.''')
        
        self.plot_colors = dict(inner='blue', outer='darkblue')
    
    def create_mesh(self, MSF, symmetry, higher_order):
        """Create the spherical deflection analyzer from the following paper
        
        D. Cubric, B. Lencova, F.H. Read, J. Zlamal
        Comparison of FDM, FEM and BEM for electrostatic charged particle optics.
        1999.
        """
        def add_shell(radius, name):
            arc = T.Path.arc([0., 0., 0.], [0, 0, -radius], [radius, 0, 0.]).extend_with_arc([0., 0., 0.], [0., 0., radius])
            arc.name = name

            if symmetry.is_3d():
                arc = arc.revolve_z()
                return arc.mesh(mesh_size_factor=MSF)
            else:
                return arc.mesh(mesh_size_factor=MSF, higher_order=higher_order)
         
        return add_shell(r1, 'inner') + add_shell(r2, 'outer')
 
    def get_excitation(self, mesh, symmetry):
        exc = T.Excitation(mesh, symmetry)
        exc.add_voltage(inner=5/3, outer=3/5)
        return exc
     
    def correct_value_of_interest(self):
        correct = -10/(2/cos(angle)**2 - 1)
        assert -12.5 <= correct <= 7.5 # Between spheres
        return correct
    
    def compute_value_of_interest(self, mesh, field):
        position = np.array([0.0, 0.0, 10.0]) 
          
        speed = sqrt(2*e/m_e) # corresponding to 1eV
        vel = speed*np.array([np.cos(angle), 0.0, -np.sin(angle)])
        
        tracer = field.get_tracer( [(-0.1, 12.5), (-0.1, 0.1), (-12.5, 12.5)] )
        print('Starting electron trace...')
        times, pos = tracer(position, vel)
        r_final = T.axis_intersection(pos)
        
        return r_final 

if __name__ == '__main__':
    SphericalCapacitor().run_validation()


