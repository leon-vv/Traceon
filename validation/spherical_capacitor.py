from math import cos

import matplotlib.pyplot as plt
import numpy as np
from pygmsh import *

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S
import traceon.tracing as T

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
    
    def default_MSF(self, symmetry):
        if symmetry.is_3d():
            return [20, 100, 300, 500]
        else:
            return [10, 100, 200]
    
    def create_mesh(self, MSF, symmetry):
        """Create the spherical deflection analyzer from the following paper
        
        D. Cubric, B. Lencova, F.H. Read, J. Zlamal
        Comparison of FDM, FEM and BEM for electrostatic charged particle optics.
        1999.
        """
        with G.Geometry(symmetry) as geom:
            
            points = [
                [0, -r2],
                [0, -r1],
                [r1, 0],
                [0, r1],
                [0, r2],
                [r2, 0]
            ]
            
            points = [geom.add_point([p[0], 0, p[1]] if geom.is_3d() else p) for p in points]
            center = geom.add_point([0, 0, 0])
            
            l2 = geom.add_circle_arc(points[1], center, points[2])
            l3 = geom.add_circle_arc(points[2], center, points[3])
            
            l5 = geom.add_circle_arc(points[4], center, points[5])
            l6 = geom.add_circle_arc(points[5], center, points[0])
            
            if not geom.is_3d():
                geom.add_physical([l2, l3], 'inner')
                geom.add_physical([l5, l6], 'outer')
            else:
                s1 = G.revolve_around_optical_axis(geom, [l2, l3])
                s2 = G.revolve_around_optical_axis(geom, [l5, l6])
                geom.add_physical(s1, 'inner')
                geom.add_physical(s2, 'outer')

            geom.set_mesh_size_factor(MSF)
            return geom.generate_mesh()

    def get_excitation(self, mesh):
        exc = E.Excitation(mesh)
        exc.add_voltage(inner=5/3, outer=3/5)
        return exc
     
    def correct_value_of_interest(self):
        correct = -10/(2/cos(angle)**2 - 1)
        assert -12.5 <= correct <= 7.5 # Between spheres
        return correct
    
    def compute_value_of_interest(self, mesh, field):
        if not mesh.is_3d():
            bounds = ((-0.1, 12.5), (-12.5, 12.5))
            position = np.array([0.0, 10.0]) 
            vel = np.array([np.cos(angle), -np.sin(angle)])*0.5930969604919433
        else:
            bounds = ((-0.1, 12.5), (-0.1, 0.1), (-12.5, 12.5))
            position = np.array([0.0, 0.0, 10.0]) 
            vel = np.array([np.cos(angle), 0.0, -np.sin(angle)])*0.5930969604919433
        
        tracer = T.Tracer(field, bounds)
        print('Starting electron trace...')
        times, pos = tracer(position, vel)
        
        r_final = T.axis_intersection(pos)
        
        return r_final 

if __name__ == '__main__':
    SphericalCapacitor().run_validation()


