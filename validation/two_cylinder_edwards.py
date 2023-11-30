import time
import sys

from scipy.interpolate import *
import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as solver
import traceon.plotting as P

from validation import Validation

def gap_voltage(x, y):
    return (y-9.9)/0.2 * 10

class TwoCylinderEdwards(Validation):
    
    def __init__(self):
        super().__init__('''Compute the potential inside a two cylinder lens.  The correct values for the potential are taken from the paper:\n
            Accurate Potential Calculations For The Two Tube Electrostatic Lens Using A Multiregion FDM Method.  David Edwards, Jr. 2007.
            ''')
        self.plot_colors = dict(v1='blue', v2='green', gap='orange')
    
    def default_MSF(self, symmetry):
        if symmetry.is_3d():
            return [400, 600, 1000, 1500]
        else:
            return [25, 100, 200, 400]
    
    def create_mesh(self, MSF, symmetry, higher_order):
        S = 0.2
        R = 1.0
        wall_thickness = 1
        boundary_length = 20
        
        with G.Geometry(symmetry) as geom:
            cylinder_length = (boundary_length - S)/2
            assert boundary_length == 2*cylinder_length + S
            assert wall_thickness == 0.0 or wall_thickness > 0.1
            
            points = [
                [0, 0],
                [R, 0],
                [R, cylinder_length],
                [R + wall_thickness, cylinder_length],
                [R + wall_thickness, cylinder_length + S],
                [R, cylinder_length + S],
                [R, boundary_length],
                [0, boundary_length]
            ]
             
            physicals = [('v1', [0, 1, 2, 3]), ('v2', [4, 5, 6, 7]), ('gap', [3, 4])]
             
            if symmetry.is_3d():
                points = [geom.add_point([p[0], 0., p[1]]) for p in points]
                
                for key, indices in physicals:
                    p = [points[i] for i in indices]
                    lines = [geom.add_line(p1, p2) for p1, p2 in zip(p, p[1:])]
                    revolved = G.revolve_around_optical_axis(geom, lines, 1.0)
                    geom.add_physical(revolved, key)
            else:
                points = [geom.add_point(p) for p in points]
                
                for key, indices in physicals:
                    p = [points[i] for i in indices]
                    lines = [geom.add_line(p1, p2) for p1, p2 in zip(p, p[1:])]
                    geom.add_physical(lines, key)
            
            geom.set_mesh_size_factor(MSF)
            return geom.generate_line_mesh(higher_order) if geom.is_2d() else geom.generate_triangle_mesh(higher_order)

    def get_excitation(self, geom):
        exc = E.Excitation(geom)
        exc.add_voltage(v1=0, v2=10)#, gap=gap_voltage)
        return exc

    def correct_value_of_interest(self):
        #edwards = np.array([5.0, 2.5966375108359858, 1.1195606398479115, .4448739946832647, .1720028130382, .065954697686])
        #z = 2*np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        return 2.5966375108359858
    
    def compute_value_of_interest(self, mesh, field):
        if mesh.is_3d():
            point = np.array([0.0, 0.0, 10.0 - 0.4])
        else:
            point = np.array([0.0, 10.0 - 0.4])

        return field.potential_at_point(point)

if __name__ == '__main__':
    TwoCylinderEdwards().run_validation()
