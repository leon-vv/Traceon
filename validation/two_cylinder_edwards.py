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

gap_size = 0.2
R = 1.0
wall_thickness = 1
boundary_length = 20

def gap_voltage(x, y, z):
    return (z-9.9)/0.2 * 10

class TwoCylinderEdwards(Validation):
    
    def __init__(self):
        super().__init__('''Compute the potential inside a two cylinder lens.  The correct values for the potential are taken from the paper:\n
            Accurate Potential Calculations For The Two Tube Electrostatic Lens Using A Multiregion FDM Method.  David Edwards, Jr. 2007.
            ''')
        self.plot_colors = dict(v1='blue', v2='green', gap='orange')
    
    def create_mesh(self, MSF, symmetry, higher_order):
        cylinder_length = (boundary_length - gap_size)/2
        
        bottom  = G.Path.line([0., 0., 0.], [R, 0., 0.]).line_to([R, 0., cylinder_length])
        gap = G.Path.line([R, 0., cylinder_length], [R, 0., cylinder_length+gap_size])
        top = G.Path.line([R, 0., cylinder_length+gap_size], [R, 0., boundary_length]).line_to([0., 0., boundary_length])

        bottom.name = 'v1'
        gap.name = 'gap'
        top.name = 'v2'
         
        if symmetry.is_3d():
            bottom = bottom.revolve_z()
            gap = gap.revolve_z()
            top = top.revolve_z()
            return gap.mesh(mesh_size_factor=2) + (bottom + top).mesh(mesh_size_factor=MSF)
        else:
            return (bottom + gap + top).mesh(mesh_size_factor=MSF, higher_order=higher_order)
    
    def get_excitation(self, geom, symmetry):
        exc = E.Excitation(geom, symmetry)
        exc.add_voltage(v1=0, v2=10, gap=gap_voltage)
        return exc

    def correct_value_of_interest(self):
        #edwards = np.array([5.0, 2.5966375108359858, 1.1195606398479115, .4448739946832647, .1720028130382, .065954697686])
        #z = 2*np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        return 2.5966375108359858
     
    def compute_value_of_interest(self, mesh, field):
        return field.potential_at_point([0., 0., 10 - 0.4])

if __name__ == '__main__':
    TwoCylinderEdwards().run_validation()
