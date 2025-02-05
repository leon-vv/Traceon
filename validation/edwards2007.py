import numpy as np
import time

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S

from validation import Validation


class Edwards2007(Validation):
    """Create the geometry g5 (figure 2) from the following paper:
    D. Edwards. High precision electrostatic potential calculations for cylindrically
    symmetric lenses. 2007.
    """
     
    def __init__(self):
        super().__init__('Compute the potential at point (12, 4) inside two coaxial cylinders. See paper: \
                            High precision electrostatic potential calculations for cylindrically symmetric lenses. David Edwards. 2007.')
         
        self.plot_colors = dict(boundary='blue', inner='orange')
    
    def create_mesh(self, MSF, symmetry, higher_order):
        points = [
            [0, 0., 5],
            [12, 0., 5],
            [12, 0., 15],
            [0, 0., 15],
            [0, 0., 0],
            [20, 0., 0],
            [20, 0., 20],
            [0, 0., 20]
        ]

        inner = G.Path.line(points[0], points[1])\
            .extend_with_line(points[2]).extend_with_line(points[3])
         
        boundary = G.Path.line(points[4], points[5])\
            .extend_with_line(points[6]).extend_with_line(points[7])
        
        if symmetry.is_3d():
            inner = inner.revolve_z()
            boundary = boundary.revolve_z()
        
        inner.name = 'inner'
        boundary.name = 'boundary'
        
        return (inner+boundary).mesh(mesh_size_factor=MSF)
    
    def get_excitation(self, geometry, symmetry):
        excitation = E.Excitation(geometry, symmetry)
        excitation.add_voltage(boundary=0, inner=10)
        return excitation
    
    def correct_value_of_interest(self):
        # Correct voltage from paper 
        return 6.69099430708
    
    def compute_value_of_interest(self, geometry, field):
        return field.potential_at_point([12, 0.0, 4])

if __name__ == '__main__':
    Edwards2007().run_validation()




