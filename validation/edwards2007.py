import matplotlib.pyplot as plt
import numpy as np
import gmsh
from pygmsh import *
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
    
    def create_mesh(self, MSF, symmetry):
        with G.Geometry(symmetry) as geom:
            points = [
                [0, 0],
                [0, 5],
                [12, 5],
                [12, 15],
                [0, 15],
                [0, 20],
                [20, 20],
                [20, 0]
            ]

            _3d = symmetry != G.Symmetry.RADIAL
            
            geom.set_mesh_size_factor(MSF)
            
            if _3d:
                points = [geom.add_point([p[0], 0.0, p[1]]) for p in points]
            else:
                points = [geom.add_point(p) for p in points]
            
            l1 = geom.add_line(points[1], points[2])
            l2 = geom.add_line(points[2], points[3])
            l3 = geom.add_line(points[3], points[4])
            
            l4 = geom.add_line(points[0], points[-1])
            l5 = geom.add_line(points[-3], points[-2])
            l6 = geom.add_line(points[-2], points[-1])
            
            if _3d:
                inner = G.revolve_around_optical_axis(geom, [l1, l2, l3])
                boundary = G.revolve_around_optical_axis(geom, [l4, l5, l6])
                
                geom.add_physical(inner, 'inner')
                geom.add_physical(boundary, 'boundary')
                
                return geom.generate_mesh()
            else:
                geom.add_physical([l1, l2, l3], 'inner')
                geom.add_physical([l4, l5, l6], 'boundary')
                
                return geom.generate_mesh()

    def get_excitation(self, geometry):
        excitation = E.Excitation(geometry)
        excitation.add_voltage(boundary=0, inner=10)
        return excitation
    
    def correct_value_of_interest(self):
        # Correct voltage from paper 
        return 6.69099430708
    
    def compute_value_of_interest(self, geometry, field):
        st = time.time()
        
        if geometry.is_3d():
            pot = field.potential_at_point(np.array([12, 0.0, 4]))
        else:
            pot = field.potential_at_point(np.array([12, 4]))
         
        print(f'Time for computing potential {(time.time()-st)*1000:.2f} ms')
        return pot

if __name__ == '__main__':
    Edwards2007().run_validation()




