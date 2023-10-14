import time, math

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pygmsh import *

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S

from validation import Validation


class SimpleMirror(Validation):

    def __init__(self):
        super().__init__('Calculated a trajectory through a simple flat mirror.')
        self.plot_colors = dict(mirror='brown', lens='blue', ground='green')

    def default_MSF(self, symmetry):
        if symmetry.is_3d():
            return [200, 400, 600, 1000, 1500]
        else:
            return [10, 50, 150, 250, 600]

    def create_mesh(self, MSF, symmetry):
        _3d = symmetry.is_3d()
        
        with G.Geometry(symmetry, size_from_distance=True, zmin=0.1, zmax=2) as geom:
            
            points = [ [0, -1], [2, -1], [2, 1], [0.3, 1] ]
            if _3d: 
                points = [ [p[0], 0.0, p[1]] for p in points ]
            
            points = [geom.add_point(p) for p in points]
            
            ground_lines = [geom.add_line(p1, p2) for p1, p2 in zip(points, points[1:])]
                
            if _3d:
                revolved = G.revolve_around_optical_axis(geom, ground_lines)
                geom.add_physical(revolved, 'ground')
            else:
                geom.add_physical(ground_lines, 'ground')
            
            points = [ [0.0, 0.0], [1, 0] ]
            if _3d:
                points = [ [p[0], 0.0, p[1]] for p in points ]
            
            points = [geom.add_point(p) for p in points]
            
            mirror_line = geom.add_line(points[0], points[1])

            if _3d:
                revolved = G.revolve_around_optical_axis(geom, mirror_line)
                geom.add_physical(revolved, 'mirror')
            else:
                geom.add_physical(mirror_line, 'mirror')
            
            geom.set_mesh_size_factor(MSF)
            return geom.generate_mesh()
     
    def get_excitation(self, mesh):
        excitation = E.Excitation(mesh)
        excitation.add_voltage(mirror=-110, ground=0.0)
        return excitation
    
    def correct_value_of_interest(self): 
        return 1.6327355811e-01

    def compute_value_of_interest(self, geometry, field):
        _3d = geometry.is_3d()
        bounds = ((-0.22, 0.22), (0.02, 11))
        
        if _3d:
            bounds = ((-0.22, 0.22), (-0.22, 0.22), (0.02, 11))
         
        axial_field = field.axial_derivative_interpolation(0.02, 4)
        tracer = T.Tracer(axial_field, bounds)
         
        pos = np.array([0.0, 10.0]) if not _3d else np.array([0.0, 0.0, 10.0])
        vel = T.velocity_vec_xz_plane(100, 1e-3, three_dimensional=_3d)
        
        st = time.time()
        _, pos = tracer(pos, vel)
        print(f'Trace took {(time.time()-st)*1000:.0f} ms')
        
        p = T.xy_plane_intersection(pos, 10)
        
        calculated = np.linalg.norm(p[:2] if _3d else p[:1])
        return calculated

if __name__ == '__main__':
    SimpleMirror().run_validation()


