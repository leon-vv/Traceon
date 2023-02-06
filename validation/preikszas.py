import matplotlib.pyplot as plt
import numpy as np
from pygmsh import *

import traceon.geometry as G
import traceon.tracing as T
import traceon.aberrations as A
import traceon.excitation as E
import traceon.solver as solver

import util

# Correct values according to
# D. Preikszas and H Ros.e Correction properties of electron mirrors. 1997.
C30 = -0.61629*5 # Radius used is 5
C50 = -169.63*5 
C11 = 0.187461*5
C31 = 8.669*5
C12 = -0.0777*5

correct = [(3, 0, C30), (5, 0, C50), (1, 1, C11), (3, 1, C31), (1, 2, C12)]

def create_geometry(N=100):
    """Generate the mirror studied in the paper:
    D. Preikszas and H. Rose. Correction properties of electron mirrors. 1997.
    
    Args:
        N: number of mesh points. The size of the mesh elements will be chosen such that a line that is
            as long as the geometry is tall will have N points."""
     
    with occ.Geometry() as geom:
        
        points = [
            [0, 0],   #0
            [3, 0],   #1
            [5, -2],  #2 radius is r=5mm
            [5, -3],  #3
            [7, -5],  #4
            [25, -5], #5
            [25, -10],#6
            [7, -10], #7
            [5, -12], #8
            [5, -45], #9
            [0, -45], #10
        ] 

        centers = [[3, -2], [7, -3], [7, -12]]
        
        lcar = 45/N
        points = [geom.add_point(p, lcar) for p in points]
        centers = [geom.add_point(p, lcar) for p in centers]
             
        l1 = geom.add_line(points[0], points[1])
        l2 = geom.add_circle_arc(points[1], centers[0], points[2])
        l3 = geom.add_line(points[2], points[3])
        l4 = geom.add_circle_arc(points[3], centers[1], points[4])
        l5 = geom.add_line(points[4], points[5])
        l6 = geom.add_line(points[5], points[6])
        l7 = geom.add_line(points[6], points[7])
        l8 = geom.add_circle_arc(points[7], centers[2], points[8])
        l9 = geom.add_line(points[8], points[9])
        l10 = geom.add_line(points[9], points[10])
        l11 = geom.add_line(points[10], points[0])
         
        geom.add_physical([l1, l2, l3, l4, l5], 'mirror')
        geom.add_physical([l7, l8, l9, l10], 'corrector')

        cl = geom.add_curve_loop([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11])
        
        return G.Geometry(geom.generate_mesh(dim=1), N)


def compute_error(N):
    excitation = E.Excitation(create_geometry(N))
    excitation.add_voltage(mirror=-250, corrector=1000)
      
    field = solver.solve_bem(excitation)
     
    tracer = T.PlaneTracer(field, -38.688)
     
    C, dE, angles, intersections = A.compute_coefficients(tracer, 1000, dr=0.25)
    
    print('-'*20 + ' Accuracy')
    for i, j, c in correct:
        print(f'C{i}{j} = {C[i,j]:.4e} (correct: {1000*c:.4e}) \t~ {abs(C[i,j])/abs(c*1000) - 1:+.1e}')
    print('-'*20)
    
    return excitation.get_number_of_active_lines(), abs(C[3,0]/(C30*1000) - 1)

util.parser.description = '''
Calculate the aberration coefficients of a diode mirror. The accuracy plotted is determined from the 
spherical aberration coefficient, but the accuracy is printed for all coefficients. See paper:

Correction properties of electron mirrors. D. Preikszas and H. Rose. 1997.
'''
util.parse_validation_args(create_geometry, compute_error, mirror='blue', corrector='green',
    N=[50, 100, 300, 500, 700, 1000])

