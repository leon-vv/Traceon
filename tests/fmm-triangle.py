import numpy as np
import matplotlib.pyplot as plt

import traceon.backend as B
import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as S
import traceon.fast_multipole_method as FMM


points = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    
    [0., 0., 1.],
    [1., 0., 1.],
    [0., 1., 1.],

    [.5, .5, .5],
    [1.5, .5, .5],
    [.5, 1.5, .5]
])

elements = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
physical_to_elements = dict(first=np.array([0, 1, 2]))

mesh = G.Mesh(points, elements, physical_to_elements, G.Symmetry.THREE_D)

exc = E.Excitation(mesh)
exc.add_voltage(first=1)

## Result of matrix application by FMM
vertices, names = exc.get_active_elements()
geometry = FMM.get_geometry_in_fortran_layout(vertices)

charges = np.ones(elements.shape[0])

dielectric_indices, dielectric_factors = FMM.get_dielectric_indices_and_factors(names, exc)
result_fmm = FMM.apply_matrix(charges, geometry, 2, dielectric_indices, dielectric_factors)

fmm_matrix_apply = result_fmm

## Result of matrix application by direct method
points = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [.5, 0., 0.],
    [.5, .5, 0],
    [0., .5, 0.],
    
    [0., 0., 1.],
    [1., 0., 1.],
    [0., 1., 1.],
    [.5, 0., 1.],
    [.5, .5, 1],
    [0., .5, 1.],
    
    [0.5, 0.5, 0.5],
    [1.5, 0.5, 0.5],
    [0.5, 1.5, 0.5],
    [1.0, 0.5, 0.5],
    [1.0, 1.0, .5],
    [.5, 1.0, .5]

])

elements = np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]])
physical_to_elements = dict(first=np.array([0, 1, 2]))

mesh = G.Mesh(points, elements, physical_to_elements, G.Symmetry.THREE_D_HIGHER_ORDER)

exc = E.Excitation(mesh)
exc.add_voltage(first=1)

vertices, names = exc.get_active_elements()
matrix, _, _ = S._excitation_to_matrix(exc, vertices, names)

direct_matrix_apply = matrix @ charges

assert np.allclose(fmm_matrix_apply, direct_matrix_apply)

