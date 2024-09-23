from math import sqrt, pi
import time
from scipy.sparse.linalg import LinearOperator, gmres
import numpy as np

try:
    import pyfmmlib
except ImportError:
    pyfmmlib = None

try:
    import solucia
except ImportError:
    solucia = None

from . import backend
from . import excitation as E


def apply_fast_multipole_method(charges, geometry, precision=1, return_field=False):
    # Assert Fortran order
    triangles, centroids, normals = geometry
    
    N = triangles.shape[2]
    assert charges.shape == (N,)
    assert triangles.shape == (3, 3, N)
    assert centroids.shape == (3, N)
    assert normals.shape == (3, N)
    
    result = pyfmmlib.lfmm3dtriatarg(precision,
                            triangles, normals, centroids,
                            1, charges,
                            0, np.empty_like(charges), np.empty_like(normals), # Disable dipoles
                            1, 1 if return_field else 0, # Return potential, and optionally field
                            0, np.zeros( (3, 0) ), 0, np.zeros(1), 0, np.zeros( (3, 1) )) # Disable computation at targets
    
    assert result[0] == 0
     
    if return_field:
        return result[1].real/(4*pi), result[2].real/(4*pi)
    else:
        return result[1].real/(4*pi)

def apply_matrix(charges, geometry, precision, dielectric_indices, dielectric_factors):
    # Compute the result of the matrix, without actually building the matrix
    # using the fast multipole method. Let's call this a virtual matrix application.
    # Care should be taken that the virtual matrix application should return exactly
    # the same values as the matrix application in traceon-backend.c.
    assert dielectric_indices.shape == dielectric_factors.shape
     
    contains_dielectric = len(dielectric_indices) > 0
    
    if not contains_dielectric:
        return apply_fast_multipole_method(charges, geometry, precision, False)
    
    triangles, centroids, normals = geometry
    N = len(triangles)
    
    pot, field = apply_fast_multipole_method(charges, geometry, precision, True) 
      
    # Remember: Fortran order!
    # Overwrite some potential with field dot normal
    field_at_dielectrics = field[:, dielectric_indices]
    normals_at_dielectrics = normals[:, dielectric_indices]
     
    assert field_at_dielectrics.shape == (3, len(dielectric_indices)) \
        and normals_at_dielectrics.shape == (3, len(dielectric_indices))
     
    dotted = np.sum(field_at_dielectrics * normals_at_dielectrics, axis=0)
    assert dotted.shape == dielectric_indices.shape
      
    # Why do we subtract charges[dielectric_indices]?
    # Notice that in fill_self_voltage in traceon-backend.c there is also a subtraction of -1 from the diagonal of the matrix.
    # This follows form the equation needed to solve dielectric. So the subtraction here is equivalent to subtracting 1
    # from the diagonal.
    pot[dielectric_indices] = dielectric_factors * dotted - charges[dielectric_indices]
     
    return pot

def get_geometry_in_fortran_layout(triangles):
    N = len(triangles)
    assert triangles.shape == (N, 3, 3)
    
    triangles = triangles
    centroids = np.mean(triangles, axis=1)
    normals = np.array([backend.normal_3d(1/3, 1/3, t) for t in triangles])
      
    # Convert to Fortran order..
    # This file will work in Fotran order, to be compatible with the FMM library used
    triangles = np.swapaxes(triangles, 0, 2)
    centroids, normals = centroids.T, normals.T
    
    assert triangles.shape == (3, 3, N) and centroids.shape == (3, N) and normals.shape == (3, N)
    
    return (triangles, centroids, normals)

def solve_iteratively_pyfmmlib(triangles, dielectric_indices, dielectric_values, right_hand_side, precision):
    N = len(triangles)
    assert triangles.shape == (N, 3, 3)
    assert right_hand_side.shape == (N,)
    
    geometry = get_geometry_in_fortran_layout(triangles) 
         
    count = 0
    def increase_count(*_):
        nonlocal count
        count += 1
    
    # See the comment about this factor in traceon-backend.c
    K = dielectric_values
    dielectric_factors = np.array([backend.flux_density_to_charge_factor(k) for k in K])
     
    def matvec(charges):
        assert charges.shape == (N,)
        return apply_matrix(charges, geometry, precision, dielectric_indices, dielectric_factors)
     
    # Average accuracy of the computed potential
    accuracy = 5e-8
    # To reach that accuracy we want each element of the residual to be accurate to within
    # 5e-8. The accuracy of norm of the residual is then sqrt(N) * 5e-8
    tol = accuracy * sqrt(N)
     
    charges, _ = gmres(LinearOperator(matvec=matvec, shape=(N, N)),
        right_hand_side,
        x0 = np.ones(len(triangles)),
        callback=increase_count,
        callback_type='pr_norm',
        restart=750,
        atol=0., rtol=tol)
     
    assert np.all(np.isfinite(charges))
     
    return charges, count

def solve_iteratively_solucia(triangles, dielectric_indices, dielectric_values, right_hand_side, precision):
    
    count = 0
    def increase_count(*args):
        nonlocal count, tol
        count += 1
    
    assert len(dielectric_indices) == 0, "Dielectrics (or boundary) not yet supported in Solucia"
    
    if precision <= 0:
        l_max = 4
    elif precision == 1:
        l_max = 8
    elif precision == 2:
        l_max = 16
    elif precision == 3:
        l_max = 24
    elif precision > 4:
        l_max = 32
    
    N_max = 475
      
    st = time.time()
    fmm = solucia.FastMultipoleMethodTriangles(triangles, N_max, l_max)
    print(f'Solucia preparation took: {time.time()-st:.2f} s')
     
    def matvec(charges):
        return fmm.potentials(charges) / (4*pi)
     
    # Average accuracy of the computed potential
    accuracy = 5e-8
    # To reach that accuracy we want each element of the residual to be accurate to within
    # 5e-8. The accuracy of norm of the residual is then sqrt(N) * 5e-8
    N = len(triangles)
    tol = accuracy * sqrt(N)
     
    charges, _ = gmres(LinearOperator(matvec=matvec, shape=(N, N)),
        right_hand_side,
        x0 = np.ones(len(triangles)),
        callback=increase_count,
        callback_type='pr_norm',
        restart=750,
        atol=0., rtol=tol)
    
    assert np.all(np.isfinite(charges))
    
    return charges, count

def solve_iteratively(*args, **kwargs):
    assert pyfmmlib is not None or solucia is not None, "Either pyfmmlib or solucia should be installed to use fast multipole method"

    if solucia is not None:
        return solve_iteratively_solucia(*args, **kwargs)
    else:
        return solve_iteratively_pyfmmlib(*args, **kwargs)
