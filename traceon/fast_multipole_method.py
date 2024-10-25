from math import sqrt, pi
import time
from scipy.sparse.linalg import LinearOperator, gmres
import numpy as np

try:
    import solucia
except ImportError:
    solucia = None

from . import backend
from . import excitation as E


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
    assert solucia is not None, "Solucia should be installed to use fast multipole method"
    return solve_iteratively_solucia(*args, **kwargs)
