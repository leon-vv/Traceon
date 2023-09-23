from scipy.sparse.linalg import LinearOperator, gmres
import numpy as np

try:
    import pyfmmlib
except ImportError:
    print('WARNING: pyfmmlib not found, fast multipole method not supported')

from . import backend

def calculate_potentials(charges, mesh):
    return pyfmmlib.fmm_tria("p", 1, pyfmmlib.LaplaceKernel(), mesh, slp_density=charges).real

class MeshWrapperPyFMMLib:
    def __init__(self, triangles):
        self.triangles = triangles
        self.normals = np.array([backend.normal_3d(*t) for t in triangles])
        self.centroids = np.mean(triangles, axis=1)
     
    def __len__(self):
        return len(self.triangles)

def solve_iteratively(triangles, right_hand_side):
    N = len(triangles)
    assert triangles.shape == (N, 3, 3)
    assert right_hand_side.shape == (N,)
     
    mesh = MeshWrapperPyFMMLib(triangles)
     
    count = 0
    def increase_count(*_):
        nonlocal count
        count += 1
    
    operator = LinearOperator(matvec=lambda c: calculate_potentials(c, mesh), shape=(N, N))
    charges, _ = gmres(operator, right_hand_side, callback=increase_count)
    assert np.all(np.isfinite(charges))
    
    return charges/np.pi, count


