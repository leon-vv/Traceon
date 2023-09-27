from scipy.sparse.linalg import LinearOperator, gmres
import numpy as np

try:
    import pyfmmlib
except ImportError:
    print('WARNING: pyfmmlib not found, fast multipole method not supported')

from . import backend
from . import excitation as E


def apply_matrix(charges, mesh, dielectric_indices, dielectric_factors):
    # Compute the result of the matrix, without actually building the matrix
    # using the fast multipole method. Let's call this a virtual matrix application.
    # Care should be taken that the virtual matrix application should return exactly
    # the same values as the matrix application in traceon-backend.c.
    assert dielectric_indices.shape == dielectric_factors.shape

    charges = np.copy(charges)*np.pi
    
    contains_dielectric = len(dielectric_indices) > 0
          
    if not contains_dielectric:
        return pyfmmlib.fmm_tria("p", -1, pyfmmlib.LaplaceKernel(), mesh, slp_density=charges).real
     
    pot, field = pyfmmlib.fmm_tria("pg", -1, pyfmmlib.LaplaceKernel(), mesh, slp_density=charges)
    pot, field = pot.real, field.real

    # The values returned by pyfmmlib are actually the gradients.
    # So we have to multiply by -1 to get the real field values
    field *= -1
     
    # Overwrite some potential with field dot normal
    field_at_dielectrics = field[dielectric_indices]
    normals_at_dielectrics = mesh.normals[dielectric_indices]
    
    assert field_at_dielectrics.shape == (len(dielectric_factors), 3) and \
        field_at_dielectrics.shape == normals_at_dielectrics.shape
    
    dotted = np.sum(field_at_dielectrics * normals_at_dielectrics, axis=1)
     
    # Why do we subtract charges[dielectric_indices]?
    # Notice that in fill_self_voltage in traceon-backend.c there is also a subtraction of -1 from the diagonal of the matrix.
    # This follows form the equation needed to solve dielectric. So the subtraction here is equivalent to subtracting 1
    # from the diagonal.
    pot[dielectric_indices] = dielectric_factors * dotted - charges[dielectric_indices]/np.pi
     
    return pot


class MeshWrapperPyFMMLib:
    def __init__(self, triangles):
        self.triangles = triangles
        self.normals = np.array([backend.normal_3d(*t) for t in triangles])
        self.centroids = np.mean(triangles, axis=1)
     
    def __len__(self):
        return len(self.triangles)

def get_dielectric_indices_and_factors(names, excitation):
    dielectric_names = [n for n, _ in names.items() if excitation.excitation_types[n][0] == E.ExcitationType.DIELECTRIC]
    
    if len(dielectric_names):
        dielectric_indices = np.concatenate([names[n] for n in dielectric_names])
        K = np.concatenate([np.full(len(names[n]), excitation.excitation_types[n][1]) for n in dielectric_names])
        # See the comment about this factor in traceon-backend.c
        dielectric_factors = (2*K - 2) / (np.pi*(1+K))

        return dielectric_indices, dielectric_factors
    else:
        return np.array([]), np.array([])

def solve_iteratively(names, excitation, triangles, right_hand_side):
    N = len(triangles)
    assert triangles.shape == (N, 3, 3)
    assert right_hand_side.shape == (N,)
    assert isinstance(names, dict)
     
    mesh = MeshWrapperPyFMMLib(triangles)
            
    count = 0
    def increase_count(*_):
        nonlocal count
        count += 1

    dielectric_indices, dielectric_factors = get_dielectric_indices_and_factors(names, excitation)
     
    operator = LinearOperator(matvec=lambda c: apply_matrix(c, mesh, dielectric_indices, dielectric_factors), shape=(N, N))
    charges, _ = gmres(operator, right_hand_side, callback=increase_count)
    assert np.all(np.isfinite(charges))
    
    return charges, count


