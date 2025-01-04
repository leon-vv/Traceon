"""The backend module is a small wrapper around code implemented in C. All performance critical code is 
implemented in the C backend. As a user you should not use these functions directly yourself."""

import importlib
import ctypes as C
import os.path as path
import platform
from typing import Callable, Optional, Tuple, List

from numpy.ctypeslib import ndpointer
from numpy.typing import DTypeLike
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad

from .. import logging

DEBUG = False

if DEBUG:
    logging.set_log_level(logging.LogLevel.DEBUG)

## Attempt 1: load local
if platform.system() in ['Linux', 'Darwin']:
    local_path = path.join(path.dirname(__file__), 'traceon_backend.so')
    global_file = 'traceon/backend/traceon_backend.abi3.so'
else:
    local_path = path.join(path.dirname(__file__), 'traceon_backend.pyd')
    global_file = 'traceon/backend/traceon_backend.pyd'

if path.isfile(local_path):
    backend_lib = C.CDLL(local_path)
else:
    ## Attempt 2: use getsitepackages
    import site
    paths = site.getsitepackages() + [site.getusersitepackages()]
    
    global_path = None

    for p in paths:
        if path.isfile(path.join(p, global_file)):
            global_path = path.join(p, global_file)
            break
     
    if global_path is None:
        help_txt = '''
        Cannot find Traceon backend (C compiled dynamic library).
        It should have been compiled automatically when installing this package using 'pip3 install traceon'.
        If you're running this package locally (i.e. git clone) you have to build this dynamic library yourself.
        On Linux you can use:
            gcc ./traceon/backend/traceon-backend.c -o ./traceon/backend/traceon-backend.so -lm -shared -fPIC -O3 -ffast-math -std=c99 -march=native'''
        
        raise RuntimeError(help_txt)
    
    backend_lib = C.CDLL(global_path)


TRACING_BLOCK_SIZE = C.c_size_t.in_dll(backend_lib, 'TRACING_BLOCK_SIZE').value

DERIV_2D_MAX = C.c_int.in_dll(backend_lib, 'DERIV_2D_MAX_SYM').value

N_QUAD_2D = C.c_int.in_dll(backend_lib, 'N_QUAD_2D_SYM').value
N_TRIANGLE_QUAD = C.c_int.in_dll(backend_lib, 'N_TRIANGLE_QUAD_SYM').value

NU_MAX = C.c_int.in_dll(backend_lib, 'NU_MAX_SYM').value
M_MAX = C.c_int.in_dll(backend_lib, 'M_MAX_SYM').value

# Pass numpy array to C
def arr(ndim=None, shape=None, dtype:DTypeLike=np.float64):
    return ndpointer(ndim=ndim, shape=shape, dtype=dtype, flags=('C_CONTIGUOUS', 'ALIGNED')) # type: ignore

# Pass one dimensional Numpy array to C
v2 = arr(shape=(2,))
v3 = arr(shape=(3,))

dbl = C.c_double
dbl_p = C.POINTER(dbl)
vp = C.c_void_p
sz = C.c_size_t

integration_cb_1d = C.CFUNCTYPE(dbl, dbl, vp)

# The low level field function has the following arguments:
# - A pointer to position (three doubles)
# - A pointer to velocity (three doubles)
# - A pointer to auxillary data the field function needs
# - A pointer to write the computed electric field (three doubles)
# - A pointer to write the computed magnetic field (three doubles)
field_fun = C.CFUNCTYPE(None, C.POINTER(dbl), C.POINTER(dbl), vp, C.POINTER(dbl), C.POINTER(dbl));

vertices = arr(ndim=3)
lines = arr(ndim=3)
charges_3d = arr(ndim=1)
charges_2d = arr(ndim=1)
currents_2d = arr(ndim=1)
z_values = arr(ndim=1)

jac_buffer_3d = arr(ndim=2)
pos_buffer_3d = arr(ndim=3)

jac_buffer_2d = arr(ndim=2)
pos_buffer_2d = arr(ndim=3)

radial_coeffs = arr(ndim=3)


class EffectivePointCharges2D(C.Structure):
    _fields_ = [
        ("charges", dbl_p),
        ("jacobians", dbl_p),
        ("positions", dbl_p),
        ("N", C.c_size_t)
    ]

    def __init__(self, eff, *args, **kwargs):
        assert eff.is_2d()
        super(EffectivePointCharges2D, self).__init__(*args, **kwargs)

        # Beware, we need to keep references to the arrays pointed to by the C.Structure
        # otherwise, they are garbage collected and bad things happen
        self.charges_arr = ensure_contiguous_aligned(eff.charges)
        self.jacobians_arr = ensure_contiguous_aligned(eff.jacobians)
        self.positions_arr = ensure_contiguous_aligned(eff.positions)
         
        self.charges = self.charges_arr.ctypes.data_as(dbl_p)
        self.jacobians = self.jacobians_arr.ctypes.data_as(dbl_p)
        self.positions = self.positions_arr.ctypes.data_as(dbl_p)
        self.N = len(eff)
        
class EffectivePointCharges3D(C.Structure):
    _fields_ = [
        ("charges", dbl_p),
        ("jacobians", dbl_p),
        ("positions", dbl_p),
        ("N", C.c_size_t)
    ]
    
    def __init__(self, eff, *args, **kwargs):
        assert eff.is_3d()
        super().__init__(*args, **kwargs)
        
        # Beware, we need to keep references to the arrays pointed to by the C.Structure
        # otherwise, they are garbage collected and bad things happen
        self.charges_arr = ensure_contiguous_aligned(eff.charges)
        self.jacobians_arr = ensure_contiguous_aligned(eff.jacobians)
        self.positions_arr = ensure_contiguous_aligned(eff.positions)
             
        self.charges = self.charges_arr.ctypes.data_as(dbl_p)
        self.jacobians = self.jacobians_arr.ctypes.data_as(dbl_p)
        self.positions = self.positions_arr.ctypes.data_as(dbl_p)
        self.N = len(eff)

class EffectivePointCurrents3D(C.Structure):
    _fields_ = [
        ("currents", dbl_p),
        ("jacobians", dbl_p),
        ("positions", dbl_p),
        ("directions", dbl_p),
        ("N", C.c_size_t)
    ]
    
    def __init__(self, eff, *args, **kwargs):
        super(EffectivePointCurrents3D, self).__init__(*args, **kwargs)

        # In solver.py we use consistently the EffectivePointCharges class
        # so when storing effective point currents, the charges are actually currents
        currents = eff.charges
        
        N = len(currents)
        assert currents.shape == (N,) and currents.dtype == np.double
        assert eff.jacobians.shape == (N, N_QUAD_2D) and eff.jacobians.dtype == np.double
        assert eff.positions.shape == (N, N_QUAD_2D, 3) and eff.positions.dtype == np.double
        assert eff.directions.shape == (N, N_QUAD_2D, 3) and eff.directions.dtype == np.double

        # Beware, we need to keep references to the arrays pointed to by the C.Structure
        # otherwise, they are garbage collected and bad things happen
        self.currents_arr = ensure_contiguous_aligned(currents)
        self.jacobians_arr = ensure_contiguous_aligned(eff.jacobians)
        self.positions_arr = ensure_contiguous_aligned(eff.positions)
        self.directions_arr = ensure_contiguous_aligned(eff.directions)

        self.currents = self.currents_arr.ctypes.data_as(dbl_p)
        self.jacobians = self.jacobians_arr.ctypes.data_as(dbl_p)
        self.positions = self.positions_arr.ctypes.data_as(dbl_p)
        self.directions = self.directions_arr.ctypes.data_as(dbl_p)
        
        self.N = N


class FieldEvaluationArgsRadial(C.Structure):
    _fields_ = [
        ("elec_charges", C.c_void_p),
        ("mag_charges", C.c_void_p),
        ("current_charges", C.c_void_p),
        ("bounds", C.POINTER(C.c_double))
    ]

    def __init__(self, elec, mag, current, bounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert bounds is None or bounds.shape == (3, 2)
        
        # Beware, we need to keep references to the arrays pointed to by the C.Structure
        # otherwise, they are garbage collected and bad things happen
        self.eff_elec = EffectivePointCharges2D(elec)
        self.eff_mag = EffectivePointCharges2D(mag)
        self.eff_current = EffectivePointCharges3D(current)
        
        self.elec_charges = C.cast(C.pointer(self.eff_elec), C.c_void_p)
        self.mag_charges = C.cast(C.pointer(self.eff_mag), C.c_void_p)
        self.current_charges = C.cast(C.pointer(self.eff_current), C.c_void_p)

        if bounds is None:
            self.bounds = None
        else:
            self.bounds_arr = ensure_contiguous_aligned(bounds)
            self.bounds = self.bounds_arr.ctypes.data_as(dbl_p)

class FieldEvaluationArgs3D(C.Structure):
    _fields_ = [
        ("elec_charges", C.c_void_p),
        ("mag_charges", C.c_void_p),
        ("current_charges", C.c_void_p),
        ("bounds", C.POINTER(C.c_double))
    ]

    def __init__(self, elec, mag, currents, bounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert bounds is None or bounds.shape == (3, 2)
        
        self.eff_elec = EffectivePointCharges3D(elec)
        self.eff_mag = EffectivePointCharges3D(mag)
        self.eff_current = EffectivePointCurrents3D(currents)

        self.elec_charges = C.cast(C.pointer(self.eff_elec), C.c_void_p)
        self.mag_charges = C.cast(C.pointer(self.eff_mag), C.c_void_p)
        self.current_charges = C.cast(C.pointer(self.eff_current), C.c_void_p)
        
        if bounds is None:
            self.bounds = None
        else:
            self.bounds_arr = ensure_contiguous_aligned(bounds)
            self.bounds = self.bounds_arr.ctypes.data_as(dbl_p)



class FieldDerivsArgs(C.Structure):
    _fields_ = [
        ("z_interpolation", dbl_p),
        ("electrostatic_axial_coeffs", dbl_p),
        ("magnetostatic_axial_coeffs", dbl_p),
        ("N_z", C.c_size_t)
    ]

    def __init__(self, z, elec, mag, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert z.shape == (len(z),)
        assert elec.shape[0] == len(z)-1
        assert mag.shape[0] == len(z)-1

        self.z_arr = ensure_contiguous_aligned(z)
        self.elec_arr = ensure_contiguous_aligned(elec)
        self.mag_arr = ensure_contiguous_aligned(mag)
        
        self.z_interpolation = self.z_arr.ctypes.data_as(dbl_p)
        self.electrostatic_axial_coeffs = self.elec_arr.ctypes.data_as(dbl_p)
        self.magnetostatic_axial_coeffs = self.mag_arr.ctypes.data_as(dbl_p)
        self.N_z = len(z)

bounds = arr(shape=(3, 2))

times_block = arr(shape=(TRACING_BLOCK_SIZE,))
tracing_block = arr(shape=(TRACING_BLOCK_SIZE, 6))

backend_functions = {
    # triangle_contribution.c
    'kronrod_adaptive': (dbl, integration_cb_1d, dbl, dbl, vp, dbl, dbl),
     
    'ellipkm1' : (dbl, dbl),
    'ellipk' : (dbl, dbl),
    'ellipem1' : (dbl, dbl),
    'ellipe': (dbl, dbl),
    'normal_2d': (None, v2, v2, v2),
    'higher_order_normal_radial': (None, dbl, v3, v3, v3, v3, v2),
    'normal_3d': (None, arr(shape=(3,3)), v3),
    'position_and_jacobian_3d': (None, dbl, dbl, arr(ndim=2), v3, dbl_p),
    'position_and_jacobian_radial': (None, dbl, v2, v2, v2, v2, v2, dbl_p),
    'delta_position_and_jacobian_radial': (None, dbl, v2, v2, v2, v2, v2, dbl_p),
    'trace_particle': (sz, times_block, tracing_block, dbl, field_fun, bounds, dbl, vp),
    'potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'dr1_potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'dz1_potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'axial_derivatives_radial': (None, arr(ndim=2), charges_2d, jac_buffer_2d, pos_buffer_2d, sz, z_values, sz),
    'potential_radial': (dbl, v3, charges_2d, jac_buffer_2d, pos_buffer_2d, sz),
    'potential_radial_derivs': (dbl, v3, z_values, arr(ndim=3), sz),
    'flux_density_to_charge_factor': (dbl, dbl),
    'charge_radial': (dbl, arr(ndim=2), dbl),
    'field_radial': (None, v3, v3, charges_2d, jac_buffer_2d, pos_buffer_2d, sz),
    'field_radial_derivs': (None, v3, v3, z_values, arr(ndim=3), sz),
    'axial_coefficients_3d': (None, charges_3d, jac_buffer_3d, pos_buffer_3d, arr(ndim=3), arr(ndim=3), sz, z_values, arr(ndim=4), sz),
    'fill_jacobian_buffer_current_three_d': (None, lines, jac_buffer_3d, pos_buffer_3d, arr(ndim=3), sz),
    'potential_3d_derivs': (dbl, v3, z_values, arr(ndim=5), sz),
    'field_3d_derivs': (None, v3, v3, z_values, arr(ndim=5), sz),
    'current_potential_axial_radial_ring': (dbl, dbl, dbl, dbl),
    'current_potential_axial': (dbl, dbl, currents_2d, jac_buffer_3d, pos_buffer_3d, sz),
    'current_field_radial_ring': (None, dbl, dbl, dbl, dbl, v2),
    'current_field_radial': (None, v3, v3, currents_2d, jac_buffer_3d, pos_buffer_3d, sz),
    'current_axial_derivatives_radial': (None, arr(ndim=2), currents_2d, jac_buffer_3d, pos_buffer_3d, sz, z_values, sz),
    'fill_jacobian_buffer_radial': (None, jac_buffer_2d, pos_buffer_2d, vertices, sz),
    'self_potential_radial': (dbl, dbl, vp),
    'self_field_dot_normal_radial': (dbl, dbl, vp),
    'fill_matrix_radial': (None, arr(ndim=2), lines, arr(dtype=np.uint8, ndim=1), arr(ndim=1), jac_buffer_2d, pos_buffer_2d, sz, sz, C.c_int, C.c_int),
    'fill_jacobian_buffer_3d': (None, jac_buffer_3d, pos_buffer_3d, vertices, sz),
    'plane_intersection': (bool, v3, v3, arr(ndim=2), sz, arr(shape=(6,))),
    'triangle_areas': (None, vertices, arr(ndim=1), sz)
}


def ensure_contiguous_aligned(arr):
    assert isinstance(arr, np.ndarray)
    new_arr = np.require(arr, requirements=('C_CONTIGUOUS', 'ALIGNED'));
    assert not DEBUG or (new_arr is arr), "Made copy while ensuring contiguous array"
    return new_arr

# These are passed directly to scipy.LowLevelCallable, so we shouldn't wrap them in a function
# that checks the numpy arrays
numpy_wrapper_exceptions = ['self_potential_radial', 'self_field_dot_normal_radial']

for (fun, (res, *args)) in backend_functions.items():
    libfun = getattr(backend_lib, fun)
    
    if fun not in numpy_wrapper_exceptions:
        def backend_check_numpy_requirements_wrapper(*args, _cfun_reference=libfun, _cfun_name=fun):
            new_args = [ (ensure_contiguous_aligned(a) if isinstance(a, np.ndarray) else a) for a in args ]
            return _cfun_reference(*new_args)
        
        setattr(backend_lib, fun, backend_check_numpy_requirements_wrapper)
     
    libfun.restype = res
    libfun.argtypes = args


ellipkm1 = np.vectorize(backend_lib.ellipkm1)
ellipk = np.vectorize(backend_lib.ellipk)
ellipem1 = np.vectorize(backend_lib.ellipem1)
ellipe = np.vectorize(backend_lib.ellipe)

def kronrod_adaptive(fun: Callable[[float], float], a: float, b: float, 
                     epsabs: float=1.49e-08, epsrel: float=1.49e-08) -> float:
    callback = integration_cb_1d(lambda x, _: fun(x))
    return backend_lib.kronrod_adaptive(callback, a, b, vp(None), epsabs, epsrel)

def higher_order_normal_radial(alpha: float, vertices: np.ndarray) -> np.ndarray:
    assert vertices.shape == (4,3) 
    normal = np.zeros(2)
    backend_lib.higher_order_normal_radial(alpha, vertices[0], vertices[2], vertices[3], vertices[1], normal)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    return normal
    
def normal_2d(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    normal = np.zeros( (2,) )
    backend_lib.normal_2d(p1, p2, normal)
    return normal

def normal_3d(tri: np.ndarray) -> np.ndarray:
    normal = np.zeros( (3,) )
    backend_lib.normal_3d(tri, normal)
    return normal

def trace_particle_wrapper(position_: np.ndarray, velocity_: np.ndarray, 
                           fill_positions_fun: Callable[[np.ndarray, np.ndarray], int]) -> Tuple[np.ndarray, np.ndarray]:
    position = np.array(position_)
    velocity = np.array(velocity_)
    assert position.shape == (3,) and velocity.shape == (3,)
     
    N = TRACING_BLOCK_SIZE
    pos_blocks: List[np.ndarray] = []
    times_blocks: List[np.ndarray] = []
     
    times = np.zeros(TRACING_BLOCK_SIZE)
    positions = np.zeros( (TRACING_BLOCK_SIZE, 6) )
    positions[0] = np.concatenate( (position, velocity) )
    
    while True:
        N = fill_positions_fun(times, positions)
         
        # Prevent the starting positions to be both at the end of the previous block and the start
        # of the current block.
        pos_blocks.append(positions[1:N] if len(pos_blocks) > 0  else positions[:N])
        times_blocks.append(times[1:N] if len(times_blocks) > 0  else times[:N])
        
        if N != TRACING_BLOCK_SIZE:
            break
          
        times = np.zeros(TRACING_BLOCK_SIZE)
        positions = np.zeros( (TRACING_BLOCK_SIZE, 6) )
         
        positions[0] = pos_blocks[-1][-1]
        times[0] = times_blocks[-1][-1]

    assert len(pos_blocks) == len(times_blocks)
    
    # Speedup, usually no concatenation needed
    if len(pos_blocks) == 1:
        return times_blocks[0], pos_blocks[0]
    else:
        return np.concatenate(times_blocks), np.concatenate(pos_blocks)



def position_and_jacobian_3d(alpha: float, beta: float, triangle: np.ndarray) -> Tuple[float, np.ndarray]:
    assert triangle.shape == (3, 3)
     
    pos = np.zeros(3)
    jac = C.c_double(0.0)
     
    backend_lib.position_and_jacobian_3d(alpha, beta, triangle, pos, C.pointer(jac))
        
    return jac.value, pos


def position_and_jacobian_radial(alpha: float, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray) -> Tuple[float, np.ndarray]:
    assert v1.shape == (2,) or v1.shape == (3,)
    assert v2.shape == (2,) or v2.shape == (3,)
    assert v3.shape == (2,) or v3.shape == (3,)
    assert v4.shape == (2,) or v4.shape == (3,)
    
    assert all([v.shape == (2,) or v[1] == 0. for v in [v1,v2,v3,v4]])
    
    pos = np.zeros(2)
    jac = C.c_double(0.0)
     
    backend_lib.position_and_jacobian_radial(alpha, v1[:2], v2[:2], v3[:2], v4[:2], pos, C.pointer(jac))
    
    return jac.value, pos

def delta_position_and_jacobian_radial(alpha: float, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray) -> Tuple[float, np.ndarray]:
    assert v1.shape == (2,) or v1.shape == (3,)
    assert v2.shape == (2,) or v2.shape == (3,)
    assert v3.shape == (2,) or v3.shape == (3,)
    assert v4.shape == (2,) or v4.shape == (3,)
    
    assert all([v.shape == (2,) or v[1] == 0. for v in [v1,v2,v3,v4]])
    
    pos = np.zeros(2)
    jac = C.c_double(0.0)
     
    backend_lib.delta_position_and_jacobian_radial(alpha, v1[:2], v2[:2], v3[:2], v4[:2], pos, C.pointer(jac))
    
    return jac.value, pos

def wrap_field_fun(ff: Callable) -> Callable:
    def field_fun_wrapper(pos, vel, _, elec_out, mag_out):
        elec, mag = ff(np.array([pos[0], pos[1], pos[2]]), np.array([vel[0], vel[1], vel[2]]))
        assert elec.shape == (3,) and mag.shape == (3,)

        elec_out[0] = elec[0]
        elec_out[1] = elec[1]
        elec_out[2] = elec[2]
        
        mag_out[0] = mag[0]
        mag_out[1] = mag[1]
        mag_out[2] = mag[2]
    
    return field_fun(field_fun_wrapper)

def trace_particle(position: np.ndarray, velocity: np.ndarray, charge_over_mass: float, field, bounds: np.ndarray, atol: float, args =None) -> Tuple[np.ndarray, np.ndarray]:
    bounds = np.array(bounds)
     
    return trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle(T, P, charge_over_mass, field, bounds, atol, args))

def trace_particle_radial_derivs(position: np.ndarray, velocity: np.ndarray, charge_over_mass: float, 
                                 bounds: np.ndarray, atol: float, z: np.ndarray, 
                                 elec_coeffs: np.ndarray, mag_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert elec_coeffs.shape == (len(z)-1, DERIV_2D_MAX, 6)
    assert mag_coeffs.shape == (len(z)-1, DERIV_2D_MAX, 6)
    
    bounds = np.array(bounds)

    if bounds.shape[0] == 2:
        bounds = np.array([bounds[0], bounds[0], bounds[1]])
    
    times, positions = trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle_radial_derivs(T, P, charge_over_mass, bounds, atol, z, elec_coeffs, mag_coeffs, len(z)))
    
    return times, positions

def trace_particle_3d_derivs(position: np.ndarray, velocity: np.ndarray, charge_over_mass: float, 
                             bounds: np.ndarray, atol: float, z: np.ndarray, 
                             electrostatic_coeffs: np.ndarray, magnetostatic_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert electrostatic_coeffs.shape == (len(z)-1, 2, NU_MAX, M_MAX, 4)
    assert magnetostatic_coeffs.shape == (len(z)-1, 2, NU_MAX, M_MAX, 4)
    
    bounds = np.array(bounds)
     
    return trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle_3d_derivs(T, P, charge_over_mass, bounds, atol, z, electrostatic_coeffs, magnetostatic_coeffs, len(z)))

def potential_radial_ring(r0: float, z0: float, delta_r: float, delta_z: float) -> float:
    return backend_lib.potential_radial_ring(r0, z0, delta_r, delta_z, None)

def dr1_potential_radial_ring(r0: float, z0: float, delta_r: float, delta_z: float) -> float:
    return backend_lib.dr1_potential_radial_ring(r0, z0, delta_r, delta_z, None)

def dz1_potential_radial_ring(r0: float, z0: float, delta_r: float, delta_z: float) -> float:
    return backend_lib.dz1_potential_radial_ring(r0, z0, delta_r, delta_z, None)

def axial_derivatives_radial(z: np.ndarray, charges: np.ndarray, jac_buffer: np.ndarray, pos_buffer: np.ndarray) -> np.ndarray:
    derivs = np.zeros( (z.size, DERIV_2D_MAX) )
    
    assert jac_buffer.shape == (len(charges), N_QUAD_2D)
    assert pos_buffer.shape == (len(charges), N_QUAD_2D, 2)
    assert charges.shape == (len(charges),)
     
    backend_lib.axial_derivatives_radial(derivs,charges, jac_buffer, pos_buffer, len(charges), z, len(z))
    return derivs

def potential_radial(point: np.ndarray, charges: np.ndarray, jac_buffer: np.ndarray, pos_buffer: np.ndarray) -> float:
    assert point.shape == (3,)
    assert jac_buffer.shape == (len(charges), N_QUAD_2D)
    assert pos_buffer.shape == (len(charges), N_QUAD_2D, 2)
    return backend_lib.potential_radial(point.astype(np.float64), charges, jac_buffer, pos_buffer, len(charges))

def potential_radial_derivs(point: np.ndarray, z: np.ndarray, coeffs: np.ndarray) -> float:
    assert coeffs.shape == (len(z)-1, DERIV_2D_MAX, 6)
    return backend_lib.potential_radial_derivs(point.astype(np.float64), z, coeffs, len(z))

def charge_radial(vertices: np.ndarray, charge: float) -> float:
    assert vertices.shape == (len(vertices), 3)
    return backend_lib.charge_radial(vertices, charge)

def field_radial(point: np.ndarray, charges: np.ndarray, jac_buffer: np.ndarray, pos_buffer: np.ndarray) -> np.ndarray:
    assert point.shape == (3,)
    assert jac_buffer.shape == (len(charges), N_QUAD_2D)
    assert pos_buffer.shape == (len(charges), N_QUAD_2D, 2)
    assert charges.shape == (len(charges),)
        
    field = np.zeros( (3,) )
    backend_lib.field_radial(point.astype(np.float64), field, charges, jac_buffer, pos_buffer, len(charges))
    return field

def field_radial_derivs(point: np.ndarray, z: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    assert point.shape == (3,)
    assert coeffs.shape == (len(z)-1, DERIV_2D_MAX, 6)
    field = np.zeros( (3,) )
    backend_lib.field_radial_derivs(point.astype(np.float64), field, z, coeffs, len(z))
    return field

def flux_density_to_charge_factor(K: float) -> float:
    return backend_lib.flux_density_to_charge_factor(K)

def axial_coefficients_3d(charges: np.ndarray, jacobian_buffer: np.ndarray, pos_buffer: np.ndarray, z: np.ndarray) -> np.ndarray:
    if len(charges) == 0:
        return np.zeros( (len(z), 2, NU_MAX, M_MAX) )
     
    assert jacobian_buffer.shape == (len(charges), N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (len(charges), N_TRIANGLE_QUAD, 3)
    
    output_coeffs = np.zeros( (len(z), 2, NU_MAX, M_MAX) )
      
    trig_cos_buffer = np.zeros( (len(charges), N_TRIANGLE_QUAD, M_MAX) )
    trig_sin_buffer = np.zeros( (len(charges), N_TRIANGLE_QUAD, M_MAX) )
     
    backend_lib.axial_coefficients_3d(charges, jacobian_buffer, pos_buffer, trig_cos_buffer, trig_sin_buffer,
                                      len(charges), z, output_coeffs, len(z))
      
    return output_coeffs

def fill_jacobian_buffer_current_three_d(lines):
    assert lines.shape == (len(lines), 2, 3)

    jacobians = np.zeros( (len(lines), N_QUAD_2D) )
    positions = np.zeros( (len(lines), N_QUAD_2D, 3) )
    directions = np.zeros( (len(lines), N_QUAD_2D, 3) )

    backend_lib.fill_jacobian_buffer_current_three_d(lines, jacobians, positions, directions, len(lines))

    return jacobians, positions, directions

def potential_3d_derivs(point: np.ndarray, z: np.ndarray, coeffs: np.ndarray) -> float:
    assert coeffs.shape == (len(z)-1, 2, NU_MAX, M_MAX, 4)
    assert point.shape == (3,)
    
    return backend_lib.potential_3d_derivs(point.astype(np.float64), z, coeffs, len(z))

def field_3d_derivs(point: np.ndarray, z: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    assert point.shape == (3,)
    assert coeffs.shape == (len(z)-1, 2, NU_MAX, M_MAX, 4)

    field = np.zeros( (3,) )
    backend_lib.field_3d_derivs(point.astype(np.float64), field, z, coeffs, len(z))
    return field

def current_potential_axial_radial_ring(z0: float, r: float, z: float) -> float:
    return backend_lib.current_potential_axial_radial_ring(z0, r, z)

def current_potential_axial(z: float, currents: np.ndarray, jac_buffer: np.ndarray, pos_buffer: np.ndarray) -> float:
    N = len(currents)
    assert currents.shape == (N,)
    assert jac_buffer.shape == (N, N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (N, N_TRIANGLE_QUAD, 3)
      
    assert np.all(pos_buffer[:, :, 1] == 0.)
    
    return backend_lib.current_potential_axial(z, currents, jac_buffer, pos_buffer, N)

def current_field_radial_ring(x0: float, y0: float, x: float, y: float) -> np.ndarray:
    res = np.zeros((2,))
    backend_lib.current_field_radial_ring(x0, y0, x, y, res)
    return res

def current_field_radial(p0: np.ndarray, currents: np.ndarray, jac_buffer: np.ndarray, pos_buffer: np.ndarray) -> np.ndarray:
    assert p0.shape == (3,)
    N = len(currents)
    assert currents.shape == (N,)
    assert jac_buffer.shape == (N, N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (N, N_TRIANGLE_QUAD, 3)
     
    assert np.all(pos_buffer[:, :, 1] == 0.)
    
    result = np.zeros( (3,) )
    backend_lib.current_field_radial(p0, result, currents, jac_buffer, pos_buffer, N)
    return result

def current_axial_derivatives_radial(z: np.ndarray, currents: np.ndarray, jac_buffer: np.ndarray, pos_buffer: np.ndarray) -> np.ndarray:
    N_z = len(z)
    N_vertices = len(currents)

    assert z.shape == (N_z,)
    assert currents.shape == (N_vertices,)
    assert jac_buffer.shape == (N_vertices, N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (N_vertices, N_TRIANGLE_QUAD, 3)
    
    derivs = np.zeros( (z.size, DERIV_2D_MAX) )
    backend_lib.current_axial_derivatives_radial(derivs, currents, jac_buffer, pos_buffer, N_vertices, z, N_z)
    return derivs


def fill_jacobian_buffer_radial(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert vertices.shape == (len(vertices), 4, 3)
    assert np.all(vertices[:, :, 1] == 0.)
        
    N = len(vertices)
    jac_buffer = np.zeros( (N, N_QUAD_2D) )
    pos_buffer = np.zeros( (N, N_QUAD_2D, 2) )
     
    backend_lib.fill_jacobian_buffer_radial(jac_buffer, pos_buffer, vertices, N)
    
    return jac_buffer, pos_buffer

def self_potential_radial(vertices: np.ndarray) -> float:
    assert vertices.shape == (4,3) and vertices.dtype == np.double
    user_data = vertices.ctypes.data_as(C.c_void_p)
    low_level = LowLevelCallable(backend_lib.self_potential_radial, user_data=user_data) # type: ignore
    return quad(low_level, -1, 1, points=(0,), epsabs=1e-9, epsrel=1e-9, limit=250)[0] # type: ignore

class SelfFieldDotNormalRadialArgs(C.Structure):
    _fields_ = [("line_points", C.POINTER(C.c_double)), ("K", C.c_double)]

def self_field_dot_normal_radial(vertices: np.ndarray, K: float) -> float:
    assert vertices.shape == (4,3) and vertices.dtype == np.double
    user_data = vertices.ctypes.data_as(C.c_void_p)

    args = SelfFieldDotNormalRadialArgs()
    args.K = float(K)
    args.line_points = vertices.ctypes.data_as(dbl_p)

    user_data = C.cast(C.pointer(args), vp)
        
    low_level = LowLevelCallable(backend_lib.self_field_dot_normal_radial, user_data=user_data) # type: ignore
    return quad(low_level, -1, 1, points=(0,), epsabs=1e-9, epsrel=1e-9, limit=250)[0] # type: ignore



def fill_matrix_radial(matrix: np.ndarray,
                    lines: np.ndarray,
                    excitation_types: np.ndarray,
                    excitation_values: np.ndarray,
                    jac_buffer: np.ndarray,
                    pos_buffer: np.ndarray,
                    start_index: int, end_index: int):
    N = len(lines)
    assert np.all(lines[:, :, 1] == 0.0)
    assert matrix.shape[0] == N and matrix.shape[1] == N and matrix.shape[0] == matrix.shape[1]
    assert lines.shape == (N, 4, 3)
    assert excitation_types.shape == (N,)
    assert excitation_values.shape == (N,)
    assert jac_buffer.shape == (N, N_QUAD_2D)
    assert pos_buffer.shape == (N, N_QUAD_2D, 2)
    assert 0 <= start_index < N and 0 <= end_index < N and start_index < end_index
     
    backend_lib.fill_matrix_radial(matrix, lines, excitation_types, excitation_values, jac_buffer, pos_buffer, N, matrix.shape[0], start_index, end_index)

def fill_jacobian_buffer_3d(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = len(vertices)
    assert vertices.shape == (N, 3, 3)
    jac_buffer = np.zeros( (N, N_TRIANGLE_QUAD) )
    pos_buffer = np.zeros( (N, N_TRIANGLE_QUAD, 3) )
    
    backend_lib.fill_jacobian_buffer_3d(jac_buffer, pos_buffer, vertices, N)

    return jac_buffer, pos_buffer


def fill_matrix_3d(matrix: np.ndarray,
                vertices: np.ndarray,
                excitation_types: np.ndarray,
                excitation_values: np.ndarray,
                jac_buffer: np.ndarray,
                pos_buffer: np.ndarray,
                start_index: int, end_index: int):
    N = len(vertices)
    assert matrix.shape[0] == N and matrix.shape[1] == N and matrix.shape[0] == matrix.shape[1]
    assert vertices.shape == (N, 3, 3)
    assert excitation_types.shape == (N,)
    assert excitation_values.shape == (N,)
    assert jac_buffer.shape == (N, N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (N, N_TRIANGLE_QUAD, 3)
    assert 0 <= start_index < N and 0 <= end_index < N and start_index <= end_index
     
    backend_lib.fill_matrix_3d(matrix, vertices, excitation_types, excitation_values, jac_buffer, pos_buffer, N, matrix.shape[0], start_index, end_index)

def plane_intersection(positions: np.ndarray, p0: np.ndarray, normal: np.ndarray) -> np.ndarray:
    assert p0.shape == (3,)
    assert normal.shape == (3,)
    assert positions.shape == (len(positions), 6)
    
    result = np.zeros(6)
    found = backend_lib.plane_intersection(p0, normal, positions, len(positions), result)
     
    if not found:
        raise ValueError("Plane intersection not found. Does the trajectory actually cross the plane?")
     
    return result

def triangle_areas(triangles: np.ndarray) -> np.ndarray:
    assert triangles.shape == (len(triangles), 3, 3)
    out = np.zeros(len(triangles))
    backend_lib.triangle_areas(triangles, out, len(triangles))
    return out

    
    



    

    


    

