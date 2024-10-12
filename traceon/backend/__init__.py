"""The backend module is a small wrapper around code implemented in C. All performance critical code is 
implemented in the C backend. As a user you should not use these functions directly yourself."""

import importlib
import ctypes as C
import os.path as path
import platform

from numpy.ctypeslib import ndpointer
import numpy as np

from .. import logging

DEBUG = False

if DEBUG:
    logging.set_log_level(logging.LogLevel.DEBUG)

## Attempt 1: load local
if platform.system() in ['Linux', 'Darwin']:
    local_path = path.join(path.dirname(__file__), 'traceon_backend.so')
    global_file = 'traceon/backend/traceon_backend.abi3.so'
else:
    local_path = path.join(path.dirname(__file__), 'traceon_backend.dll')
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
def arr(*args, dtype=np.float64, **kwargs):
    return ndpointer(*args, dtype=dtype, flags=('C_CONTIGUOUS', 'ALIGNED'), **kwargs);

# Pass one dimensional Numpy array to C
v2 = arr(shape=(2,))
v3 = arr(shape=(3,))

dbl = C.c_double
dbl_p = C.POINTER(dbl)
vp = C.c_void_p
sz = C.c_size_t

integration_cb_1d = C.CFUNCTYPE(dbl, dbl, vp)
integration_cb_2d = C.CFUNCTYPE(dbl, dbl, dbl, dbl, dbl, vp)
integration_cb_3d = C.CFUNCTYPE(dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp)
field_fun = C.CFUNCTYPE(None, C.POINTER(dbl), C.POINTER(dbl), vp);

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
         
        self.charges = ensure_contiguous_aligned(eff.charges).ctypes.data_as(dbl_p)
        self.jacobians = ensure_contiguous_aligned(eff.jacobians).ctypes.data_as(dbl_p)
        self.positions = ensure_contiguous_aligned(eff.positions).ctypes.data_as(dbl_p)
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
        super(EffectivePointCharges3D, self).__init__(*args, **kwargs)
         
        self.charges = ensure_contiguous_aligned(eff.charges).ctypes.data_as(dbl_p)
        self.jacobians = ensure_contiguous_aligned(eff.jacobians).ctypes.data_as(dbl_p)
        self.positions = ensure_contiguous_aligned(eff.positions).ctypes.data_as(dbl_p)
        self.N = len(eff)

bounds = arr(shape=(3, 2))

times_block = arr(shape=(TRACING_BLOCK_SIZE,))
tracing_block = arr(shape=(TRACING_BLOCK_SIZE, 6))

backend_functions = {
    # triangle_contribution.c
    'potential_triangle': (dbl, v3, v3, v3, v3),
    'self_potential_triangle_v0': (dbl, v3, v3, v3),
    'self_potential_triangle': (dbl, v3, v3, v3, v3),
    'flux_triangle': (dbl, v3, v3, v3, v3, v3),
     
    'ellipkm1' : (dbl, dbl),
    'ellipk' : (dbl, dbl),
    'ellipem1' : (dbl, dbl),
    'ellipe': (dbl, dbl),
    'tanh_sinh_integration': (dbl, integration_cb_1d, dbl, dbl, dbl, dbl, vp),
    'normal_2d': (None, v2, v2, v2),
    'higher_order_normal_radial': (None, dbl, v2, v2, v2, v2, v2),
    'normal_3d': (None, dbl, dbl, arr(shape=(3,3)), v3),
    'position_and_jacobian_3d': (None, dbl, dbl, arr(ndim=2), v3, dbl_p),
    'position_and_jacobian_radial': (None, dbl, v2, v2, v2, v2, v2, dbl_p),
    'trace_particle': (sz, times_block, tracing_block, field_fun, bounds, dbl, vp),
    'potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'dr1_potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'dz1_potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'axial_derivatives_radial_ring': (None, arr(ndim=2), charges_2d, jac_buffer_2d, pos_buffer_2d, sz, z_values, sz),
    'potential_radial': (dbl, v2, charges_2d, jac_buffer_2d, pos_buffer_2d, sz),
    'potential_radial_derivs': (dbl, v2, z_values, arr(ndim=3), sz),
    'flux_density_to_charge_factor': (dbl, dbl),
    'charge_radial': (dbl, arr(ndim=2), dbl),
    'field_radial': (None, v3, v3, charges_2d, jac_buffer_2d, pos_buffer_2d, sz),
    'combine_elec_magnetic_field': (None, v3, v3, v3, v3, v3),
    'trace_particle_radial': (sz, times_block, tracing_block, bounds, dbl, dbl_p, EffectivePointCharges2D, EffectivePointCharges2D, EffectivePointCharges3D),
    'field_radial_derivs': (None, v3, v3, z_values, arr(ndim=3), sz),
    'trace_particle_radial_derivs': (sz, times_block, tracing_block, bounds, dbl, z_values, radial_coeffs, radial_coeffs, sz),
    'dx1_potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'dy1_potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'dz1_potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'axial_coefficients_3d': (None, charges_3d, jac_buffer_3d, pos_buffer_3d, arr(ndim=3), arr(ndim=3), sz, z_values, arr(ndim=4), sz),
    'potential_3d': (dbl, v3, charges_3d, jac_buffer_3d, pos_buffer_3d, sz),
    'potential_3d_derivs': (dbl, v3, z_values, arr(ndim=5), sz),
    'field_3d': (None, v3, v3, charges_3d, jac_buffer_3d, pos_buffer_3d, sz),
    'trace_particle_3d': (sz, times_block, tracing_block, bounds, dbl, EffectivePointCharges3D, EffectivePointCharges3D, dbl_p),
    'field_3d_derivs': (None, v3, v3, z_values, arr(ndim=5), sz),
    'trace_particle_3d_derivs': (sz, times_block, tracing_block, bounds, dbl, z_values, arr(ndim=5), arr(ndim=5), sz),
    'current_potential_axial_radial_ring': (dbl, dbl, dbl, dbl),
    'current_potential_axial': (dbl, dbl, currents_2d, jac_buffer_3d, pos_buffer_3d, sz),
    'current_field_radial_ring': (None, dbl, dbl, dbl, dbl, v2),
    'current_field': (None, v3, v3, currents_2d, jac_buffer_3d, pos_buffer_3d, sz),
    'current_axial_derivatives_radial_ring': (None, arr(ndim=2), currents_2d, jac_buffer_3d, pos_buffer_3d, sz, z_values, sz),
    'fill_jacobian_buffer_radial': (None, jac_buffer_2d, pos_buffer_2d, vertices, sz),
    'fill_matrix_radial': (None, arr(ndim=2), lines, arr(dtype=C.c_uint8, ndim=1), arr(ndim=1), jac_buffer_2d, pos_buffer_2d, sz, sz, C.c_int, C.c_int),
    'fill_jacobian_buffer_3d': (None, jac_buffer_3d, pos_buffer_3d, vertices, sz),
    'fill_matrix_3d': (None, arr(ndim=2), vertices, arr(dtype=C.c_uint8, ndim=1), arr(ndim=1), jac_buffer_3d, pos_buffer_3d, sz, sz, C.c_int, C.c_int),
    'plane_intersection': (bool, v3, v3, arr(ndim=2), sz, arr(shape=(6,))),
    'line_intersection': (bool, v2, v2, arr(ndim=2), sz, arr(shape=(4,)))
}


def ensure_contiguous_aligned(arr):
    assert isinstance(arr, np.ndarray)
    new_arr = np.require(arr, requirements=('C_CONTIGUOUS', 'ALIGNED'));
    assert not DEBUG or (new_arr is arr), "Made copy while ensuring contiguous array"
    return new_arr

for (fun, (res, *args)) in backend_functions.items():
    libfun = getattr(backend_lib, fun)

    def backend_check_numpy_requirements_wrapper(*args, _cfun_reference=libfun, _cfun_name=fun):
        new_args = [ (ensure_contiguous_aligned(a) if isinstance(a, np.ndarray) else a) for a in args ]
        return _cfun_reference(*new_args)
      
    setattr(backend_lib, fun, backend_check_numpy_requirements_wrapper)
     
    libfun.restype = res
    libfun.argtypes = args

self_potential_triangle_v0 = backend_lib.self_potential_triangle_v0
self_potential_triangle = backend_lib.self_potential_triangle
potential_triangle = backend_lib.potential_triangle
flux_triangle = backend_lib.flux_triangle

ellipkm1 = np.vectorize(backend_lib.ellipkm1)
ellipk = np.vectorize(backend_lib.ellipk)
ellipem1 = np.vectorize(backend_lib.ellipem1)
ellipe = np.vectorize(backend_lib.ellipe)

def tanh_sinh_integration(integrand, x_min, x_max, epsabs, epsrel):
    wrapped = lambda x, args: integrand(x)
    return backend_lib.tanh_sinh_integration(integration_cb_1d(wrapped), x_min, x_max, epsabs, epsrel, None);

def higher_order_normal_radial(alpha, vertices):
    normal = np.zeros(2)
    backend_lib.higher_order_normal_radial(alpha, vertices[0, :2], vertices[2, :2], vertices[3, :2], vertices[1, :2], normal)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    return normal
    
def normal_2d(p1, p2):
    normal = np.zeros( (2,) )
    backend_lib.normal_2d(p1, p2, normal)
    return normal

# Remove the last argument, which is usually a void pointer to optional data
# passed to the function. In Python we don't need this functionality
# as we can simply use closures.
def remove_arg(fun):
    return lambda *args: fun(*args[:-1])

def normal_3d(alpha, beta, tri):
    normal = np.zeros( (3,) )
    backend_lib.normal_3d(alpha, beta, tri, normal)
    return normal
   
def _vec_2d_to_3d(vec):
    assert vec.shape == (2,) or vec.shape == (3,)
     
    if vec.shape == (2,):
        return np.array([vec[0], 0., vec[1]])
    
    return vec

def _vec_3d_to_2d(vec):
    assert vec.shape == (2,) or vec.shape == (3,)
     
    if vec.shape == (3,):
        return np.array([vec[0], vec[2]])
    
    return vec


def trace_particle_wrapper(position, velocity, fill_positions_fun):
    position = _vec_2d_to_3d(position)
    velocity = _vec_2d_to_3d(velocity)
     
    assert position.shape == (3,) and velocity.shape == (3,)
     
    N = TRACING_BLOCK_SIZE
    pos_blocks = []
    times_blocks = []
     
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

def wrap_field_fun(ff):

    def wrapper(y, result, _):
        field = ff(y[0], y[1], y[2], y[3], y[4], y[5])
        assert field.shape == (3,)
        result[0] = field[0]
        result[1] = field[1]
        result[2] = field[2]
    
    return field_fun(wrapper)

def position_and_jacobian_3d(alpha, beta, triangle):
    assert triangle.shape == (3, 3)
     
    pos = np.zeros(3)
    jac = C.c_double(0.0)
     
    backend_lib.position_and_jacobian_3d(alpha, beta, triangle, pos, C.pointer(jac))
        
    return jac.value, pos


def position_and_jacobian_radial(alpha, v1, v2, v3, v4):
    assert v1.shape == (2,) or v1.shape == (3,)
    assert v2.shape == (2,) or v2.shape == (3,)
    assert v3.shape == (2,) or v3.shape == (3,)
    assert v4.shape == (2,) or v4.shape == (3,)
    
    pos = np.zeros(2)
    jac = C.c_double(0.0)
     
    backend_lib.position_and_jacobian_radial(alpha, v1[:2], v2[:2], v3[:2], v4[:2], pos, C.pointer(jac))
    
    return jac.value, pos


def trace_particle(position, velocity, field, bounds, atol):
    bounds = np.array(bounds)
    
    return trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle(T, P, wrap_field_fun(field), bounds, atol, None))

def trace_particle_radial(position, velocity, bounds, atol, eff_elec, eff_mag, eff_current, field_bounds=None):
    
    eff_elec = EffectivePointCharges2D(eff_elec)
    eff_mag = EffectivePointCharges2D(eff_mag)
    eff_current = EffectivePointCharges3D(eff_current)
    
    bounds = np.array(bounds)
     
    field_bounds = field_bounds.ctypes.data_as(dbl_p) if field_bounds is not None else None
     
    times, positions = trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle_radial(T, P, bounds, atol, field_bounds, eff_elec, eff_mag, eff_current))
    
    return times, positions

def trace_particle_radial_derivs(position, velocity, bounds, atol, z, elec_coeffs, mag_coeffs):
    assert elec_coeffs.shape == (len(z)-1, DERIV_2D_MAX, 6)
    assert mag_coeffs.shape == (len(z)-1, DERIV_2D_MAX, 6)
    
    bounds = np.array(bounds)

    if bounds.shape[0] == 2:
        bounds = np.array([bounds[0], bounds[0], bounds[1]])
    
    times, positions = trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle_radial_derivs(T, P, bounds, atol, z, elec_coeffs, mag_coeffs, len(z)))
    
    return times, positions

def trace_particle_3d(position, velocity, bounds, atol, eff_elec, eff_mag, field_bounds=None):
    assert position.shape == (3,)
    assert velocity.shape == (3,)
    assert field_bounds is None or field_bounds.shape == (3,2)
     
    bounds = np.array(bounds)
    
    field_bounds = field_bounds.ctypes.data_as(dbl_p) if field_bounds is not None else None

    eff_elec = EffectivePointCharges3D(eff_elec)
    eff_mag = EffectivePointCharges3D(eff_mag)
     
    return trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle_3d(T, P, bounds, atol, eff_elec, eff_mag, field_bounds))

def trace_particle_3d_derivs(position, velocity, bounds, atol, z, electrostatic_coeffs, magnetostatic_coeffs):
    assert position.shape == (3,)
    assert velocity.shape == (3,)
    assert electrostatic_coeffs.shape == (len(z)-1, 2, NU_MAX, M_MAX, 4)
    assert magnetostatic_coeffs.shape == (len(z)-1, 2, NU_MAX, M_MAX, 4)
    
    bounds = np.array(bounds)
     
    return trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle_3d_derivs(T, P, bounds, atol, z, electrostatic_coeffs, magnetostatic_coeffs, len(z)))

potential_radial_ring = lambda *args: backend_lib.potential_radial_ring(*args, None)
dr1_potential_radial_ring = lambda *args: backend_lib.dr1_potential_radial_ring(*args, None)
dz1_potential_radial_ring = lambda *args: backend_lib.dz1_potential_radial_ring(*args, None)

def axial_derivatives_radial_ring(z, charges, jac_buffer, pos_buffer):
    derivs = np.zeros( (z.size, DERIV_2D_MAX) )
    
    assert jac_buffer.shape == (len(charges), N_QUAD_2D)
    assert pos_buffer.shape == (len(charges), N_QUAD_2D, 2)
    assert charges.shape == (len(charges),)
     
    backend_lib.axial_derivatives_radial_ring(derivs,charges, jac_buffer, pos_buffer, len(charges), z, len(z))
    return derivs

def potential_radial(point, charges, jac_buffer, pos_buffer):
    assert point.shape == (2,)
    assert jac_buffer.shape == (len(charges), N_QUAD_2D)
    assert pos_buffer.shape == (len(charges), N_QUAD_2D, 2)
    return backend_lib.potential_radial(point.astype(np.float64), charges, jac_buffer, pos_buffer, len(charges))

def potential_radial_derivs(point, z, coeffs):
    assert coeffs.shape == (len(z)-1, DERIV_2D_MAX, 6)
    return backend_lib.potential_radial_derivs(point.astype(np.float64), z, coeffs, len(z))

def charge_radial(vertices, charge):
    assert vertices.shape == (len(vertices), 3)
    return backend_lib.charge_radial(vertices, charge)

def field_radial(point, charges, jac_buffer, pos_buffer):
    point = _vec_2d_to_3d(point)
    assert jac_buffer.shape == (len(charges), N_QUAD_2D)
    assert pos_buffer.shape == (len(charges), N_QUAD_2D, 2)
    assert charges.shape == (len(charges),)
    field = np.zeros( (3,) )
    backend_lib.field_radial(point.astype(np.float64), field, charges, jac_buffer, pos_buffer, len(charges))
    return _vec_3d_to_2d(field)

def combine_elec_magnetic_field(vel, elec, mag, current):
    result = np.zeros( (3,) )
    backend_lib.combine_elec_magnetic_field(vel, elec, mag, current, result)
    return result

def field_radial_derivs(point, z, coeffs):
    point = _vec_2d_to_3d(point)
    assert coeffs.shape == (len(z)-1, DERIV_2D_MAX, 6)
    field = np.zeros( (3,) )
    backend_lib.field_radial_derivs(point.astype(np.float64), field, z, coeffs, len(z))
    return _vec_3d_to_2d(field)

dx1_potential_3d_point = remove_arg(backend_lib.dx1_potential_3d_point)
dy1_potential_3d_point = remove_arg(backend_lib.dy1_potential_3d_point)
dz1_potential_3d_point = remove_arg(backend_lib.dz1_potential_3d_point)
potential_3d_point = remove_arg(backend_lib.potential_3d_point)
flux_density_to_charge_factor = backend_lib.flux_density_to_charge_factor

def axial_coefficients_3d(charges, jacobian_buffer, pos_buffer, z):
    assert jacobian_buffer.shape == (len(charges), N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (len(charges), N_TRIANGLE_QUAD, 3)
    
    output_coeffs = np.zeros( (len(z), 2, NU_MAX, M_MAX) )
      
    trig_cos_buffer = np.zeros( (len(charges), N_TRIANGLE_QUAD, M_MAX) )
    trig_sin_buffer = np.zeros( (len(charges), N_TRIANGLE_QUAD, M_MAX) )
     
    backend_lib.axial_coefficients_3d(charges, 
        jacobian_buffer, pos_buffer, trig_cos_buffer, trig_sin_buffer,
        len(charges), z, output_coeffs, len(z))
      
    return output_coeffs

def potential_3d(point, charges, jac_buffer, pos_buffer):
    assert point.shape == (3,)
    N = len(charges)
    assert charges.shape == (N,)
    assert jac_buffer.shape == (N, N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (N, N_TRIANGLE_QUAD, 3)
    
    return backend_lib.potential_3d(point.astype(np.float64), charges, jac_buffer, pos_buffer, N)

def potential_3d_derivs(point, z, coeffs):
    assert coeffs.shape == (len(z)-1, 2, NU_MAX, M_MAX, 4)
    assert point.shape == (3,)
    
    return backend_lib.potential_3d_derivs(point.astype(np.float64), z, coeffs, len(z))

def field_3d(point, charges, jacobian_buffer, position_buffer):
    N = len(charges)
    assert point.shape == (3,)
    assert jacobian_buffer.shape == (N, N_TRIANGLE_QUAD)
    assert position_buffer.shape == (N, N_TRIANGLE_QUAD, 3)
     
    field = np.zeros( (3,) )
    backend_lib.field_3d(point.astype(np.float64), field, charges, jacobian_buffer, position_buffer, N)
    return field

def field_3d_derivs(point, z, coeffs):
    assert point.shape == (3,)
    assert coeffs.shape == (len(z)-1, 2, NU_MAX, M_MAX, 4)

    field = np.zeros( (3,) )
    backend_lib.field_3d_derivs(point.astype(np.float64), field, z, coeffs, len(z))

current_potential_axial_radial_ring = backend_lib.current_potential_axial_radial_ring

def current_potential_axial(z, currents, jac_buffer, pos_buffer):
    N = len(currents)
    assert currents.shape == (N,)
    assert jac_buffer.shape == (N, N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (N, N_TRIANGLE_QUAD, 3)
      
    assert np.all(pos_buffer[:, :, 2] == 0.)
    
    return backend_lib.current_potential_axial(z, currents, jac_buffer, pos_buffer, N)

def current_field_radial_ring(x0, y0, x, y):
    res = np.zeros( (2,) )
    backend_lib.current_field_radial_ring(x0, y0, x, y, res)
    return res

def current_field(p0, currents, jac_buffer, pos_buffer):
    assert p0.shape == (3,)
    N = len(currents)
    assert currents.shape == (N,)
    assert jac_buffer.shape == (N, N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (N, N_TRIANGLE_QUAD, 3)
     
    assert np.all(pos_buffer[:, :, 2] == 0.)
    
    result = np.zeros( (3,) )
    backend_lib.current_field(p0, result, currents, jac_buffer, pos_buffer, N)
    return result

def current_axial_derivatives_radial_ring(z, currents, jac_buffer, pos_buffer):
    N_z = len(z)
    N_vertices = len(currents)

    assert z.shape == (N_z,)
    assert currents.shape == (N_vertices,)
    assert jac_buffer.shape == (N_vertices, N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (N_vertices, N_TRIANGLE_QUAD, 3)
    
    derivs = np.zeros( (z.size, DERIV_2D_MAX) )
    backend_lib.current_axial_derivatives_radial_ring(derivs, currents, jac_buffer, pos_buffer, N_vertices, z, N_z)
    return derivs


def fill_jacobian_buffer_radial(vertices):
    N = len(vertices)
    jac_buffer = np.zeros( (N, N_QUAD_2D) )
    pos_buffer = np.zeros( (N, N_QUAD_2D, 2) )
     
    backend_lib.fill_jacobian_buffer_radial(jac_buffer, pos_buffer, vertices, N)
    
    return jac_buffer, pos_buffer

def fill_matrix_radial(matrix, lines, excitation_types, excitation_values, jac_buffer, pos_buffer, start_index, end_index):
    N = len(lines)
    assert matrix.shape[0] == N and matrix.shape[1] == N and matrix.shape[0] == matrix.shape[1]
    assert lines.shape == (N, 4, 3)
    assert excitation_types.shape == (N,)
    assert excitation_values.shape == (N,)
    assert jac_buffer.shape == (N, N_QUAD_2D)
    assert pos_buffer.shape == (N, N_QUAD_2D, 2)
    assert 0 <= start_index < N and 0 <= end_index < N and start_index < end_index
     
    backend_lib.fill_matrix_radial(matrix, lines, excitation_types, excitation_values, jac_buffer, pos_buffer, N, matrix.shape[0], start_index, end_index)

def fill_jacobian_buffer_3d(vertices):
    N = len(vertices)
    assert vertices.shape == (N, 3, 3)
    jac_buffer = np.zeros( (N, N_TRIANGLE_QUAD) )
    pos_buffer = np.zeros( (N, N_TRIANGLE_QUAD, 3) )
    
    backend_lib.fill_jacobian_buffer_3d(jac_buffer, pos_buffer, vertices, N)

    return jac_buffer, pos_buffer



def fill_matrix_3d(matrix, vertices, excitation_types, excitation_values, jac_buffer, pos_buffer, start_index, end_index):
    N = len(vertices)
    assert matrix.shape[0] == N and matrix.shape[1] == N and matrix.shape[0] == matrix.shape[1]
    assert vertices.shape == (N, 3, 3)
    assert excitation_types.shape == (N,)
    assert excitation_values.shape == (N,)
    assert jac_buffer.shape == (N, N_TRIANGLE_QUAD)
    assert pos_buffer.shape == (N, N_TRIANGLE_QUAD, 3)
    assert 0 <= start_index < N and 0 <= end_index < N and start_index <= end_index
     
    backend_lib.fill_matrix_3d(matrix, vertices, excitation_types, excitation_values, jac_buffer, pos_buffer, N, matrix.shape[0], start_index, end_index)

def plane_intersection(positions, p0, normal):
    assert p0.shape == (3,)
    assert normal.shape == (3,)
    assert positions.shape == (len(positions), 6)
    
    result = np.zeros(6)
    found = backend_lib.plane_intersection(p0, normal, positions, len(positions), result)
    
    return result if found else None

def line_intersection(positions, p0, tangent):
    assert p0.shape == (2,)
    assert tangent.shape == (2,)
    assert positions.shape == (len(positions), 4)
    
    result = np.zeros(4)
    found = backend_lib.line_intersection(p0, tangent, positions, len(positions), result)
     
    return result if found else None





    

    


    

