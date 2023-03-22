"""The backend module is a small wrapper around code implemented in C. All performance critical code is 
implemented in the C backend. As a user you should not use these functions directly yourself."""

import importlib
import ctypes as C
import os.path as path
import platform

from numpy.ctypeslib import ndpointer, as_array
import numpy as np

DEBUG = False

## Attempt 1: load local
if platform.system() in ['Linux', 'Darwin']:
    local_path = path.join(path.dirname(__file__), 'traceon_backend.so')
else:
    local_path = path.join(path.dirname(__file__), 'traceon_backend.dll')

if path.isfile(local_path):
    backend_lib = C.CDLL(local_path)
else:
    ## Attempt 2: load from pip installed path
    global_path = importlib.util.find_spec('traceon.backend.traceon_backend')
    
    if global_path is None:
        help_txt = '''
        Cannot find Traceon backend (C compiled dynamic library).
        It should have been compiled automatically when installing this package using 'pip3 install traceon'.
        If you're running this package locally (i.e. git clone) you have to build this dynamic library yourself.
        On Linux you can use:
            gcc ./traceon/backend/traceon-backend.c -o ./traceon/backend/traceon-backend.so -lm -shared -fPIC -O3 -ffast-math -std=c99 -march=native'''

        raise RuntimeError(help_txt)

    global_path = global_path.origin
    backend_lib = C.cdll.LoadLibrary(global_path)


TRACING_BLOCK_SIZE = C.c_size_t.in_dll(backend_lib, 'TRACING_BLOCK_SIZE').value

DERIV_2D_MAX = C.c_int.in_dll(backend_lib, 'DERIV_2D_MAX_SYM').value

NU_MAX = C.c_int.in_dll(backend_lib, 'NU_MAX_SYM').value
M_MAX = C.c_int.in_dll(backend_lib, 'M_MAX_SYM').value

# Pass numpy array to C
def arr(*args, dtype=np.float64, **kwargs):
    return ndpointer(*args, dtype=dtype, flags=('C_CONTIGUOUS', 'ALIGNED'), **kwargs);

# Pass one dimensional Numpy array to C
v2 = arr(shape=(2,))
v3 = arr(shape=(3,))

dbl = C.c_double
vp = C.c_void_p
sz = C.c_size_t

integration_cb_2d = C.CFUNCTYPE(dbl, dbl, dbl, dbl, dbl, vp)
integration_cb_3d = C.CFUNCTYPE(dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp)
field_fun = C.CFUNCTYPE(None, C.POINTER(dbl), C.POINTER(dbl), vp);

vertices = arr(ndim=3)
lines = arr(ndim=3)
charges = arr(ndim=1)
z_values = arr(ndim=1)

bounds = arr(shape=(3, 2))

times_block = arr(shape=(TRACING_BLOCK_SIZE,))
tracing_block = arr(shape=(TRACING_BLOCK_SIZE, 6))

backend_functions = {
    'ellipk' : (dbl, dbl),
    'ellipe': (dbl, dbl),
    'normal_2d': (None, v2, v2, v2),
    'line_integral': (dbl, v2, v2, v2, integration_cb_2d, C.c_void_p),
    'normal_3d': (None, v3, v3, v3),
    'triangle_integral': (dbl, v3, v3, v3, v3, integration_cb_3d, C.c_void_p),
    'trace_particle': (sz, times_block, tracing_block, field_fun, bounds, dbl, vp),
    'potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'dr1_potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'dz1_potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'axial_derivatives_radial_ring': (None, arr(ndim=2), lines, charges, sz, z_values, sz),
    'potential_radial': (dbl, v3, vertices, charges, sz),
    'potential_radial_derivs': (dbl, v2, z_values, arr(ndim=3), sz),
    'field_radial': (None, v3, v3, vertices, charges, sz),
    'trace_particle_radial': (sz, times_block, tracing_block, bounds, dbl, vertices, charges, sz),
    'field_radial_derivs': (None, v3, v3, z_values, arr(ndim=3), sz),
    'trace_particle_radial_derivs': (sz, times_block, tracing_block, bounds, dbl, z_values, arr(ndim=3), sz),
    'dx1_potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'dy1_potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'dz1_potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'axial_coefficients_3d': (None, vertices, charges, sz, z_values, arr(ndim=4), sz, arr(ndim=1), arr(ndim=4), sz),
    'potential_3d': (dbl, v3, vertices, charges, sz),
    'potential_3d_derivs': (dbl, v3, z_values, arr(ndim=5), sz),
    'field_3d': (None, v3, v3, vertices, charges, sz),
    'trace_particle_3d': (sz, times_block, tracing_block, bounds, dbl, vertices, charges, sz),
    'field_3d_derivs': (None, v3, v3, z_values, arr(ndim=5), sz),
    'trace_particle_3d_derivs': (sz, times_block, tracing_block, bounds, dbl, z_values, arr(ndim=5), sz),
    'fill_matrix_radial': (None, arr(ndim=2), lines, arr(dtype=C.c_uint8, ndim=1), arr(ndim=1), sz, sz, C.c_int, C.c_int),
    'fill_matrix_3d': (None, arr(ndim=2), vertices, arr(dtype=C.c_uint8, ndim=1), arr(ndim=1), sz, sz, C.c_int, C.c_int),
    'xy_plane_intersection_2d': (C.c_bool, arr(ndim=2), sz, arr(shape=(4,)), dbl),
    'xy_plane_intersection_3d': (C.c_bool, arr(ndim=2), sz, arr(shape=(6,)), dbl)
}


for (fun, (res, *args)) in backend_functions.items():
    libfun = getattr(backend_lib, fun)

    def backend_check_numpy_requirements_wrapper(*args, _cfun_reference=libfun, _cfun_name=fun):
        new_args = []
        
        for a in args:
            if isinstance(a, np.ndarray):
                new_arr = np.require(a, requirements=('C_CONTIGUOUS', 'ALIGNED'));
                
                if DEBUG and not (new_arr is a):
                    print('WARNING: copied Numpy array while making call to C backend function: ' + _cfun_name)
                new_args.append(new_arr)
            else:
                new_args.append(a)
         
        assert len(args) == len(new_args)
        return _cfun_reference(*new_args)
      
    setattr(backend_lib, fun, backend_check_numpy_requirements_wrapper)
     
    libfun.restype = res
    libfun.argtypes = args

ellipk = np.frompyfunc(backend_lib.ellipk, 1, 1)
ellipe = np.frompyfunc(backend_lib.ellipe, 1, 1)

def normal_2d(p1, p2):
    normal = np.zeros( (2,) )
    backend_lib.normal_2d(p1, p2, normal)
    return normal

# Remove the last argument, which is usually a void pointer to optional data
# passed to the function. In Python we don't need this functionality
# as we can simply use closures.
def remove_arg(fun):
    return lambda *args: fun(*args[:-1])

def line_integral(point, v1, v2, callback):
    assert point.shape == (2,) and v1.shape == (2,) and v2.shape == (2,)
    return backend_lib.line_integral(point, v1, v2, integration_cb_2d(remove_arg(callback)), None)

def normal_3d(p1, p2, p3):
    normal = np.zeros( (3,) )
    backend_lib.normal_2d(p1, p2, normal)
    return normal
   
def triangle_integral(point, v1, v2, v3, callback):
    assert point.shape == (3,) and v1.shape == (3,) and v2.shape == (3,) and v3.shape == (3,)
    return backend_lib.triangle_integral(point, v1, v2, v3, integration_cb_3d(remove_arg(callback)), None)

def _vec_2d_to_3d(vec):
    assert vec.shape == (2,) or vec.shape == (3,)
     
    if vec.shape == (2,):
        return np.array([vec[0], vec[1], 0.0])
    
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

def trace_particle(position, velocity, field, bounds, atol):
    bounds = np.array(bounds)
    
    return trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle(T, P, wrap_field_fun(field), bounds, atol, None))

def trace_particle_radial(position, velocity, bounds, atol, vertices, charges):
    assert vertices.shape == (len(charges), 2, 3)
    bounds = np.array(bounds)
    
    if bounds.shape[0] == 2:
        bounds = np.array([bounds[0], bounds[1], [-1.0, 0.0]])
     
    times, positions = trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle_radial(T, P, bounds, atol, vertices, charges, len(charges)))
    
    return times, positions[:, [0,1,3,4]]

def trace_particle_radial_derivs(position, velocity, bounds, atol, z, coeffs):
    assert coeffs.shape == (len(z)-1, DERIV_2D_MAX, 6)
    bounds = np.array(bounds)

    if bounds.shape[0] == 2:
        bounds = np.array([bounds[0], bounds[1], [-1.0, 0.0]])
    
    times, positions = trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle_radial_derivs(T, P, bounds, atol, z, coeffs, len(z)))
    
    return times, positions[:, [0,1,3,4]]

def trace_particle_3d(position, velocity, bounds, atol, vertices, charges):
    assert position.shape == (3,)
    assert velocity.shape == (3,)
    assert vertices.shape == (len(charges), 3, 3)
    bounds = np.array(bounds)
    
    return trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle_3d(T, P, bounds, atol, vertices, charges, len(charges)))

def trace_particle_3d_derivs(position, velocity, bounds, atol, z, coeffs):
    assert position.shape == (3,)
    assert velocity.shape == (3,)
    assert coeffs.shape == (len(z)-1, 2, NU_MAX, M_MAX, 4)
    bounds = np.array(bounds)
     
    return trace_particle_wrapper(position, velocity,
        lambda T, P: backend_lib.trace_particle_3d_derivs(T, P, bounds, atol, z, coeffs, len(z)))

potential_radial_ring = remove_arg(backend_lib.potential_radial_ring)
dr1_potential_radial_ring = remove_arg(backend_lib.dr1_potential_radial_ring)
dz1_potential_radial_ring = remove_arg(backend_lib.dz1_potential_radial_ring)

def axial_derivatives_radial_ring(z, lines, charges):
    derivs = np.zeros( (z.size, DERIV_2D_MAX) )
    assert lines.shape[1] == 2 and lines.shape[2] == 3
    assert len(lines) == len(charges)
    
    backend_lib.axial_derivatives_radial_ring(derivs, lines, charges, len(lines), z, len(z))
    return derivs

def potential_radial(point, vertices, charges):
    point = _vec_2d_to_3d(point)
    assert vertices.shape == (len(charges), 2, 3)
    return backend_lib.potential_radial(point, vertices, charges, len(charges))

def potential_radial_derivs(point, z, coeffs):
    assert coeffs.shape == (len(z)-1, DERIV_2D_MAX, 6)
    return backend_lib.potential_radial_derivs(point, z, coeffs, len(z))

def field_radial(point, vertices, charges):
    point = _vec_2d_to_3d(point)
    assert vertices.shape == (len(charges), 2, 3)
     
    field = np.zeros( (3,) )
    backend_lib.field_radial(point, field, vertices, charges, len(charges))
    return field[:2]

def field_radial_derivs(point, z, coeffs):
    point = _vec_2d_to_3d(point)
    assert coeffs.shape == (len(z), DERIV_2D_MAX, 6)
    field = np.zeros( (3,) )
    backend_lib.field_radial_derivs(point, field, z, coeffs, len(z))
    return field[:2]

dx1_potential_3d_point = remove_arg(backend_lib.dx1_potential_3d_point)
dy1_potential_3d_point = remove_arg(backend_lib.dy1_potential_3d_point)
dz1_potential_3d_point = remove_arg(backend_lib.dz1_potential_3d_point)
potential_3d_point = remove_arg(backend_lib.potential_3d_point)

def axial_coefficients_3d(vertices, charges, z, thetas, theta_interpolation):
    assert vertices.shape == (len(charges), 3, 3)
    assert theta_interpolation.shape == (len(thetas)-1, NU_MAX, M_MAX, 4)

    output_coeffs = np.zeros( (len(z), 2, NU_MAX, M_MAX) )
    
    backend_lib.axial_coefficients_3d(vertices, charges, len(vertices),
        z, output_coeffs, len(z),
        thetas, theta_interpolation, len(thetas))
    
    return output_coeffs

def potential_3d(point, vertices, charges):
    assert vertices.shape == (len(charges), 3, 3)
    assert point.shape == (3,)
     
    return backend_lib.potential_3d(point, vertices, charges, len(charges))

def potential_3d_derivs(point, z, coeffs):
    assert coeffs.shape == (len(z), NU_MAX, M_MAX, 4)
    assert point.shape == (3,)
    
    return backend_lib.potential_3d_derivs(point, z, coeffs, len(z))

def field_3d(point, vertices, charges):
    assert vertices.shape == (len(charges), 3, 3)
    assert point.shape == (3,)

    field = np.zeros( (3,) )
    backend_lib.field_3d(point, field, vertices, charges, len(vertices))
    return field

def field_3d_derivs(point, z, coeffs):
    assert point.shape == (3,)
    assert coeffs.shape == (len(z), NU_MAX, M_MAX, 4)

    field = np.zeros( (3,) )
    backend_lib.field_3d_derivs(point, field, z, coeffs, len(z))

def fill_matrix_radial(matrix, lines, excitation_types, excitation_values, start_index, end_index):
    N = len(lines)
    # Due to floating conductor constraints the matrix might actually be bigger than NxN
    assert matrix.shape[0] >= N and matrix.shape[1] >= N and matrix.shape[0] == matrix.shape[1]
    assert lines.shape == (N, 2, 3)
    assert excitation_types.shape == (N,)
    assert excitation_values.shape == (N,)
    assert 0 <= start_index < N and 0 <= end_index < N and start_index < end_index
     
    backend_lib.fill_matrix_radial(matrix, lines, excitation_types, excitation_values, N, matrix.shape[0], start_index, end_index)

def fill_matrix_3d(matrix, vertices, excitation_types, excitation_values, start_index, end_index):
    N = len(vertices)
    # Due to floating conductor constraints the matrix might actually be bigger than NxN
    assert matrix.shape[0] >= N and matrix.shape[1] >= N and matrix.shape[0] == matrix.shape[1]
    assert vertices.shape == (N, 3, 3)
    assert excitation_types.shape == (N,)
    assert excitation_values.shape == (N,)
    assert 0 <= start_index < N and 0 <= end_index < N and start_index < end_index
    
    backend_lib.fill_matrix_3d(matrix, vertices, excitation_types, excitation_values, N, matrix.shape[0], start_index, end_index)

def xy_plane_intersection(positions, z):
    
    assert positions.shape[1] == 4 or positions.shape[1] == 6
    
    positions = np.require(positions, dtype=np.float64, requirements=('C_CONTIGUOUS', 'ALIGNED'))
     
    if positions.shape[1] == 4:
        result = np.zeros( (4,) )
        found = backend_lib.xy_plane_intersection_2d(positions, len(positions), result, z)
        
        return result if found else None

    if positions.shape[1] == 6:
        result = np.zeros( (6,) )
        found = backend_lib.xy_plane_intersection_3d(positions, len(positions), result, z)
        
        return result if found else None


    










    

    


    

