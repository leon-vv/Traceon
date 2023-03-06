import ctypes as C
import os.path as path

from numpy.ctypeslib import ndpointer
import numpy as np

backend_lib = C.CDLL(path.join(path.dirname(__file__), 'backend.so'))

TRACING_BLOCK_SIZE = C.c_size_t.in_dll(backend_lib, 'TRACING_BLOCK_SIZE').value

DERIV_2D_MAX = C.c_int.in_dll(backend_lib, 'DERIV_2D_MAX').value

DERIV_3D_MAX = C.c_int.in_dll(backend_lib, 'DERIV_2D_MAX').value
NU_MAX = DERIV_3D_MAX//2
M_MAX = DERIV_3D_MAX


# Pass numpy array to C
def arr(*args, dtype=np.float64, **kwargs):
    return ndpointer(*args, dtype=dtype, flags='C_CONTIGUOUS', **kwargs);

# Pass one dimensional Numpy array to C
v2 = arr(shape=(2,))
v3 = arr(shape=(3,))

dbl = C.c_double
vp = C.c_void_p
sz = C.c_size_t

integration_cb_2d = C.CFUNCTYPE(dbl, dbl, dbl, dbl, dbl, vp)
integration_cb_3d = C.CFUNCTYPE(dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp)
field_fun = C.CFUNCTYPE(None, dbl*6, dbl*3, vp);

vertices = arr(ndim=3)
lines = arr(ndim=3)
charges = arr(ndim=1)
z_values = arr(ndim=1)

bounds = (dbl*3)*3

tracing_block = arr(shape=(TRACING_BLOCK_SIZE, 6))

backend_functions = {
    'ellipk' : (dbl, dbl),
    'ellipe': (dbl, dbl),
    'line_integral': (dbl, v2, v2, v2, integration_cb_2d, C.c_void_p),
    'triangle_integral': (dbl, v3, v3, v3, v3, integration_cb_3d, C.c_void_p),
    'trace_particle': (sz, tracing_block, field_fun, bounds, dbl, vp),
    'potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'dr1_potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'dz1_potential_radial_ring': (dbl, dbl, dbl, dbl, dbl, vp), 
    'axial_derivatives_radial_ring': (None, arr(ndim=2), lines, charges, sz, z_values, sz),
    'potential_radial': (dbl, v3, vertices, charges, sz),
    'field_radial': (None, v3, v3, vertices, charges, sz),
    'trace_particle_radial': (sz, tracing_block, bounds, dbl, vertices, charges, sz),
    'field_radial_derivs': (None, v3, v3, z_values, arr(ndim=3), sz),
    'trace_particle_radial_derivs': (sz, tracing_block, bounds, dbl, z_values, arr(ndim=3), sz),
    'dx1_potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'dy1_potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'dz1_potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'potential_3d_point': (dbl, dbl, dbl, dbl, dbl, dbl, dbl, vp),
    'axial_coefficients_3d': (None, vertices, charges, sz, z_values, arr(ndim=4), sz, arr(ndim=1), arr(ndim=4), sz),
    'potential_3d': (dbl, v3, vertices, charges, sz),
    'potential_3d_derivs': (dbl, v3, z_values, arr(ndim=5), sz),
    'field_3d': (None, v3, v3, vertices, charges, sz),
    'trace_particle_3d': (sz, tracing_block, bounds, dbl, vertices, charges, sz),
    'field_3d_derivs': (None, v3, v3, z_values, arr(ndim=5), sz),
    'trace_particle_3d_derivs': (sz, tracing_block, bounds, dbl, z_values, arr(ndim=5), sz),
    'fill_matrix_radial': (None, arr(ndim=2), lines, arr(dtype=int, ndim=1), arr(ndim=1), sz, C.c_int, C.c_int),
    'fill_matrix_3d': (None, arr(ndim=2), vertices, arr(dtype=int, ndim=1), arr(ndim=1), sz, C.c_int, C.c_int)
}


for (fun, (res, *args)) in backend_functions.items():
    libfun = getattr(backend_lib, fun)
    
    libfun.restype = res
    libfun.argtypes = args

ellipk = np.frompyfunc(backend_lib.ellipk, 1, 1)
ellipe = np.frompyfunc(backend_lib.ellipe, 1, 1)

# Remove the last argument, which is usually a void pointer to optional data
# passed to the function. In Python we don't need this functionality
# as we can simply use closures.
def remove_arg(fun):
    return lambda *args: fun(*args[:-1])

def line_integral(point, v1, v2, callback):
    assert point.shape == (2,) and v1.shape == (2,) and v2.shape == (2,)
    return backend_lib.line_integral(point, v1, v2, integration_cb_2d(remove_arg(callback)), None)
    
def triangle_integral(point, v1, v2, v3, callback):
    assert point.shape == (3,) and v1.shape == (3,) and v2.shape == (3,) and v3.shape == (3,)
    return backend_lib.triangle_integral(point, v1, v2, v3, integration_cb_3d(remove_arg(callback)), None)

def trace_particle_wrapper(position, velocity, fill_positions_fun):
    assert position.shape == (3,) and velocity.shape == (3,)
    
    N = TRACING_BLOCK_SIZE
    blocks = []
     
    positions = np.zeros( (TRACING_BLOCK_SIZE, 6) )
    positions[0] = np.concatenate(position, velocity)
    
    while N == TRACING_BLOCK_SIZE:
        N = fill_positions_fun(positions)
         
        # Prevent the starting positions to be both at the end of the previous block and the start
        # of the current block.
        blocks.append(positions[1:N] if len(blocks) > 0  else positions[:N])
        positions = np.zeros( (TRACING_BLOCK_SIZE, 6) )
        positions[0] = blocks[-1][-1]
    
    # Speedup, usually no concatenation needed
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

def trace_particle(position, velocity, field, bounds, atol):
    assert bounds.shape == (3, 2)
    return trace_particle_wrapper(position, velocity,
        lambda P: backend_lib.trace_particle(P, field_fun(remove_arg(field)), bounds, atol, None))

def trace_particle_radial(position, velocity, bounds, atol, vertices, charges):
    assert bounds.shape == (3,2)
    assert vertices.shape == (len(charges), 2, 3)
    
    return trace_particle_wrapper(position, velocity,
        lambda P: backend_lib.trace_particle_radial(P, bounds, atol, vertices, charges, len(charges)))

def trace_particle_radial_derivs(position, velocity, bounds, atol, z, coeffs):
    assert bounds.shape == (3,2)
    assert coeffs.shape == (len(z), DERIV_2D_MAX, 4)
     
    return trace_particle_wrapper(position, velocity,
        lambda P: backend_lib.trace_particle_radial_derivs(P, bounds, atol, z, coeffs, len(z)))

def trace_particle_3d(position, velocity, bounds, atol, vertices, charges):
    assert bounds.shape == (3,2)
    assert vertices.shape == (len(charges), 3, 3)
    
    return trace_particle_wrapper(position, velocity,
        lambda P: backend_lib.trace_particle_3d(P, bounds, atol, vertices, charges, len(charges)))

def trace_particle_3d_derivs(position, velocity, bounds, atol, z, coeffs):
    assert bounds.shape == (3,2)
    assert coeffs.shape == (len(z), NU_MAX, M_MAX, 4)
     
    return trace_particle_wrapper(position, velocity,
        lambda P: backend_lib.trace_particle_3d_derivs(P, bounds, atol, z, coeffs, len(z)))

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
    assert point.shape == (3,)
    assert vertices.shape == (len(charges), 2, 3)
    return backend_lib.potential_radial(point, vertices, charges, len(charges))

def field_radial(point, vertices, charges):
    assert point.shape == (3,)
    assert vertices.shape == (len(charges), 2, 3)
    
    field = np.zeros( (3,) )
    backend_lib.field_radial(point, field, vertices, charges, len(charges))
    return field

def field_radial_derivs(point, z, coeffs):
    assert coeffs.shape == (len(z), DERIV_2D_MAX, 4)
    field = np.zeros( (3,) )
    backend_lib.field_radial_derivs(point, field, z, coeffs, len(z))
    return field

dx1_potential_3d_point = remove_arg(backend_lib.dx1_potential_3d_point)
dy1_potential_3d_point = remove_arg(backend_lib.dy1_potential_3d_point)
dz1_potential_3d_point = remove_arg(backend_lib.dz1_potential_3d_point)
potential_3d_point = remove_arg(backend_lib.potential_3d_point)

def axial_coefficients_3d(vertices, charges, z, thetas, theta_interpolation):
    assert vertices.shape == (len(charges), 3, 3)
    assert theta_interpolation.shape == (len(thetas), NU_MAX, M_MAX, 4)

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
    backend_lib.field_3d(point, field, vertices, charges, len(z))
    return field

def field_3d_derivs(point, z, coeffs):
    assert point.shape == (3,)
    assert coeffs.shape == (len(z), NU_MAX, M_MAX, 4)

    field = np.zeros( (3,) )
    backend_lib.field_3d_derivs(point, field, z, coeffs, len(z))

def fill_matrix_radial(matrix, lines, excitation_types, excitation_values, start_index, end_index):
    N = len(lines)
    assert matrix.shape == (N, N)
    assert lines.shape == (N, 2, 3)
    assert excitation_types.shape == (N,)
    assert excitation_values.shape == (N,)
    assert 0 <= start_index < N and 0 <= end_index < N and start_index < end_index
    
    backend_lib.fill_matrix(matrix, lines, excitation_types, excitation_values, N, start_index, end_index)

def fill_matrix_3d(matrix, vertices, excitation_types, excitation_values, start_index, end_index):
    N = len(vertices)
    assert matrix.shape == (N, N)
    assert vertices.shape == (N, 3, 3)
    assert excitation_types.shape == (N,)
    assert excitation_values.shape == (N,)
    assert 0 <= start_index < N and 0 <= end_index < N and start_index < end_index
    
    backend_lib.fill_matrix(matrix, vertices, excitation_types, excitation_values, N, start_index, end_index)












    

    


    

