import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.plotting as P
import traceon.tracing as T

import util

THICKNESS = 0.5
SPACING = 0.5
RADIUS = 0.15
z0 = -THICKNESS - SPACING - THICKNESS/2 - 0.5

def create_geometry(MSF, symmetry):
    revolve_factor = 0.0 if symmetry == G.Symmetry.RADIAL else 1.0
    with G.MEMSStack(z0=z0, zmin=-1.5, zmax=1.5, revolve_factor=revolve_factor, size_from_distance=True) as geom:
        geom.add_electrode(RADIUS, THICKNESS, 'ground')
        geom.add_spacer(THICKNESS)
        geom.add_electrode(RADIUS, THICKNESS, 'lens')
        geom.add_spacer(THICKNESS)
        geom.add_electrode(RADIUS, THICKNESS, 'ground')
        geom.set_mesh_size_factor(MSF)
        return geom.generate_mesh()

def compute_field(mesh):
    excitation = E.Excitation(mesh)
    excitation.add_voltage(ground=0.0, lens=1000)
    excitation.add_boundary('boundary')
    field = S.solve_bem(excitation)
    return excitation, field

def compute_error(exc, field, geom):
    _3d = geom.symmetry == G.Symmetry.THREE_D
    
    field.set_bounds( ((-RADIUS, RADIUS), (-1.5, 1.5)) if not _3d else ((-RADIUS, RADIUS), (-RADIUS, RADIUS), (-1.5,1.5)) )
    field_axial = field.axial_derivative_interpolation(-1.5, 1.5, 400)
    
    z = np.linspace(-1.5, 1.5, 150)
    pot = [field.potential_at_point(np.array([0.0, z_] if not _3d else [0.0, 0.0, z_])) for z_ in z]
    #plt.plot(z, pot)
    #plt.show()
     
    bounds = ((-RADIUS, RADIUS), (-5, 3.5)) if not _3d else ((-RADIUS, RADIUS), (-RADIUS, RADIUS), (-5, 3.5))
    tracer = T.Tracer(field_axial, bounds)
    
    p0 = np.array([RADIUS/3, 3]) if not _3d else np.array([RADIUS/3, 0.0, 3])
    v0 = T.velocity_vec_xz_plane(1000, 0, three_dimensional=_3d)
     
    _, pos = tracer(p0, v0)
    
    f = -T.yz_plane_intersection(pos)
    correct = 3.916282058300268
    print('Focal length calculated:\t', f)
    print('Actual focal length:\t\t', correct)

    
    return exc, f/correct - 1

util.parser.description = '''   '''
util.parse_validation_args(create_geometry, compute_field, compute_error, lens='blue', ground='green', boundary='purple')
#    MSF={'radial': [200, 300, 400, 500], '3d': [100, 175, 250, 350, 500]})


