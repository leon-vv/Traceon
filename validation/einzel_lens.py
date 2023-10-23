import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.plotting as P
import traceon.tracing as T

from validation import Validation

THICKNESS = 0.5
SPACING = 0.5
RADIUS = 0.15
z0 = -THICKNESS - SPACING - THICKNESS/2 - 0.5

class EinzelLens(Validation):

    def __init__(self):
        super().__init__('Calculate focal length of an einzel lens.')
        self.plot_colors = dict(lens='blue', ground='green', boundary='purple')

    def create_mesh(self, MSF, symmetry):
        with G.MEMSStack(z0=z0, zmin=-1.5, zmax=1.5, symmetry=symmetry, size_from_distance=True) as geom:
            geom.add_electrode(RADIUS, THICKNESS, 'ground')
            geom.add_spacer(THICKNESS)
            geom.add_electrode(RADIUS, THICKNESS, 'lens')
            geom.add_spacer(THICKNESS)
            geom.add_electrode(RADIUS, THICKNESS, 'ground')
            geom.set_mesh_size_factor(MSF)
            return geom.generate_mesh()

    def get_excitation(self, mesh):
        excitation = E.Excitation(mesh)
        excitation.add_voltage(ground=0.0, lens=1000)
        excitation.add_boundary('boundary')
        return excitation
     
    def correct_value_of_interest(self):
        return 3.915970140918643
      
    def compute_value_of_interest(self, geom, field):
        _3d = geom.is_3d()
        
        field.set_bounds( ((-RADIUS, RADIUS), (-1.5, 1.5)) if not _3d else ((-RADIUS, RADIUS), (-RADIUS, RADIUS), (-1.5,1.5)) )
        field_axial = field.axial_derivative_interpolation(-1.5, 1.5, 600)
        
        z = np.linspace(-1.5, 1.5, 150)
        pot = [field.potential_at_point(np.array([0.0, z_] if not _3d else [0.0, 0.0, z_])) for z_ in z]
        #plt.plot(z, pot)
        #plt.show()
        
        bounds = ((-RADIUS, RADIUS), (-5, 3.5)) if not _3d else ((-RADIUS, RADIUS), (-RADIUS, RADIUS), (-5, 3.5))
        tracer = T.Tracer(field_axial, bounds)
        
        p0 = np.array([RADIUS/3, 3]) if not _3d else np.array([RADIUS/3, 0.0, 3])
        v0 = T.velocity_vec_xz_plane(1000, 0, three_dimensional=_3d)
        
        _, pos = tracer(p0, v0)
        
        return -T.axis_intersection(pos)

if __name__ == '__main__':
    EinzelLens().run_validation()

