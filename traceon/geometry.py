import numpy as np
from pygmsh import *

import pickle


def revolve_around_optical_axis(geom, elements, factor=1.0):
    revolved = []
    
    for e in elements:
        
        top = e
        for i in range(4):
            top, extruded, lateral = geom.revolve(top, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], factor*0.5*np.pi)
            revolved.append(extruded)
        
    return revolved


class Geometry:
    """Class containing a mesh and related metadata.
    
    Attributes:
        mesh: [meshio](https://github.com/nschloe/meshio) object containing the mesh
        metadata: dictionary containing arbitrary metadata
    """
    
    def __init__(self, mesh, N, bounds, metadata={}, symmetry='radial'):
        """ Args: """
        self.mesh = mesh
        self.metadata = metadata
        self.N = N
        self._symmetry = symmetry

        assert (symmetry == '3d' and len(bounds) == 3) or len(bounds) == 2
         
        self.bounds = bounds
     
    def get_z_bounds(self):
        if self.symmetry == '3d':
            return self.bounds[2]
        else:
            return self.bounds[1]
          
    def write(self, filename):
        """Write a mesh to a file. The pickle module will be used
        to save the Geometry object.

        Args:
            filename: name of the file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def get_mesh_size(self):
        line_points = self.mesh.points[ self.mesh.cells_dict['line'] ]
        return np.mean(np.linalg.norm(line_points[:, 1] - line_points[:, 0], axis=1))
    
    @property
    def symmetry(self):
        if hasattr(self, '_symmetry'):
            return self._symmetry
        else:
            return 'radial'

    def get_electrodes(self):
        """Get the names of all the electrodes in the geometry.

        Returns:
            List of electrode names"""
        return list(self.mesh.cell_sets_dict.keys())
    
    def read(filename):
        """Read a geometry from disk (previously saved with the write method)
        
        Args:
            filename: the name of the file.
        """
        with open(filename, 'rb') as f:
            object_ = pickle.load(f)
        
        if isinstance(object_, dict): 
            # Backwards compatibility
            return Geometry(object_['mesh'], object_['N'], metadata=object_['metadata'])
        else:
            return object_
    
    def __str__(self):
        return str(self.mesh) + ' (metadata: ' + str(self.metadata) + ')'

class MEMSStack(occ.Geometry):
    
    def __init__(self, z0=0.0, mesh_size=1/10, rmax=2, enclose_right=True, margin_right=0.5):
        super().__init__()
        self.z0 = z0
        self.mesh_size = mesh_size
        self.rmax = rmax
        self.margin_right = margin_right
        self.enclose_right = enclose_right
        
        self._current_z = z0
        self._last_name = None
        self._physical_queue = dict()
     
    def add_spacer(self, thickness):
        self._current_z += thickness

    def _add_physical(self, name, entities):
        if name in self._physical_queue:
            self._physical_queue[name].extend(entities)
        else:
            self._physical_queue[name] = entities
    
    def add_electrode(self, radius, thickness, name):
        x0 = [radius, self._current_z]
         
        points = [x0, [x0[0], x0[1]+thickness], [self.rmax, x0[1]+thickness], [self.rmax, x0[1]]]
        points = [self.add_point(p, self.mesh_size) for p in points]
        
        lines = [self.add_line(points[i], points[j]) for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0])]
        cl = self.add_curve_loop(lines)
         
        self._add_physical(name, lines)
        self._last_name = name
         
        self._current_z += thickness

        return cl
    
    def generate_mesh(self, *args, **kwargs):
        # Enclose on right

        points = [[self.rmax + self.margin_right, self.z0], [self.rmax + self.margin_right, self._current_z]]
        line = self.add_line(self.add_point(points[0], self.mesh_size), self.add_point(points[1], self.mesh_size))
         
        self._add_physical(self._last_name, [line])

        for label, entities in self._physical_queue.items():
            self.add_physical(entities, label)
        
        return super().generate_mesh(*args, **kwargs)

        
def create_two_cylinder_lens(N=200, S=0.2, R=1, wall_thickness=1, boundary_length=20, gap_at_zero=False, include_boundary=True, **kwargs):
    """Generate lens consisting of two concentric cylinders. For example studied in
    David Edwards, Jr. Accurate Potential Calculations For The Two Tube Electrostatic Lens Using FDM A Multiregion Method.  2007.
    
    D. Cubric, B. Lencova, F.H. Read, J. Zlamal. Comparison of FDM, FEM and BEM for electrostatic charged particle optics. 1999.
     
    Args:
        S: spacing between the cylinders.
        R: radius of the cylinders.
        wall_thickness: thickness of the cylinders. 0.0 is a valid input.
        boundary_length: length of the entire lens.
    """
 
    with occ.Geometry() as geom:
        cylinder_length = (boundary_length - S)/2
        assert boundary_length == 2*cylinder_length + S

        assert wall_thickness == 0.0 or wall_thickness > 0.1
         
        if wall_thickness != 0.0:
            points = [
                [0, 0],
                [R, 0],
                [R, cylinder_length],
                [R + wall_thickness, cylinder_length],
                [R + wall_thickness, cylinder_length + S],
                [R, cylinder_length + S],
                [R, boundary_length],
                [0, boundary_length]
            ]
            if include_boundary:
                physicals = [('v1', [0, 1, 2]), ('v2', [4, 5, 6]), ('gap', [3])]
            else:
                physicals = [('v1', [1, 2]), ('v2', [4, 5]), ('gap', [3])]
        else:
            points = [ [0, 0],
                [R, 0],
                [R, cylinder_length],
                [R, cylinder_length + S],
                [R, boundary_length],
                [0, boundary_length]]
            if include_boundary:
                physicals = [('v1', [0, 1]), ('v2', [3, 4]), ('gap', [2])]
            else:
                physicals = [('v1', [1]), ('v2', [3]), ('gap', [2])]
         
        lcar = boundary_length/N
        poly = geom.add_polygon(points, lcar)
         
        for key, indices in physicals:
            lines = [poly.curves[idx] for idx in indices]
            geom.add_physical(lines, key)

        mesh = geom.generate_mesh(dim=1)
        
        if gap_at_zero:
            mesh.points[:, 1] -= cylinder_length + S/2
         
        return Geometry(mesh, N, **kwargs)



