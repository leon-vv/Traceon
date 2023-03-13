import numpy as np
from math import sqrt
from pygmsh import *
import gmsh
from enum import Enum

import pickle

from .util import Saveable


def revolve_around_optical_axis(geom, elements, factor=1.0):
    revolved = []
    
    for e in (elements if isinstance(elements, list) else [elements]):
        
        top = e
        for i in range(4):
            top, extruded, lateral = geom.revolve(top, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], factor*0.5*np.pi)
            revolved.append(extruded)
     
    return revolved

class Symmetry(Enum):
    RADIAL = 0
    THREE_D = 1

class Geometry(occ.Geometry):
    def __init__(self, symmetry, size_from_distance=False, zmin=None, zmax=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size_from_distance = size_from_distance
        self.zmin = zmin
        self.zmax = zmax
        self.symmetry = symmetry
        self._physical_queue = dict()
        
    def _add_physical(self, name, entities):
        assert isinstance(entities, list)
        
        if name in self._physical_queue:
            self._physical_queue[name].extend(entities)
        else:
            self._physical_queue[name] = entities

    def generate_mesh(self, *args, **kwargs):
        # Enclose on right
          
        for label, entities in self._physical_queue.items():
            self.add_physical(entities, label)
          
        if self.size_from_distance:
            self.set_mesh_size_callback(self.mesh_size_callback)
        
        dim = 2 if self.symmetry == Symmetry.THREE_D else 1
        
        return Mesh(super().generate_mesh(dim=dim, *args, **kwargs), self.symmetry)

    def set_mesh_size_factor(self, factor):
        if self.symmetry == Symmetry.RADIAL:
            gmsh.option.setNumber('Mesh.MeshSizeFactor', 1/factor)
        elif self.symmetry == Symmetry.THREE_D:
            # GMSH seems to produce meshes which contain way more elements for 3D geometries
            # with the same mesh factor. This is confusing for users and therefore we arbtrarily
            # incrase the mesh size to roughly correspond with the 2D number of elements.
            gmsh.option.setNumber('Mesh.MeshSizeFactor', 4*sqrt(1/factor))

    def set_minimum_mesh_size(self, size):
        gmsh.option.setNumber('Mesh.MeshSizeMin', size)
     
    def mesh_size_callback(self, dim, tag, x, y, z):
        # Scale mesh size with distance to optical axis, but only the part of the optical
        # axis that lies between zmin and zmax
        
        z_optical = z if self.symmetry == Symmetry.THREE_D else y
         
        if self.zmin is not None:
            z_optical = max(z_optical, self.zmin)
        if self.zmax is not None:
            z_optical = min(z_optical, self.zmax)
         
        if self.symmetry == Symmetry.THREE_D:
            return sqrt( x**2 + y**2 + (z-z_optical)**2 )
        else:
            return sqrt( x**2 + (y-z_optical)**2 )



class Mesh:
    """Class containing a mesh and related metadata.
    
    Attributes:
        mesh: [meshio](https://github.com/nschloe/meshio) object containing the mesh
        metadata: dictionary containing arbitrary metadata
    """
    
    def __init__(self, mesh, symmetry, metadata={}):
        assert isinstance(symmetry, Symmetry)
        self.mesh = mesh
        self.metadata = metadata
        self.symmetry = symmetry
    
    def write(self, filename):
        """Write a mesh to a file. The pickle module will be used
        to save the Geometry object.

        Args:
            filename: name of the file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def read(filename):
        """Read a geometry from disk (previously saved with the write method)
        
        Args:
            filename: the name of the file.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
     
    def get_electrodes(self):
        """Get the names of all the electrodes in the geometry.

        Returns:
            List of electrode names"""
        return list(self.mesh.cell_sets_dict.keys())
    
    def __str__(self):
        return str(self.mesh) + ' (metadata: ' + str(self.metadata) + ')'

class MEMSStack(Geometry):
    
    def __init__(self, *args, z0=0.0, revolve_factor=0.0, rmax=2, enclose_right=True, margin_right=0.1, **kwargs):
        self.symmetry = Symmetry.RADIAL if revolve_factor == 0.0 else Symmetry.THREE_D
        super().__init__(self.symmetry, *args, **kwargs)
        
        self.z0 = z0
        self.revolve_factor = revolve_factor
        self._3d = self.revolve_factor != 0.0
        self.rmax = rmax
        self.margin_right = margin_right
        self.enclose_right = enclose_right
        
        self._current_z = z0
        self._last_name = None
     
    def add_spacer(self, thickness):
        self._current_z += thickness
    
    def add_electrode(self, radius, thickness, name):
        x0 = [radius, self._current_z]
          
        points = [x0, [x0[0], x0[1]+thickness], [self.rmax, x0[1]+thickness], [self.rmax, x0[1]]]
        
        if self._3d:
            points = [self.add_point([p[0], 0.0, p[1]]) for p in points]
        else:
            points = [self.add_point(p) for p in points]
        
        lines = [self.add_line(points[i], points[j]) for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0])]
        cl = self.add_curve_loop(lines)
        
        if self._3d:
            revolved = revolve_around_optical_axis(self, lines, self.revolve_factor)
            self._add_physical(name, revolved)
        else:
            self._add_physical(name, lines)
        
        self._last_name = name
         
        self._current_z += thickness
        
        return cl
    
    def generate_mesh(self, *args, **kwargs):
        # Enclose on right
        
        if self.enclose_right:
            points = [[self.rmax, self.z0], [self.rmax + self.margin_right, self.z0], [self.rmax + self.margin_right, self._current_z], [self.rmax, self._current_z]]
            
            if self._3d:
                points = [[p[0], 0.0, p[1]] for p in points]
                lines = [self.add_line(self.add_point(p1), self.add_point(p2)) for p1, p2 in zip(points[1:], points)]
                revolved = revolve_around_optical_axis(self, lines, self.revolve_factor)
                self._add_physical(self._last_name, revolved)
            else:
                lines = [self.add_line(self.add_point(p1), self.add_point(p2)) for p1, p2 in zip(points[1:], points)]
                self._add_physical(self._last_name, lines)
        
        return super().generate_mesh(*args, **kwargs)        

    
        
def create_two_cylinder_lens(MSF, S=0.2, R=1, wall_thickness=1, boundary_length=20, include_boundary=True):
    """Generate lens consisting of two concentric cylinders. For example studied in
    David Edwards, Jr. Accurate Potential Calculations For The Two Tube Electrostatic Lens Using FDM A Multiregion Method.  2007.
    
    D. Cubric, B. Lencova, F.H. Read, J. Zlamal. Comparison of FDM, FEM and BEM for electrostatic charged particle optics. 1999.
     
    Args:
        S: spacing between the cylinders.
        R: radius of the cylinders.
        wall_thickness: thickness of the cylinders. 0.0 is a valid input.
        boundary_length: length of the entire lens.
    """
 
    with Geometry(Symmetry.RADIAL) as geom:
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
         
        poly = geom.add_polygon(points)
         
        for key, indices in physicals:
            lines = [poly.curves[idx] for idx in indices]
            geom.add_physical(lines, key)

        geom.set_mesh_size_factor(MSF)
        return geom.generate_mesh()



