"""The geometry module allows the creation of general geometries in 2D and 3D and generate
the resulting meshes. The heavy lifting is done by the powerful [GMSH](https://gmsh.info/) library, and we access this library
through the convenient [pygmsh](https://github.com/nschloe/pygmsh) library.

The GMSH library has the concept of _physical groups_. These are simply elements inside your geometry which
are assigned a given name. When using Traceon, usually every electrode gets assigned its
own name (or physical group) with which it can be referenced when later specifying the excitation
of this electrode.

From this module you will likely use either the `Geometry` class when creating arbitrary geometries,
or the `MEMSStack` class, if your geometry consists of a stack of MEMS fabricated elements.
"""
from math import sqrt
from enum import Enum
import pickle

import numpy as np
from pygmsh import *
import gmsh
import meshio

from .util import Saveable
from .backend import N_QUAD_2D, position_and_jacobian_radial, position_and_jacobian_3d
from . import logging

def revolve_around_optical_axis(geom, elements, factor=1.0):
    """
    Revolve geometry elements around the optical axis. Useful when you
    want to generate 3D geometries from a cylindrically symmetric 2D geometry.
    
    Parameters
    ----------
    geom : Geometry
         
    elements : list of GMSH elements
        The geometry elements to revolve. These should have been returned previously from for example
        a call to geom.add_line(...).
        
    factor : float
         How far the elements should be revolved around the optical axis. factor=1.0 corresponds
         to a full revolution ( \(2\pi \) radians) around the optical axis, while for example 0.5
         corresponds to a revolution of only \(\pi\) radians. (Default value = 1.0).

    Returns
    -------
    A list of surface elements representing the revolution around the optical axis.
    """
    revolved = []
    
    for e in (elements if isinstance(elements, list) else [elements]):
        
        top = e
        for i in range(4):
            top, extruded, lateral = geom.revolve(top, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], factor*0.5*np.pi)
            revolved.append(extruded)
     
    return revolved

class Symmetry(Enum):
    """
    Symmetry of the geometry. Used when deciding which formulas to use in the Boundary Element Method. The currently
    supported symmetries are radial symmetry (also called cylindrical symmetry) and general 3D geometries.
    """
    RADIAL = 0
    THREE_D = 2

    def __str__(self):
        if self == Symmetry.RADIAL:
            return 'radial'
        elif self == Symmetry.THREE_D:
            return '3d' 
    
    def is_2d(self):
        return self == Symmetry.RADIAL
        
    def is_3d(self):
        return self == Symmetry.THREE_D

class Geometry(geo.Geometry):
    """
    Small wrapper class around pygmsh.geo.Geometry which itself is a small wrapper around the powerful GMSH library.
    See the GMSH and pygmsh documentation to learn how to build any 2D or 3D geometry. This class makes it easier to control
    the mesh size (using the _mesh size factor_) and optionally allows to scale the mesh size with the distance from the optical
    axis. It also add support for multiple calls to the `add_physical` method with the same name.
    
    Parameters
    ---------
    symmetry: Symmetry

    size_from_distance: bool, optional
        Scale the mesh size with the distance from the optical axis (z-axis).

    zmin: float, optional
    zmax: float, optional
        When `size_from_distance=True` geometric elements that touch the optical axis (as in electrostatic mirrors)
        will imply a zero mesh size and therefore cause singularities. The zmin and zmax arguments
        allow to specify which section of the optical axis will be reachable by the electrons. When
        calculating the mesh size the distance from the closest point on this section of the optical axis is used.
        This prevents singularities.
    """
    def __init__(self, symmetry, size_from_distance=False, zmin=None, zmax=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size_from_distance = size_from_distance
        self.zmin = zmin
        self.zmax = zmax
        self.MSF = None
        self.symmetry = symmetry
        self._physical_queue = dict()

    def __str__(self):
        if self.zmin is not None and self.zmax is not None:
            return f'<Traceon Geometry {self.symmetry}, zmin={self.zmin:.2f} mm, zmax={self.zmax:.2f} mm'
        else:
            return f'<Traceon Geometry {self.symmetry}>'

    def is_3d(self):
        """Check if the geometry is three dimensional.

        Returns
        ----------------
        True if geometry is three dimensional, False if the geometry is two dimensional"""
        return self.symmetry.is_3d()

    def is_2d(self):
        """Check if the geometry is two dimensional.

        Returns
        ----------------
        True if geometry is two dimensional, False if the geometry is three dimensional"""

        return self.symmetry.is_2d()
     
    def add_physical(self, entities, name):
        """

        Parameters
        ----------
        entities : list of GMSH elements or GMSH element
            Geometric entities to assign the given name (the given _physical group_ in GMSH terminology).
        name : string
            Name of the physical group.
        """
        if not isinstance(entities, list):
            entities = [entities]
        
        if name in self._physical_queue:
            self._physical_queue[name].extend(entities)
        else:
            self._physical_queue[name] = entities

    def _generate_mesh(self, dimension, higher_order=False, *args, **kwargs):
        assert dimension == 1 or dimension == 2, "Currently only line and triangle meshes supported (dimension 1 or 2)"
        
        for label, entities in self._physical_queue.items():
            super().add_physical(entities, label)
          
        if self.size_from_distance:
            self.set_mesh_size_callback(self._mesh_size_callback)
        
        assert not higher_order or self.is_2d(), "Higher order is not supported in 3d"
        
        if dimension == 1 and higher_order:
            gmsh.option.setNumber('Mesh.ElementOrder', 3)
        elif dimension == 1 and not higher_order:
            gmsh.option.setNumber('Mesh.ElementOrder', 1)
        elif dimension == 2:
            gmsh.option.setNumber('Mesh.ElementOrder', 1)
        
        return Mesh.from_meshio(super().generate_mesh(dim=dimension, *args, **kwargs), self.symmetry)
    
    def generate_line_mesh(self, higher_order, *args, **kwargs):
        """Generate boundary mesh in 2D, by splitting the boundary in line elements.

        Parameters
        -----------------
        higher_order: bool
            Whether to use higher order (curved) line elements.

        Returns
        ----------------
        `Mesh`
        """
        if self.MSF is not None:
            gmsh.option.setNumber('Mesh.MeshSizeFactor', 1/self.MSF)
        return self._generate_mesh(*args, higher_order=higher_order, dimension=1, **kwargs)
    
    def generate_triangle_mesh(self, higher_order=False, *args, **kwargs):
        """Generate triangle mesh. Note that also 2D meshes can have triangles, which can current coils.
        
        Parameters
        -----------------
        higher_order: bool
            Outdated. Do not use.
        
        Returns
        ----------------
        `Mesh`
        """
        assert higher_order is False, "Higher order meshes are not supported in 3D"
         
        if self.MSF is not None:
            # GMSH seems to produce meshes which contain way more elements for 3D geometries
            # with the same mesh factor. This is confusing for users and therefore we arbtrarily
            # increase the mesh size to roughly correspond with the 2D number of elements.
            gmsh.option.setNumber('Mesh.MeshSizeFactor', 4*sqrt(1/self.MSF))
        return self._generate_mesh(*args, higher_order=higher_order, dimension=2, **kwargs)
    
    def set_mesh_size_factor(self, factor):
        """
        Set the mesh size factor. Which simply scales with the total number of elements in the mesh.
        
        Parameters
        ----------
        factor : float
            The mesh size factor to use. 
        
        """
        self.MSF = factor
     
    def set_minimum_mesh_size(self, size):
        """
        Set the minimum mesh size possible. Especially useful when geometric elements touch
        the optical axis and cause singularities when used with `size_from_distance=True`.
        
        Parameters
        ----------
        size : float
            The minimum mesh size.  

        """
        gmsh.option.setNumber('Mesh.MeshSizeMin', size)
     
    def _mesh_size_callback(self, dim, tag, x, y, z, _):
        # Scale mesh size with distance to optical axis, but only the part of the optical
        # axis that lies between zmin and zmax
        
        z_optical = y if self.symmetry == Symmetry.RADIAL else z
         
        if self.zmin is not None:
            z_optical = max(z_optical, self.zmin)
        if self.zmax is not None:
            z_optical = min(z_optical, self.zmax)
         
        if self.symmetry == Symmetry.RADIAL:
            return sqrt( x**2 + (y-z_optical)**2 )
        else:
            return sqrt( x**2 + y**2 + (z-z_optical)**2 )

def _concat_arrays(arr1, arr2):
    if not len(arr1):
        return np.copy(arr2)
    if not len(arr2):
        return np.copy(arr1)
      
    assert arr1.shape[1:] == arr2.shape[1:], "Cannot add meshes if one is higher order and the other is not"
    
    return np.concatenate( (arr1, arr2), axis=0)

class Mesh(Saveable):
    """Class containing a mesh.
    For now, to make things manageable only lines and triangles are supported.
    Lines and triangles can be higher order (curved) or not. But a mesh cannot contain
    both curved and simple elements at the same time.
    
    When lines are higher order they consist of four points each (corresponding to GMSH line4 type).
    Higher order triangles are not supported."""
     
    def __init__(self, symmetry,
            points=[],
            lines=[],
            triangles=[],
            physical_to_lines={},
            physical_to_triangles={}):
        
        assert isinstance(symmetry, Symmetry)
        self.symmetry = symmetry
        
        # Ensure the correct shape even if empty arrays
        if len(points):
            self.points = np.array(points, dtype=np.float64)
        else:
            self.points = np.empty((0,3), dtype=np.float64)
         
        if len(lines):
            self.lines = np.array(lines)
        else:
            self.lines = np.empty((0,2), dtype=np.uint64)
    
        if len(triangles):
            self.triangles = np.array(triangles)
        else:
            self.triangles = np.empty((0, 3), dtype=np.uint64)
         
        self.physical_to_lines = physical_to_lines.copy()
        self.physical_to_triangles = physical_to_triangles.copy()
        
        if symmetry.is_2d():
            assert points.shape[1] == 2 or np.allclose(points[:, 2], 0.), "Cannot have three dimensional points when symmetry is 2D"
        
        assert np.all( (0 <= self.lines) & (self.lines < len(self.points)) ), "Lines reference points outside points array"
        assert np.all( (0 <= self.triangles) & (self.triangles < len(self.points)) ), "Triangles reference points outside points array"
        assert np.all([np.all( (0 <= group) & (group < len(self.lines)) ) for group in self.physical_to_lines.values()])
        assert np.all([np.all( (0 <= group) & (group < len(self.triangles)) ) for group in self.physical_to_triangles.values()])
        assert not len(self.lines) or self.lines.shape[1] in [2,4], "Lines should contain either 2 or 4 points."
        assert not len(self.triangles) or self.triangles.shape[1] in [3,6], "Triangles should contain either 3 or 6 points"
    
    def is_higher_order(self):
        return (len(self.lines) and self.lines.shape[1] == 4) or (len(self.triangles) and self.triangles.shape[1] == 6)
    
    def move(self, vector):
        self.points += vector
     
    def __add__(self, other):
        assert isinstance(other, Mesh)
        assert self.symmetry == other.symmetry, "Cannot add meshes with different symmetries"
         
        N_points = len(self.points)
        N_lines = len(self.lines)
        N_triangles = len(self.triangles)
         
        points = _concat_arrays(self.points, other.points)
        lines = _concat_arrays(self.lines, other.lines+N_points)
        triangles = _concat_arrays(self.triangles, other.triangles+N_points)
         
        physical_lines = {**self.physical_to_lines, **{k:(v+N_lines) for k, v in other.physical_to_lines.items()}}
        physical_triangles = {**self.physical_to_triangles, **{k:(v+N_triangles) for k, v in other.physical_to_triangles.items()}}
         
        return Mesh(self.symmetry,
                        points=points,
                        lines=lines,
                        triangles=triangles,
                        physical_to_lines=physical_lines,
                        physical_to_triangles=physical_triangles)
    
    def extract_physical_group(self, name):
        assert name in self.physical_to_lines or name in self.physical_to_triangles, "Physical group not in mesh, so cannot extract"

        if name in self.physical_to_lines:
            elements = self.lines
            physical = self.physical_to_lines
        elif name in self.physical_to_triangles:
            elements = self.triangles
            physical = self.physical_to_triangles
         
        elements_indices = np.unique(physical[name])
        elements = elements[elements_indices]
          
        points_mask = np.full(len(self.points), False)
        points_mask[elements] = True
        points = self.points[points_mask]
          
        new_index = np.cumsum(points_mask) - 1
        elements = new_index[elements]
        physical_to_elements = {name:np.arange(len(elements))}
         
        if name in self.physical_to_lines:
            return Mesh(self.symmetry, points=points, lines=elements, physical_to_lines=physical_to_elements)
        elif name in self.physical_to_triangles:
            return Mesh(self.symmetry, points=points, triangles=triangles, physical_to_triangles=physical_to_elements)
     
    def import_file(filename, symmetry,  name=None):
        meshio_obj = meshio.read(filename)
        mesh = Mesh.from_meshio(meshio_obj, symmetry)
         
        if name is not None:
            mesh.physical_to_lines[name] = np.arange(len(mesh.lines))
            mesh.physical_to_triangles[name] = np.arange(len(mesh.triangles))
         
        return mesh
     
    def export_file(self, filename):
        meshio_obj = self.to_meshio()
        meshio_obj.write(filename)
     
    def to_meshio(self):
        to_export = []
        
        if len(self.lines):
            line_type = 'line' if self.lines.shape[1] == 2 else 'line4'
            to_export.append( (line_type, self.lines) )
        
        if len(self.triangles):
            triangle_type = 'triangle' if self.triangles.shape[1] == 3 else 'triangle6'
            to_export.append( (triangle_type, self.triangles) )
        
        return meshio.Mesh(self.points, to_export)
     
    def from_meshio(mesh, symmetry):
        """Generate a Traceon Mesh from a [meshio](https://github.com/nschloe/meshio) mesh.
        
        Parameters
        ----------
        symmetry: Symmetry
            Specifies a radially symmetric geometry (RADIAL) or a general 3D geometry (THREE_D).
        
        Returns
        ---------
        Mesh
        """
        def extract(type_):
            elements = mesh.cells_dict[type_]
            physical = {k:v[type_] for k,v in mesh.cell_sets_dict.items() if type_ in v}
            return elements, physical
        
        lines, physical_lines = [], {}
        triangles, physical_triangles = [], {}
        
        if 'line' in mesh.cells_dict:
            lines, physical_lines = extract('line')
        elif 'line4' in mesh.cells_dict:
            lines, physical_lines = extract('line4')
        
        if 'triangle' in mesh.cells_dict:
            triangles, physical_triangles = extract('triangle')
        
        if 'triangle6' in mesh.cells_dict:
            logging.warning('triangle6 present in mesh but not supported')
         
        return Mesh(symmetry,
            points=mesh.points,
            lines=lines, physical_to_lines=physical_lines,
            triangles=triangles, physical_to_triangles=physical_triangles)
     
    def is_3d(self):
        """Check if the mesh is three dimensional.

        Returns
        ----------------
        True if mesh is three dimensional, False if the mesh is two dimensional"""
        return self.symmetry.is_3d()
    
    def is_2d(self):
        """Check if the mesh is two dimensional.
        
        Returns
        ----------------
        True if mesh is two dimensional, False if the mesh is three dimensional"""
        return self.symmetry.is_2d()

    def remove_lines(self):
        return Mesh(self.symmetry, self.points, triangles=self.triangles, physical_to_triangles=self.physical_to_triangles)
    
    def remove_triangles(self):
        return Mesh(self.symmetry, self.points, lines=self.lines, physical_to_lines=self.physical_to_lines)
     
    def get_electrodes(self):
        """Get the names of all the electrodes in the geometry.
         
        Returns
        ---------
        List of electrode names

        """
        return list(self.physical_to_lines.keys()) + list(self.physical_to_triangles.keys())
     
    def _lines_to_higher_order(points, elements):
        N_elements = len(elements)
        N_points = len(points)
         
        v0, v1 = elements.T
        p2 = points[v0] + (points[v1] - points[v0]) * 1/3
        p3 = points[v0] + (points[v1] - points[v0]) * 2/3
         
        assert all(p.shape == (N_elements, points.shape[1]) for p in [p2, p3])
         
        points = np.concatenate( (points, p2, p3), axis=0)
          
        elements = np.array([
            elements[:, 0], elements[:, 1], 
            np.arange(N_points, N_points + N_elements, dtype=np.uint64),
            np.arange(N_points + N_elements, N_points + 2*N_elements, dtype=np.uint64)]).T
         
        assert np.allclose(p2, points[elements[:, 2]]) and np.allclose(p3, points[elements[:, 3]])
        return points, elements
    
    def _to_higher_order_mesh(self):
        assert self.is_2d()
        # The matrix solver currently only works with higher order meshes.
        # We can however convert a simple mesh easily to a higher order mesh, and solve that.
        
        points, lines, triangles = self.points, self.lines, self.triangles
        
        if len(lines) and lines.shape[1] == 2:
            points, lines = Mesh._lines_to_higher_order(points, lines)
         
        return Mesh(self.symmetry,
            points=points,
            lines=lines, physical_to_lines=self.physical_to_lines,
            triangles=triangles, physical_to_triangles=self.physical_to_triangles)
     
    def __str__(self):
        physical_lines = ', '.join(self.physical_to_lines.keys())
        physical_lines_nums = ', '.join([str(len(self.physical_to_lines[n])) for n in self.physical_to_lines.keys()])
        physical_triangles = ', '.join(self.physical_to_triangles.keys())
        physical_triangles_nums = ', '.join([str(len(self.physical_to_triangles[n])) for n in self.physical_to_triangles.keys()])
        
        return f'<Traceon Mesh {self.symmetry},\n' \
            f'\tNumber of points: {len(self.points)}\n' \
            f'\tNumber of lines: {len(self.lines)}\n' \
            f'\tNumber of triangles: {len(self.triangles)}\n' \
            f'\tPhysical lines: {physical_lines}\n' \
            f'\tElements in physical line groups: {physical_lines_nums}\n' \
            f'\tPhysical triangles: {physical_triangles}\n' \
            f'\tElements in physical triangle groups: {physical_triangles_nums}>'


class MEMSStack(Geometry):
    """Geometry consisting of a stack of MEMS fabricated elements. This geometry is modelled using a stack
    of rectangularly shaped elements with a variable spacing in between. Useful when doing calculations on MEMS fabricated
    lenses and mirrors.
    
    Parameters
    ----------
    z0: float
        Starting z-value to begin building up the MEMS elements from.
    revolve_factor: float
        Revolve the resulting geometry around the optical axis to generate a 3D geometry. When `revolve_factor=0.0` a
        2D geometry is returned. For `0 < revolve_factor <= 1.0` see the documentation of `revolve_around_optical_axis`.
    rmax: float
        The rectangular MEMS objects extend to \( r = r_{max} \).
    margin: float
        The distance between the electrodes and the top and bottom boundary.
    margin_right: float
        Distance between the boundary on the right and the MEMS electrodes.
    symmetry: `Symmetry`
        What symmetry to use for the resulting geometry
    """
    
    def __init__(self, *args, z0=0.0, revolve_factor=0.0, rmax=2, margin=0.5, margin_right=0.1, symmetry=None, **kwargs):
        
        if symmetry is None:
            symmetry = Symmetry.RADIAL if revolve_factor == 0.0 else Symmetry.THREE_D
        else:
            symmetry = symmetry
            
            if revolve_factor == 0.0 and symmetry.is_3d():
                revolve_factor = 1.0

        super().__init__(symmetry, *args, **kwargs)
        
        self.z0 = z0
        self.revolve_factor = revolve_factor
        self.rmax = rmax
        
        self.margin = margin
        self.margin_right = margin_right
         
        self._current_z = z0 + margin
        self._last_name = None
    
    def add_spacer(self, thickness):
        """

        Parameters
        ----------
        thickness : float
            Add the given amount of space between the previous and next electrode.

        """
        self._current_z += thickness
    
    def _add_boundary(self):
        points = [[0.0, self._current_z+self.margin],
                  [self.rmax+self.margin_right, self._current_z+self.margin],
                  [self.rmax+self.margin_right, self.z0],
                  [0.0, self.z0]]
         
        self._add_lines_from_points(points, 'boundary')
        self._current_z += self.margin
     
    def _add_lines_from_points(self, points, name):
        if self.is_3d():
            points = [self.add_point([p[0], 0.0, p[1]]) for p in points[::-1]]
        else:
            points = [self.add_point(p) for p in points]
        
        Np = len(points)
        lines = [self.add_line(points[i], points[j]) for i, j in zip(range(0,Np-1), range(1,Np))]
         
        if self.is_3d():
            revolved = revolve_around_optical_axis(self, lines, self.revolve_factor)
            self.add_physical(revolved, name)
        else:
            self.add_physical(lines, name)

    def add_electrode(self, radius, thickness, name):
        """

        Parameters
        ----------
        radius : float
            Distance from the electrode to the optical axis (in mm).
            
        thickness : float
            Thickness of the electrode (in mm).
            
        name : str
            Name to assign to the electode. Needed to later specify the correct excitation.
        """
        cz = self._current_z
        points = [[self.rmax, cz], [radius, cz], [radius, cz+thickness], [self.rmax, cz+thickness]]
        self._add_lines_from_points(points, name)
        self._current_z += thickness

    def generate_line_mesh(self, *args, **kwargs):
        self._add_boundary()
        return super().generate_line_mesh(*args, **kwargs)
    
    def generate_triangle_mesh(self, *args, **kwargs):
        self._add_boundary()
        return super().generate_triangle_mesh(*args, **kwargs)
 
    
        

