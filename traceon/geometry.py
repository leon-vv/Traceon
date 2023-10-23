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


import numpy as np
from math import sqrt
from pygmsh import *
import gmsh
from enum import Enum
import copy

import pickle

from .util import Saveable
from .backend import N_QUAD_2D, position_and_jacobian_radial, position_and_jacobian_3d, normal_3d

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
    THREE_D_HIGHER_ORDER = 1
    THREE_D = 2

    def __str__(self):
        if self == Symmetry.RADIAL:
            return 'radial'
        elif self == Symmetry.THREE_D_HIGHER_ORDER:
            return '3d higher order'
        elif self == Symmetry.THREE_D:
            return '3d first order' 
    
    def is_3d(self):
        return self in [Symmetry.THREE_D_HIGHER_ORDER, Symmetry.THREE_D]

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

    def generate_mesh(self, *args, **kwargs):
        """
        Generate the mesh, determining the mesh dimension (line elements or triangles) from the
        supplied symmetry. The arguments are passed directly to `pygmsh.geo.Geometry.generate_mesh`.
        
        Returns
        -------
        `Mesh`

        """
        for label, entities in self._physical_queue.items():
            super().add_physical(entities, label)
          
        if self.size_from_distance:
            self.set_mesh_size_callback(self._mesh_size_callback)
         
        if self.symmetry == Symmetry.RADIAL:
            dim = 1
            gmsh.option.setNumber('Mesh.ElementOrder', 3)
        elif self.symmetry == Symmetry.THREE_D:
            dim = 2
            gmsh.option.setNumber('Mesh.ElementOrder', 1)
        elif self.symmetry == Symmetry.THREE_D_HIGHER_ORDER:
            dim = 2
            gmsh.option.setNumber('Mesh.ElementOrder', 2)
        else:
            raise ValueError('Symmetry not valid: ', self.symmetry)
        
        return Mesh.from_meshio(super().generate_mesh(dim=dim, *args, **kwargs), self.symmetry)

    def set_mesh_size_factor(self, factor):
        """
        Set the mesh size factor. Which simply scales with the total number of elements in the mesh.
        
        Parameters
        ----------
        factor : float
            The mesh size factor to use. 
        
        """
        if self.symmetry == Symmetry.RADIAL:
            gmsh.option.setNumber('Mesh.MeshSizeFactor', 1/factor)
        elif self.symmetry == Symmetry.THREE_D_HIGHER_ORDER or self.symmetry == Symmetry.THREE_D:
            # GMSH seems to produce meshes which contain way more elements for 3D geometries
            # with the same mesh factor. This is confusing for users and therefore we arbtrarily
            # increase the mesh size to roughly correspond with the 2D number of elements.
            gmsh.option.setNumber('Mesh.MeshSizeFactor', 4*sqrt(1/factor))

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


class Mesh(Saveable):
    """Class containing a mesh and related metadata."""
    
    def __init__(self, points, elements, physical_to_elements, symmetry, metadata={}):
        assert isinstance(symmetry, Symmetry)
        self.points = points
        self.elements = elements
        self.physical_to_elements = physical_to_elements
        self.symmetry = symmetry
        self.metadata = metadata
            
    def from_meshio(mesh, symmetry, metadata={}):
        """Generate a Traceon Mesh from a [meshio](https://github.com/nschloe/meshio) mesh.
        
        Parameters
        ----------
        symmetry: Symmetry
            Specifies a radially symmetric geometry (RADIAL) or a general 3D geometry (THREE_D_HIGHER_ORDER).
        
        Returns
        ---------
        Mesh
        """
        if symmetry == Symmetry.RADIAL:
            type_ = 'line4'
        elif symmetry == Symmetry.THREE_D:
            type_ = 'triangle'
        elif symmetry == Symmetry.THREE_D_HIGHER_ORDER:
            type_ = 'triangle6'
        else:
            raise ValueError('Symmetry passed to from_meshio(..) not valid: ' + str(symmetry))
         
        points = mesh.points
        elements = mesh.cells_dict[type_]
        physical_to_elements = {k:v[type_] for k, v in mesh.cell_sets_dict.items() if type_ in v}
        
        return Mesh(points, elements, physical_to_elements, symmetry, metadata)
     
    def is_3d(self):
        """Check if the mesh is three dimensional.

        Returns
        ----------------
        True if mesh is three dimensional, False if the mesh is two dimensional"""
        return self.symmetry.is_3d()
    
    def get_electrodes(self):
        """Get the names of all the electrodes in the geometry.
         
        Returns
        ---------
        List of electrode names

        """
        return list(self.physical_to_elements.keys())
    
    def _invert_physical_dict(self):
        lookup = np.full(len(self.elements), None)

        for k, v in self.physical_to_elements.items():
            lookup[v] = k
        
        return lookup
     
    def _split_indices_radial(self, indices):
        assert self.symmetry == Symmetry.RADIAL
         
        elements = copy.deepcopy(self.elements)
        N_elements = len(elements)
         
        N = len(self.points)
        lines_to_add = []
        physicals = []
        points_to_add = []
         
        physical_lookup = self._invert_physical_dict()
        to_pos = lambda alpha, line: [*position_and_jacobian_radial(alpha, line[0], line[2], line[3], line[1])[1], 0.0]
        
        for idx in indices:
            pi = elements[idx]
            line = self.points[pi]
             
            points_to_add.append(to_pos(-2/3, line))
            points_to_add.append(to_pos(0.0, line))
            points_to_add.append(to_pos(2/3, line))
            l = len(points_to_add)
            lines_to_add.append( (N+l-2, pi[1].item(), pi[3].item(), N+l-1) )
            elements[idx] = (pi[0], N+l-2, N+l-3, pi[2])
            physicals.append(physical_lookup[idx])
        
        # Now actually alter the mesh
        new_points = np.concatenate( (self.points, points_to_add), axis=0)
        new_elements = np.concatenate( (elements, np.array(lines_to_add, dtype=np.uint64)), axis=0)
        new_dict = copy.copy(self.physical_to_elements)
        
        for name in [p for p in np.unique(physicals) if p is not None]:   
            old_indices = new_dict[name]
            (added_indices,) = (np.array(physicals) == name).nonzero()
            new_dict[name] = np.concatenate( (old_indices.astype(np.int64), N_elements + added_indices) )
         
        new_mesh = Mesh(new_points, new_elements, new_dict, self.symmetry, self.metadata)
        return new_mesh
         
    def _split_indices_3d(self, indices):
        assert self.symmetry == Symmetry.THREE_D_HIGHER_ORDER
         
        elements = copy.deepcopy(self.elements)
        N_elements = len(elements)
         
        N = len(self.points)
        triangles_to_add = []
        physicals = []
        points_to_add = []
         
        physical_lookup = self._invert_physical_dict()
        to_pos = lambda alpha, beta, triangle: position_and_jacobian_3d(alpha, beta, triangle)[1]
        
        for idx in indices:
            pi = elements[idx]
            triangle = self.points[pi]
             
            points_to_add.append(to_pos(1/3, 1/3, triangle)) # Middle
            points_to_add.append(to_pos(1/6, 1/6, triangle)) # s0
            points_to_add.append(to_pos(4/6, 1/6, triangle)) # s1
            points_to_add.append(to_pos(1/6, 4/6, triangle)) # s2
            l = len(points_to_add)
            # Same ordering as in C backend function 'fill_self_voltages_3d'
            t, s0, s1, s2 = N+l-4, N+l-3, N+l-2, N+l-1
            triangles_to_add.append( (t, pi[1], pi[2], s1, pi[4], s2) )
            triangles_to_add.append( (t, pi[2], pi[0], s2, pi[5], s0) )
            elements[idx] = (t, pi[0], pi[1], s0, pi[3], s1)
            
            physicals.append(physical_lookup[idx])
            physicals.append(physical_lookup[idx])
        
        # Now actually alter the mesh
        new_points = np.concatenate( (self.points, points_to_add), axis=0)
        new_elements = np.concatenate( (elements, np.array(triangles_to_add, dtype=np.uint64)), axis=0)
        new_dict = copy.copy(self.physical_to_elements)
        
        for name in [p for p in np.unique(physicals) if p is not None]:   
            old_indices = new_dict[name]
            (added_indices,) = (np.array(physicals) == name).nonzero()
            new_dict[name] = np.concatenate( (old_indices.astype(np.int64), N_elements + added_indices) )
         
        new_mesh = Mesh(new_points, new_elements, new_dict, self.symmetry, self.metadata)
        return new_mesh
     
    def split_indices(self, indices):
        if self.symmetry == Symmetry.RADIAL:
            return self._split_indices_radial(indices)
        else:
            return self._split_indices_3d(indices)

    def split_elements_based_on_charges(self, excitation, field, max_splits, mesh_factor):
        active = excitation.get_active_element_mask()
        assert np.sum(active) == len(field.vertices), "Excitation did not produce the given field"

        map_index = active.nonzero()[0]
        
        charges = np.array([field.charge_on_element(i) for i in map_index])
        charges = np.abs( np.array([field.charge_on_element(i) for i in range(len(field.vertices))]) )
         
        # In max_splits iterations, increase the number of elements in the
        # mesh by mesh_factor. The split_facotr then gives us the amount
        # of elements we need to split in every iteration.
        split_factor = mesh_factor**(1/max_splits) - 1
        
        if self.symmetry == Symmetry.THREE_D_HIGHER_ORDER:
            split_factor /= 2 # For triangles, a splitting gives two extra elements, instead of one
        
        new_mesh = self
              
        for _ in range(max_splits):
            to_split = np.argsort(charges)[-round(split_factor*len(charges)):]
            print('Splitting ', len(to_split), ' elements')
            charges[to_split] /= 2 if self.symmetry == Symmetry.RADIAL else 3
            charges = np.concatenate( (charges, np.repeat(charges[to_split], 2)), axis=0 )
            new_mesh = new_mesh.split_indices(to_split)
        
        return new_mesh
     
    def __str__(self):
        physicals = self.physical_to_elements.keys()
        physical_names = ', '.join(physicals)
        physical_nums = ', '.join([str(len(self.physical_to_elements[n])) for n in physicals])
        
        return f'<Traceon Mesh {self.symmetry},\n' \
            f'\tPhysical groups: {physical_names}\n' \
            f'\tElements in physical groups: {physical_nums}\n' \
            f'\tNumber of points: {len(self.points)}>'

    def write_gmsh(self, filename):
        self.mesh.write(filename)



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
            self.symmetry = Symmetry.RADIAL if revolve_factor == 0.0 else Symmetry.THREE_D_HIGHER_ORDER
        else:
            self.symmetry = symmetry
            
            if revolve_factor == 0.0:
                revolve_factor = 1.0
         
        super().__init__(self.symmetry, *args, **kwargs)
        
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

    def generate_mesh(self, *args, **kwargs):
        self._add_boundary()
        return super().generate_mesh(*args, **kwargs)
        
        

