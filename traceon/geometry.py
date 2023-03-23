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

import pickle

from .util import Saveable


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
    THREE_D = 1

    def __str__(self):
        if self == Symmetry.RADIAL:
            return 'radial'
        elif self == Symmetry.THREE_D:
            return '3d'

class Geometry(occ.Geometry):
    """
    Small wrapper class around pygmsh.occ.Geometry which itself is a small wrapper around the powerful GMSH library.
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
        supplied symmetry. The arguments are passed directly to `pygmsh.occ.Geometry.generate_mesh`.
        
        Returns
        -------
        `Mesh`

        """
        for label, entities in self._physical_queue.items():
            super().add_physical(entities, label)
          
        if self.size_from_distance:
            self.set_mesh_size_callback(self._mesh_size_callback)
        
        dim = 2 if self.symmetry == Symmetry.THREE_D else 1
        
        return Mesh(super().generate_mesh(dim=dim, *args, **kwargs), self.symmetry)

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
        elif self.symmetry == Symmetry.THREE_D:
            # GMSH seems to produce meshes which contain way more elements for 3D geometries
            # with the same mesh factor. This is confusing for users and therefore we arbtrarily
            # incrase the mesh size to roughly correspond with the 2D number of elements.
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
        
        z_optical = z if self.symmetry == Symmetry.THREE_D else y
         
        if self.zmin is not None:
            z_optical = max(z_optical, self.zmin)
        if self.zmax is not None:
            z_optical = min(z_optical, self.zmax)
         
        if self.symmetry == Symmetry.THREE_D:
            return sqrt( x**2 + y**2 + (z-z_optical)**2 )
        else:
            return sqrt( x**2 + (y-z_optical)**2 )



class Mesh(Saveable):
    """Class containing a mesh and related metadata."""
    
    def __init__(self, mesh, symmetry, metadata={}):
        assert isinstance(symmetry, Symmetry)
        self.mesh = mesh
        self.metadata = metadata
        self.symmetry = symmetry
     
    def get_electrodes(self):
        """Get the names of all the electrodes in the geometry.
         
        Returns
        ---------
        List of electrode names

        """
        return list(self.mesh.cell_sets_dict.keys())
    
    def __str__(self):
        physicals = self.mesh.cell_sets_dict.keys()
        physical_names = ', '.join(physicals)
        type_ = 'line' if self.symmetry != Symmetry.THREE_D else 'triangle'
        physical_nums = ', '.join([str(len(self.mesh.cell_sets_dict[n][type_])) for n in physicals])
        
        cells_type = ['point'] + [str(c.type) for c in self.mesh.cells]
        cells_count = [len(self.mesh.points)] + [len(c) for c in self.mesh.cells]
        
        return f'<Traceon Mesh {self.symmetry},\n' \
            f'\tPhysical groups: {physical_names}\n' \
            f'\tElements in physical groups: {physical_nums}\n' \
            f'\tNumber of.. \n\t    ' \
            + '\n\t    '.join([f'{t}: \t{c}' for t, c in zip(cells_type, cells_count)]) \
            + '>'


        #return f'<Traceon Mesh with {len(self.mesh.points)} points, ', '.join(self.mesh.cell_sets_dict.keys()))
        #return str(self.mesh) + ' (metadata: ' + str(self.metadata) + ')'

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
    enclose_right: bool
        When creating a MEMS component it is important to have well specified boundary conditions above and
        beneath the element. This is usually achieved by having grounded electrodes at the top and bottom of the stack.
        To finish the grounded enclosure a grounded elements should connect these electrodes vertically at the right
        side of the stack. 
    margin_right: float
        Distance between the grounded enclosure on the right and the MEMS electrodes.
    """
    
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
        """

        Parameters
        ----------
        thickness : float
            Add the given amount of space between the previous and next electrode.

        """
        self._current_z += thickness
    
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
            self.add_physical(revolved, name)
        else:
            self.add_physical(lines, name)
        
        self._last_name = name
         
        self._current_z += thickness
        
        return cl
    
    def generate_mesh(self, *args, **kwargs):
        """
        Generate the mesh, determining the mesh dimension (line elements or triangles) from the
        supplied `revolve_factor`. The arguments are passed directly to `pygmsh.occ.Geometry.generate_mesh`.
        
        Returns
        -------
        `Mesh`

        """
        # Enclose on right
        
        if self.enclose_right:
            points = [[self.rmax, self.z0], [self.rmax + self.margin_right, self.z0], [self.rmax + self.margin_right, self._current_z], [self.rmax, self._current_z]]
            
            if self._3d:
                points = [[p[0], 0.0, p[1]] for p in points]
                lines = [self.add_line(self.add_point(p1), self.add_point(p2)) for p1, p2 in zip(points[1:], points)]
                revolved = revolve_around_optical_axis(self, lines, self.revolve_factor)
                self.add_physical(revolved, self._last_name)
            else:
                lines = [self.add_line(self.add_point(p1), self.add_point(p2)) for p1, p2 in zip(points[1:], points)]
                self.add_physical(lines, self._last_name)
        
        return super().generate_mesh(*args, **kwargs)        

    
        

