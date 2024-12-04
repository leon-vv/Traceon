from math import sqrt
import numpy as np
import time
from typing import Callable, Any
from itertools import chain
from abc import ABC, abstractmethod
import ctypes as C
from typing import Dict

import meshio

from .util import Saveable
from .backend import triangle_areas
from .logging import log_debug
from . import backend as B


__pdoc__ = {}
__pdoc__['PointsWithQuads'] = False
__pdoc__['PointStack'] = False
__pdoc__['Mesh.__add__'] = True


class GeometricObject(ABC):
    """The Mesh class (and the classes defined in `traceon.geometry`) are subclasses
    of GeometricObject. This means that they all can be moved, rotated, mirrored."""
    
    @abstractmethod
    def map_points(self, fun: Callable[[np.ndarray], np.ndarray]) -> Any:
        """Create a new geometric object, by mapping each point by a function.
        
        Parameters
        -------------------------
        fun: (3,) float -> (3,) float
            Function taking a three dimensional point and returning a 
            three dimensional point.

        Returns
        ------------------------
        GeometricObject

        This function returns the same type as the object on which this method was called."""
        ...
    
    def move(self, dx=0., dy=0., dz=0.):
        """Move along x, y or z axis.

        Parameters
        ---------------------------
        dx: float
            Amount to move along the x-axis.
        dy: float
            Amount to move along the y-axis.
        dz: float
            Amount to move along the z-axis.

        Returns
        ---------------------------
        GeometricObject
        
        This function returns the same type as the object on which this method was called."""
    
        assert all([isinstance(d, float) or isinstance(d, int) for d in [dx, dy, dz]])
        return self.map_points(lambda p: p + np.array([dx, dy, dz]))
     
    def rotate(self, Rx=0., Ry=0., Rz=0., origin=[0., 0., 0.]):
        """Rotate counter clockwise around the x, y or z axis. Only one axis supported at the same time
        (rotations do not commute).

        Parameters
        ------------------------------------
        Rx: float
            Amount to rotate around the x-axis (radians).
        Ry: float
            Amount to rotate around the y-axis (radians).
        Rz: float
            Amount to rotate around the z-axis (radians).
        origin: (3,) float
            Point around which to rotate, which is the origin by default.

        Returns
        --------------------------------------
        GeometricObject
        
        This function returns the same type as the object on which this method was called."""
        
        assert sum([Rx==0., Ry==0., Rz==0.]) >= 2, "Only supply one axis of rotation"
        origin = np.array(origin)
        assert origin.shape == (3,), "Please supply a 3D point for origin"
         
        if Rx != 0.:
            matrix = np.array([[1, 0, 0],
                [0, np.cos(Rx), -np.sin(Rx)],
                [0, np.sin(Rx), np.cos(Rx)]])
        elif Ry != 0.:
            matrix = np.array([[np.cos(Ry), 0, np.sin(Ry)],
                [0, 1, 0],
                [-np.sin(Ry), 0, np.cos(Ry)]])
        elif Rz != 0.:
            matrix = np.array([[np.cos(Rz), -np.sin(Rz), 0],
                [np.sin(Rz), np.cos(Rz), 0],
                [0, 0, 1]])

        return self.map_points(lambda p: origin + matrix @ (p - origin))

    def mirror_xz(self):
        """Mirror object in the XZ plane.

        Returns
        --------------------------------------
        GeometricObject
        
        This function returns the same type as the object on which this method was called."""
        return self.map_points(lambda p: np.array([p[0], -p[1], p[2]]))
     
    def mirror_yz(self):
        """Mirror object in the YZ plane.

        Returns
        --------------------------------------
        GeometricObject
        This function returns the same type as the object on which this method was called."""
        return self.map_points(lambda p: np.array([-p[0], p[1], p[2]]))
    
    def mirror_xy(self):
        """Mirror object in the XY plane.

        Returns
        --------------------------------------
        GeometricObject
        
        This function returns the same type as the object on which this method was called."""
        return self.map_points(lambda p: np.array([p[0], p[1], -p[2]]))
 

def _concat_arrays(arr1, arr2):
    if not len(arr1):
        return np.copy(arr2)
    if not len(arr2):
        return np.copy(arr1)
      
    assert arr1.shape[1:] == arr2.shape[1:], "Cannot add meshes if one is higher order and the other is not"
    
    return np.concatenate( (arr1, arr2), axis=0)

class Mesh(Saveable, GeometricObject):
    """Mesh containing lines and triangles. Groups of lines or triangles can be named. These
    names are later used to apply the correct excitation. Line elements can be curved (or 'higher order'), 
    in which case they are represented by four points per element.  Note that `Mesh` is a subclass of
    `traceon.mesher.GeometricObject`, and therefore can be easily moved and rotated."""
     
    def __init__(self,
            points=[],
            lines=[],
            triangles=[],
            physical_to_lines={},
            physical_to_triangles={},
            ensure_outward_normals=True):
        
        # Ensure the correct shape even if empty arrays
        if len(points):
            self.points = np.array(points, dtype=np.float64)
        else:
            self.points = np.empty((0,3), dtype=np.float64)
         
        if len(lines) or (isinstance(lines, np.ndarray) and len(lines.shape)==2):
            self.lines = np.array(lines, dtype=np.uint64)
        else:
            self.lines = np.empty((0,2), dtype=np.uint64)
    
        if len(triangles):
            self.triangles = np.array(triangles, dtype=np.uint64)
        else:
            self.triangles = np.empty((0, 3), dtype=np.uint64)
         
        self.physical_to_lines = physical_to_lines.copy()
        self.physical_to_triangles = physical_to_triangles.copy()

        self._remove_degenerate_triangles()
        self._deduplicate_points()

        if ensure_outward_normals:
            for el in self.get_electrodes():
                self.ensure_outward_normals(el)
         
        assert np.all( (0 <= self.lines) & (self.lines < len(self.points)) ), "Lines reference points outside points array"
        assert np.all( (0 <= self.triangles) & (self.triangles < len(self.points)) ), "Triangles reference points outside points array"
        assert np.all([np.all( (0 <= group) & (group < len(self.lines)) ) for group in self.physical_to_lines.values()])
        assert np.all([np.all( (0 <= group) & (group < len(self.triangles)) ) for group in self.physical_to_triangles.values()])
        assert not len(self.lines) or self.lines.shape[1] in [2,4], "Lines should contain either 2 or 4 points."
        assert not len(self.triangles) or self.triangles.shape[1] in [3,6], "Triangles should contain either 3 or 6 points"
    
    def is_higher_order(self):
        """Whether the mesh contains higher order elements.

        Returns
        ----------------------------
        bool"""
        return isinstance(self.lines, np.ndarray) and len(self.lines.shape) == 2 and self.lines.shape[1] == 4
    
    def map_points(self, fun):
        """See `GeometricObject`

        """
        new_points = np.empty_like(self.points)
        for i in range(len(self.points)):
            new_points[i] = fun(self.points[i])
        assert new_points.shape == self.points.shape and new_points.dtype == self.points.dtype
        
        return Mesh(new_points, self.lines, self.triangles, self.physical_to_lines, self.physical_to_triangles)
    
    def _remove_degenerate_triangles(self):
        areas = triangle_areas(self.points[self.triangles[:,:3]])
        degenerate = areas < 1e-12
        map_index = np.arange(len(self.triangles)) - np.cumsum(degenerate)
         
        self.triangles = self.triangles[~degenerate]
        
        for k in self.physical_to_triangles.keys():
            v = self.physical_to_triangles[k]
            self.physical_to_triangles[k] = map_index[v[~degenerate[v]]]
         
        if np.any(degenerate):
            log_debug(f'Removed {sum(degenerate)} degenerate triangles')

    def _deduplicate_points(self):
        if not len(self.points):
            return
         
        # Step 1: Make a copy of the points array using np.array
        points_copy = np.array(self.points, dtype=np.float64)

        # Step 2: Zero the low 16 bits of the mantissa of the X, Y, Z coordinates
        points_copy.view(np.uint64)[:] &= np.uint64(0xFFFFFFFFFFFF0000)

        # Step 3: Use Numpy lexsort directly on points_copy
        sorted_indices = np.lexsort(points_copy.T)
        points_sorted = points_copy[sorted_indices]

        # Step 4: Create a mask to identify unique points
        equal_to_previous = np.all(points_sorted[1:] == points_sorted[:-1], axis=1)
        keep_mask = np.concatenate(([True], ~equal_to_previous))

        # Step 5: Compute new indices for the unique points
        new_indices_in_sorted_order = np.cumsum(keep_mask) - 1

        # Map old indices to new indices
        old_to_new_indices = np.empty(len(points_copy), dtype=np.uint64)
        old_to_new_indices[sorted_indices] = new_indices_in_sorted_order
        
        # Step 6: Update the points array with unique points
        self.points = points_sorted[keep_mask]

        # Step 7: Update all indices
        if len(self.triangles):
            self.triangles = old_to_new_indices[self.triangles]
        if len(self.lines):
            self.lines = old_to_new_indices[self.lines]
    
    @staticmethod
    def _merge_dicts(dict1, dict2):
        dict_: dict[str, np.ndarray] = {}
        
        for (k, v) in chain(dict1.items(), dict2.items()):
            if k in dict_:
                dict_[k] = np.concatenate( (dict_[k], v), axis=0)
            else:
                dict_[k] = v

        return dict_
     
    def __add__(self, other):
        """Add meshes together, using the + operator (mesh1 + mesh2).
        
        Returns
        ------------------------------
        Mesh

        A new mesh consisting of the elements of the added meshes"""
        if not isinstance(other, Mesh):
            return NotImplemented
         
        N_points = len(self.points)
        N_lines = len(self.lines)
        N_triangles = len(self.triangles)
         
        points = _concat_arrays(self.points, other.points)
        lines = _concat_arrays(self.lines, other.lines+N_points)
        triangles = _concat_arrays(self.triangles, other.triangles+N_points)

        other_physical_to_lines = {k:(v+N_lines) for k, v in other.physical_to_lines.items()}
        other_physical_to_triangles = {k:(v+N_triangles) for k, v in other.physical_to_triangles.items()}
         
        physical_lines = Mesh._merge_dicts(self.physical_to_lines, other_physical_to_lines)
        physical_triangles = Mesh._merge_dicts(self.physical_to_triangles, other_physical_to_triangles)
         
        return Mesh(points=points,
                    lines=lines,
                    triangles=triangles,
                    physical_to_lines=physical_lines,
                    physical_to_triangles=physical_triangles)
     
    def extract_physical_group(self, name):
        """Extract a named group from the mesh.

        Parameters
        ---------------------------
        name: str
            Name of the group of elements

        Returns
        --------------------------
        Mesh

        Subset of the mesh consisting only of the elements with the given name."""
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
            return Mesh(points=points, lines=elements, physical_to_lines=physical_to_elements)
        elif name in self.physical_to_triangles:
            return Mesh(points=points, triangles=elements, physical_to_triangles=physical_to_elements)
     
    @staticmethod
    def read_file(filename,  name=None):
        """Create a mesh from a given file. All formats supported by meshio are accepted.

        Parameters
        ------------------------------
        filename: str
            Path of the file to convert to Mesh
        name: str
            (optional) name to assign to all elements readed

        Returns
        -------------------------------
        Mesh"""
        meshio_obj = meshio.read(filename)
        mesh = Mesh.from_meshio(meshio_obj)
         
        if name is not None:
            mesh.physical_to_lines[name] = np.arange(len(mesh.lines))
            mesh.physical_to_triangles[name] = np.arange(len(mesh.triangles))
         
        return mesh
     
    def write_file(self, filename):
        """Write a mesh to a given file. The format is determined from the file extension.
        All formats supported by meshio are supported.

        Parameters
        ----------------------------------
        filename: str
            The name of the file to write the mesh to."""
        meshio_obj = self.to_meshio()
        meshio_obj.write(filename)
    
    def write(self, filename):
        self.write_file(filename)
        
     
    def to_meshio(self):
        """Convert the Mesh to a meshio object.

        Returns
        ------------------------------------
        meshio.Mesh"""
        to_write = []
        
        if len(self.lines):
            line_type = 'line' if self.lines.shape[1] == 2 else 'line4'
            to_write.append( (line_type, self.lines) )
        
        if len(self.triangles):
            triangle_type = 'triangle' if self.triangles.shape[1] == 3 else 'triangle6'
            to_write.append( (triangle_type, self.triangles) )
        
        return meshio.Mesh(self.points, to_write)
     
    @staticmethod
    def from_meshio(mesh: meshio.Mesh):
        """Create a Traceon mesh from a meshio.Mesh object.

        Parameters
        --------------------------
        mesh: meshio.Mesh
            The mesh to convert to a Traceon mesh

        Returns
        -------------------------
        Mesh"""
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
        elif 'triangle6' in mesh.cells_dict:
            triangles, physical_triangles = extract('triangle6')
        
        return Mesh(points=mesh.points,
            lines=lines, physical_to_lines=physical_lines,
            triangles=triangles, physical_to_triangles=physical_triangles)
     
    def is_3d(self):
        """Check if the mesh is three dimensional by checking whether any z coordinate is non-zero.

        Returns
        ----------------
        bool

        Whether the mesh is three dimensional"""
        return np.any(self.points[:, 1] != 0.)
    
    def is_2d(self):
        """Check if the mesh is two dimensional, by checking that all z coordinates are zero.
        
        Returns
        ----------------
        bool

        Whether the mesh is two dimensional"""
        return np.all(self.points[:, 1] == 0.)
    
    def flip_normals(self):
        """Flip the normals in the mesh by inverting the 'orientation' of the elements.

        Returns
        ----------------------------
        Mesh"""
        lines = self.lines
        triangles = self.triangles
        
        # Flip the orientation of the lines
        if lines.shape[1] == 4:
            p0, p1, p2, p3 = lines.T
            lines = np.array([p1, p0, p3, p2]).T
        else:
            p0, p1 = lines.T
            lines = np.array([p1, p0]).T
          
        # Flip the orientation of the triangles
        if triangles.shape[1] == 6:
            p0, p1, p2, p3, p4, p5 = triangles.T
            triangles = np.array([p0, p2, p1, p5, p4, p3]).T
        else:
            p0, p1, p2 = triangles.T
            triangles = np.array([p0, p2, p1]).T
        
        return Mesh(self.points, lines, triangles,
            physical_to_lines=self.physical_to_lines,
            physical_to_triangles=self.physical_to_triangles)
     
    def remove_lines(self):
        """Remove all the lines from the mesh.

        Returns
        -----------------------------
        Mesh"""
        return Mesh(self.points, triangles=self.triangles, physical_to_triangles=self.physical_to_triangles)
    
    def remove_triangles(self):
        """Remove all triangles from the mesh.

        Returns
        -------------------------------------
        Mesh"""
        return Mesh(self.points, lines=self.lines, physical_to_lines=self.physical_to_lines)
     
    def get_electrodes(self):
        """Get the names of all the named groups (i.e. electrodes) in the mesh
         
        Returns
        ---------
        str iterable

        Names
        """
        return list(self.physical_to_lines.keys()) + list(self.physical_to_triangles.keys())
     
    @staticmethod
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
        # The matrix solver currently only works with higher order meshes.
        # We can however convert a simple mesh easily to a higher order mesh, and solve that.
        
        points, lines, triangles = self.points, self.lines, self.triangles

        if not len(lines):
            lines = np.empty( (0, 4), dtype=np.float64)
        elif len(lines) and lines.shape[1] == 2:
            points, lines = Mesh._lines_to_higher_order(points, lines)
        
        assert lines.shape == (len(lines), 4)

        return Mesh(points=points,
            lines=lines, physical_to_lines=self.physical_to_lines,
            triangles=triangles, physical_to_triangles=self.physical_to_triangles)
     
    def __str__(self):
        physical_lines = ', '.join(self.physical_to_lines.keys())
        physical_lines_nums = ', '.join([str(len(self.physical_to_lines[n])) for n in self.physical_to_lines.keys()])
        physical_triangles = ', '.join(self.physical_to_triangles.keys())
        physical_triangles_nums = ', '.join([str(len(self.physical_to_triangles[n])) for n in self.physical_to_triangles.keys()])
        
        return f'<Traceon Mesh,\n' \
            f'\tNumber of points: {len(self.points)}\n' \
            f'\tNumber of lines: {len(self.lines)}\n' \
            f'\tNumber of triangles: {len(self.triangles)}\n' \
            f'\tPhysical lines: {physical_lines}\n' \
            f'\tElements in physical line groups: {physical_lines_nums}\n' \
            f'\tPhysical triangles: {physical_triangles}\n' \
            f'\tElements in physical triangle groups: {physical_triangles_nums}>'

    def _ensure_normal_orientation_triangles(self, electrode, outwards):
        assert electrode in self.physical_to_triangles, "electrode should be part of mesh"
        
        triangle_indices = self.physical_to_triangles[electrode]
        electrode_triangles = self.triangles[triangle_indices]
          
        if not len(electrode_triangles):
            return
        
        connected_indices = _get_connected_elements(electrode_triangles)
        
        for indices in connected_indices:
            connected_triangles = electrode_triangles[indices]
            _ensure_triangle_orientation(connected_triangles, self.points, outwards)
            electrode_triangles[indices] = connected_triangles

        self.triangles[triangle_indices] = electrode_triangles
     
    def _ensure_normal_orientation_lines(self, electrode, outwards):
        assert electrode in self.physical_to_lines, "electrode should be part of mesh"
        
        line_indices = self.physical_to_lines[electrode]
        electrode_lines = self.lines[line_indices]
          
        if not len(electrode_lines):
            return
        
        connected_indices = _get_connected_elements(electrode_lines)
        
        for indices in connected_indices:
            connected_lines = electrode_lines[indices]
            _ensure_line_orientation(connected_lines, self.points, outwards)
            electrode_lines[indices] = connected_lines

        self.lines[line_indices] = electrode_lines
     
    def ensure_outward_normals(self, electrode):
        if electrode in self.physical_to_triangles:
            self._ensure_normal_orientation_triangles(electrode, True)
        
        if electrode in self.physical_to_lines:
            self._ensure_normal_orientation_lines(electrode, True)
     
    def ensure_inward_normals(self, electrode):
        if electrode in self.physical_to_triangles:
            self._ensure_normal_orientation_triangles(electrode, False)
         
        if electrode in self.physical_to_lines:
            self._ensure_normal_orientation_lines(electrode, False)



## Code related to checking connectivity  

def _compute_vertex_to_indices(elements):
    # elements is either a list of line or triangles
    # containing indices into the points array
    
    vertex_to_indices: Dict[int, list] = {}

    for index, el in enumerate(elements):
        for vertex in el:
            if vertex not in vertex_to_indices:
                vertex_to_indices[vertex] = []
            vertex_to_indices[vertex].append(index)
    
    return vertex_to_indices

def _get_element_neighbours(element, vertex_to_indices):
    neighbours = []
    for vertex in element:
        n = vertex_to_indices.get(vertex, None)
        
        if n is not None:
            neighbours.extend(n)
    
    return neighbours

def _get_connected_elements(elements):
    # Get subsets of elements that are connected to each other
    # For triangle elements, this would be surfaces
    # For line elements, this would be unbroken paths
    vertex_to_indices = _compute_vertex_to_indices(elements)
    
    N = len(elements) 
    labels = np.full(N, -1, dtype=np.int32)
    component_id = 0

    for i in range(N):
        if labels[i] == -1:
            # Start a new component
            labels[i] = component_id
            queue = [elements[i]]
            while queue:
                current = queue.pop()
                for v in current:
                    for neighbor in vertex_to_indices[v]:
                        if labels[neighbor] == -1:
                            labels[neighbor] = component_id
                            queue.append(elements[neighbor])
            component_id += 1

    # Group elements by label
    connected_elements = []
    
    for comp_id in range(component_id):
        triangle_indices = np.where(labels == comp_id)[0]
        connected_elements.append(triangle_indices)
    
    return connected_elements


## Code related to checking if normals point inward or outwards, given
## that all the normals already agree on orientation (all inwards or all outwards)

def _are_triangle_normals_pointing_outwards(triangles, points):
    # Based on https://math.stackexchange.com/questions/689418/how-to-compute-surface-normal-pointing-out-of-the-object
    triangle_points = points[triangles]
    
    v0 = triangle_points[:, 0]
    v1 = triangle_points[:, 1]
    v2 = triangle_points[:, 2]

    mid_x = (v0[:, 0] + v1[:, 0] + v2[:, 0])/3

    normals = np.cross(v1-v0, v2-v0)
    double_area = np.linalg.norm(normals, axis=1)
    normals /= double_area[:, np.newaxis]

    return np.sum(mid_x * normals[:, 0] * double_area/2) > 0.0

def _are_line_normals_pointing_outwards(lines, points):
    # Based on https://math.stackexchange.com/questions/689418/how-to-compute-surface-normal-pointing-out-of-the-object
    vertices = points[lines[:, :2]]
    mid_x = (vertices[:, 0, 0] + vertices[:, 1, 0])/2.
     
    normals = np.array([B.normal_2d(v0_, v1_) for v0_, v1_ in vertices[:, :, [0,2]]])
    length = np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1)
    
    return np.sum(mid_x * normals[:, 0] * length) > 0.0

## Code related to ensuring normals over a connected surface/path agree in orientation.
## These functios allow us to flip all normals 'inward' or all 'outward'

def _reorient_triangles(triangles, points):
    if not len(triangles):
        return
     
    vertex_to_triangles = _compute_vertex_to_indices(triangles)
        
    oriented = np.full(len(triangles), False)
    active = [0]

    assert triangles.dtype == np.uint64
    assert points.dtype == np.float64
     
    triangles_ctypes = triangles.ctypes.data_as(C.c_void_p)
    points_ctypes = points.ctypes.data_as(C.c_void_p)
    
    while active:
        ti = active.pop()

        for n in _get_element_neighbours(triangles[ti], vertex_to_triangles):
            assert 0 <= n < len(triangles)
            
            if oriented[n]:
                continue

            equal_orientation = B.backend_lib.triangle_orientation_is_equal(ti, n, triangles_ctypes, points_ctypes)
            
            if equal_orientation == -1: # Vertex neighbours, but not edge neighbours
                continue
            elif equal_orientation == 0: # Not equal orientation
                v0, v1, v2 = triangles[n]
                triangles[n] = [v0, v2, v1] # Flip normal
             
            oriented[n] = True
            active.append(n)

def _ensure_triangle_orientation(triangles, points, should_be_outwards):
    # Ensure the triangles forming a closed surface have their
    # normals either outwards (should_be_outwards is True) or inwards (should_be_outwards is False)
    _reorient_triangles(triangles, points)
    
    outwards = _are_triangle_normals_pointing_outwards(triangles, points)
    
    if outwards != should_be_outwards:
        for i in range(len(triangles)):
            v0, v1, v2 = triangles[i]
            triangles[i] = [v0, v2, v1]

def _line_orientation_equal(index1, index2, lines):
    # Note that the orientation is equal if the lines do not
    # walk 'towards' or both 'away' from the common vertex
    p1, p2 = lines[index1, :2]
    n1, n2 = lines[index2, :2]
     
    if p2 == n1:
        return True # p1 -> p2 -> n1 -> n2, same orientation
    if n2 == p1:
        return True # n1 -> n2 -> p1 -> p2, same orientation

    return False

def _reorient_lines(lines, points):
    assert lines.shape == (len(lines), 2) or lines.shape == (len(lines), 4)
    
    # Reorient the normals of lines in the same direction, at this point
    # in the algorithm we don't know if we are orienting them outwards or inwards,
    # the important thing is that they all agree (all outwards or all inwards)
    
    # Only consider first two points, in the case of higher order lines
    vertex_to_indices = _compute_vertex_to_indices(lines[:, :2])
        
    oriented = np.full(len(lines), False)
    active = [0]
    
    while active:
        ti = active.pop()

        for n in _get_element_neighbours(lines[ti], vertex_to_indices):
            if oriented[n]:
                continue
             
            if not _line_orientation_equal(ti, n, lines):
                # Orientation of neighbour differs, flip it
                neighbour = lines[n]
                 
                if neighbour.shape[0] == 4:
                    p0, p1, p2, p3 = neighbour
                    lines[n] = [p1, p0, p3, p2]
                else:
                    p0, p1 = neighbour
                    lines[n] = [p1, p0]
             
            oriented[n] = True
            active.append(n)

def _ensure_line_orientation(lines, points, should_be_outwards):
    # Ensure the triangles forming a closed surface have their
    # normals either outwards (should_be_outwards is True) or inwards (should_be_outwards is False)
    _reorient_lines(lines, points)
    
    outwards = _are_line_normals_pointing_outwards(lines, points)
    
    if outwards != should_be_outwards:
        for i in range(len(lines)):
            line = lines[i]
            if line.shape[0] == 4:
                p0, p1, p2, p3 = line
                lines[i] = [p1, p0, p3, p2]
            else:
                p0, p1 = line
                lines[i] = [p1, p0]





class PointsWithQuads:
    def __init__(self, indices, quads):
        N = len(indices)
        assert indices.shape == (N, N)
        assert np.all(quads[:, 1] < N)
        assert quads.shape == (len(quads), 5)
        assert np.all(quads[:, 0] == quads[0, 0])
        
        self.indices = indices
        self.quads = quads
        self.depth = quads[0, 0]
        
        self.shape = indices.shape
    
    def to_triangles(self):
        triangles = []

        def add_triangle(p0, p1, p2):
            triangles.append([self.indices[p0[0], p0[1]], self.indices[p1[0], p1[1]], self.indices[p2[0], p2[1]]])
         
        for quad in self.quads:
            depth, i0, i1, j0, j1 = quad 
            assert depth == self.depth

            p0 = (i0, j0)
            p1 = (i0, j1)
            p2 = (i1, j1)
            p3 = (i1, j0)

            split_edge = False
            
            # Check if there is a point on the edge 
            for edge in range(4):
                # Is there a point on the first edge?
                point_on_edge = (p0[0]+p1[0])//2, (p0[1]+p1[1])//2
                
                if (abs(p0[0] - p1[0]) > 1 or abs(p0[1] - p1[1]) > 1) and \
                        self.indices[point_on_edge[0], point_on_edge[1]] != -1:
                    # Yes there is a point.. we have to split the
                    # quad into three triangles
                    add_triangle(p0, point_on_edge, p3)
                    add_triangle(point_on_edge, p2, p3)
                    add_triangle(point_on_edge, p1, p2)
                    split_edge = True
                    break
                
                # Rotate the points so we check the next edge
                p0, p1, p2, p3 = p1, p2, p3, p0
             
            if not split_edge: 
                add_triangle(p0, p1, p2)
                add_triangle(p0, p2, p3)
         
        assert not (-1 in np.array(triangles))
        return triangles
            
    def __getitem__(self, *args, **kwargs):
        return self.indices.__getitem__(*args, **kwargs)
    
    def __setitem__(self, *args, **kwargs):
        self.indices.__setitem__(*args, **kwargs)


class PointStack:
    def __init__(self, surface, points=[]):
        self.path_length1 = surface.path_length1
        self.path_length2 = surface.path_length2
        
        self.surf = surface
         
        self.points = points
        self.indices = []
    
    def index_to_u(self, depth, i):
        return self.path_length1/(self.get_number_of_indices(depth) - 1) * i
     
    def index_to_v(self, depth, j):
        return self.path_length2/(self.get_number_of_indices(depth) - 1) * j
    
    def index_to_point(self, depth, i, j):
        u = self.index_to_u(depth, i)
        v = self.index_to_v(depth, j)
        return self.surf(u, v)
    
    def get_number_of_indices(self, depth):
        return 2**depth + 1
    
    def depth(self):
        return len(self.indices) - 1
    
    def add_level(self):
        new_depth = len(self.indices)
        Nu = Nv = self.get_number_of_indices(new_depth)
        
        index_map = np.full((Nu, Nv), -1, dtype=np.int64)
        
        if new_depth != 0.:
            index_map[::2, ::2] = self.indices[-1]
        
        self.indices.append(index_map)
     
    def to_point_index(self, depth, i, j):
        assert 0 <= i <= self.get_number_of_indices(depth)
        assert 0 <= j <= self.get_number_of_indices(depth)
        
        while depth >= len(self.indices):
            self.add_level()
        
        map_ = self.indices[depth]
        
        if map_[i, j] == -1:
            self.points.append(self.index_to_point(depth, i, j))
            map_[i, j] = len(self.points) - 1

        return map_[i, j]
     
    def __getitem__(self, args):
        depth, i, j = args
        return self.points[self.to_point_index(depth, i, j)]
     
    def normalize_to_depth(self, depth, quads, start_depth):
        N = self.get_number_of_indices(depth)
        
        while self.depth() < depth:
            self.add_level()
        
        assert self.depth() == depth
        assert self.indices[-1].shape == (N, N)

        for d in range(start_depth, len(self.indices)-1):
            previous = self.indices[d]
            x, y = np.where(previous != -1)
            self.indices[d+1][2*x, 2*y] = previous[x, y]

        quads = np.array(quads)
        assert quads.shape == (len(quads), 5)
        
        for i in range(len(quads)):
            quad_depth, i0, i1, j0, j1 = quads[i]
             
            while quad_depth < depth:
                i0 *= 2
                i1 *= 2
                j0 *= 2
                j1 *= 2
                quad_depth += 1
              
            quads[i] = (quad_depth, i0, i1, j0, j1)
         
        return PointsWithQuads(self.indices[-1], quads)


def _subdivide_quads(pstack, mesh_size, to_subdivide=[], quads=[]): 
    assert isinstance(pstack, PointStack)
     
    if not callable(mesh_size):
        mesh_size_fun = lambda x, y, z: mesh_size
    else:
        mesh_size_fun = mesh_size

    while len(to_subdivide) > 0:
        depth, i0, i1, j0, j1 = to_subdivide.pop()
        
        # Determine whether should split horizontally/vertically
        p1x, p1y, p1z = pstack[depth, i0, j0]
        p2x, p2y, p2z = pstack[depth, i0, j1]
        p3x, p3y, p3z = pstack[depth, i1, j0]
        p4x, p4y, p4z = pstack[depth, i1, j1]
            
        horizontal = max(sqrt((p1x-p2x)**2 + (p1y-p2y)**2 + (p1z-p2z)**2), sqrt((p3x-p4x)**2 + (p3y-p4y)**2 + (p3z-p4z)**2))
        vertical = max(sqrt((p1x-p3x)**2 + (p1y-p3y)**2 + (p1z-p3z)**2) , sqrt((p2x-p4x)**2 + (p2y-p4y)**2 + (p2z-p4z)**2))
    
        ms: float = mesh_size_fun((p1x+p2x+p3x+p4x)/4, (p1y+p2y+p3y+p4y)/4, (p1z+p2z+p3z+p4z)/4) # type: ignore
            
        h = horizontal > ms or (horizontal > 2.5*vertical and horizontal > 1/8*ms)
        v = vertical > ms or (vertical > 2.5*horizontal and vertical > 1/8*ms)
            
        if h and v: # Split both horizontally and vertically
            to_subdivide.append((depth+1, 2*i0, 2*i0+1, 2*j0, 2*j0+1))
            to_subdivide.append((depth+1, 2*i0, 2*i0+1, 2*j0+1, 2*j0+2))
            to_subdivide.append((depth+1, 2*i0+1, 2*i0+2, 2*j0, 2*j0+1))
            to_subdivide.append((depth+1, 2*i0+1, 2*i0+2, 2*j0+1, 2*j0+2))
        elif h and not v: # Split only horizontally
            to_subdivide.append((depth+1, 2*i0, 2*i1, 2*j0, 2*j0+1))
            to_subdivide.append((depth+1, 2*i0, 2*i1, 2*j0+1, 2*j0+2)) 
        elif v and not h: # Split only vertically
            to_subdivide.append((depth+1, 2*i0, 2*i0+1, 2*j0, 2*j1))
            to_subdivide.append((depth+1, 2*i0+1, 2*i0+2, 2*j0, 2*j1))
        else: # We are done, both sides are within mesh size limits
            quads.append((depth, i0, i1, j0, j1))

def _mesh_subsections_to_quads(surface, mesh_size, start_depth):
    all_pstacks = []
    all_quads = []
    points: list[np.ndarray] = []
    
    for s in surface._sections():
        quads: list[tuple[int, int, int, int, int]] = []
        pstack = PointStack(s, points=points)
        
        for i in range(pstack.get_number_of_indices(start_depth) - 1):
            for j in range(pstack.get_number_of_indices(start_depth) - 1):
                _subdivide_quads(pstack, mesh_size, to_subdivide=[(start_depth, i, i+1, j, j+1)], quads=quads)
        
        all_pstacks.append(pstack)
        all_quads.append(quads)
        points = pstack.points

    return points, all_pstacks, all_quads
    
def _copy_over_edge(e1, e2):
    assert e1.shape == e2.shape
    mask = e2 != -1
    e1[mask] = e2[mask]
    
    mask = e1 != -1
    e2[mask] = e1[mask]

def _mesh(surface, mesh_size, start_depth=2, name=None, ensure_outward_normals=True):
    # Create a point stack for each subsection
    points, point_stacks, quads = _mesh_subsections_to_quads(surface, mesh_size, start_depth)
     
    max_depth = max([p.depth() for p in point_stacks])
     
    # Normalize all the point stacks to the max depth of all sections 
    point_with_quads = [p.normalize_to_depth(max_depth, q, start_depth) for p, q in zip(point_stacks, quads)]
    
    # Copy over the edges
    Nx, Ny = len(surface.breakpoints1)+1, len(surface.breakpoints2)+1
    assert len(point_with_quads) == Nx*Ny

    for i in range(Nx-1):
        for j in range(Ny): # Horizontal copying
            _copy_over_edge(point_with_quads[j*Nx + i][-1, :], point_with_quads[j*Nx + i + 1][0, :])
     
    for i in range(Nx):
        for j in range(Ny-1): # Vertical copying
            _copy_over_edge(point_with_quads[j*Nx + i][:, -1], point_with_quads[(j+1)*Nx + i][:, 0])
     
    points = np.array(points)
    triangles = np.concatenate([pq.to_triangles() for pq in point_with_quads], axis=0)
    
    assert points.shape == (len(points), 3)
    assert triangles.shape == (len(triangles), 3)
    assert np.all( (0 <= triangles) & (triangles < len(points)) )
     
    if name is not None:
        physical_to_triangles = {name:np.arange(len(triangles))}
    else:
        physical_to_triangles = {}
    
    return Mesh(points=points, triangles=triangles, physical_to_triangles=physical_to_triangles, ensure_outward_normals=ensure_outward_normals)












