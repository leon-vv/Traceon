from math import *
import numpy as np
import time
from itertools import chain

import meshio

from .util import Saveable
from .backend import triangle_areas
from .logging import log_debug


__pdoc__ = {}
__pdoc__['PointsWithQuads'] = False
__pdoc__['PointStack'] = False
__pdoc__['__add__'] = True


class GeometricObject:
    """The Mesh class (and the classes defined in `traceon.geometry`) are subclasses
    of GeometricObject. This means that they all can be moved, rotated, mirrored."""
    
    def map_points(self, fun):
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
        pass
    
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
            physical_to_triangles={}):
        
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
    
    def _merge_dicts(dict1, dict2):
        dict_ = {}
        
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
            return Mesh(points=points, triangles=triangles, physical_to_triangles=physical_to_elements)
     
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
     
    def from_meshio(mesh):
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

        for i in range(start_depth, len(self.indices)-1):
            self.indices[i+1][::2, ::2] = self.indices[i]

        quads = np.array(quads)
        assert quads.shape == (len(quads), 5)
        
        for i in range(len(quads)):
            quad_depth, i0, i1, j0, j1 = quads[i]
            assert quad_depth <= depth
            assert self.indices[quad_depth][i0, j0] != -1
             
            while quad_depth < depth:
                i0 *= 2
                i1 *= 2
                j0 *= 2
                j1 *= 2
                quad_depth += 1
              
            quads[i] = (quad_depth, i0, i1, j0, j1)
            assert self.indices[-1][i0, j0] != -1
            assert quad_depth == depth
         
        return PointsWithQuads(self.indices[-1], quads)

    

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
    
        ms = mesh_size_fun((p1x+p2x+p3x+p4x)/4, (p1y+p2y+p3y+p4y)/4, (p1z+p2z+p3z+p4z)/4)
            
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
    points = []
    
    for s in surface.sections():
        quads = []
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

def _mesh(surface, mesh_size, start_depth=2, name=None):
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
    
    return Mesh(points=points, triangles=triangles, physical_to_triangles=physical_to_triangles)












