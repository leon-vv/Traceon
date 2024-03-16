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
from itertools import chain

import numpy as np
from pygmsh import *
import gmsh
import meshio

from .util import Saveable
from .backend import N_QUAD_2D, position_and_jacobian_radial, position_and_jacobian_3d, normal_3d

def _points_close(p1, p2, tolerance=1e-8):
    return np.allclose(p1, p2, atol=Path._tolerance)

class GeometricObject:
    def map_points(self, fun):
        pass
    
    def move(self, dx=0., dy=0., dz=0.):
        assert isinstance(dx, float) or isinstance(dx, int)
        return self.map_points(lambda p: p + np.array([dx, dy, dz]))
     
    def rotate(self, Rx=0., Ry=0., Rz=0.):
        assert sum([Rx==0., Ry==0., Rz==0.]) >= 2, "Only supply one axis of rotation"
         
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

        return self.map_points(lambda p: matrix @ p)

    def rotate_around_point(self, origin, dphi=0., dtheta=0.):
        origin = np.array(origin)
        
        def fun(p):
            vec = p - origin
            r = sqrt(vec[0]**2 + vec[1]**2)
            phi0 = atan2(vec[1], vec[0])
            theta0 = atan2(vec[2], r)
            return origin + np.array([
                r*sin(theta0+dtheta)*cos(phi0+dphi),
                r*sin(theta0+dtheta)*sin(phi0+dphi),
                r*cos(theta0+dtheta)])

        return self.map_points(fun)
    
    def mirror_xz(self):
        return self.map_points(lambda p: np.array([p[0], -p[1], p[2]]))
     
    def mirror_yz(self):
        return self.map_points(lambda p: np.array([-p[0], -p[1], p[2]]))
    
    def mirror_xy(self):
        return self.map_points(lambda p: np.array([p[0], p[1], -p[2]]))
     

class Path(GeometricObject):
    
    def __init__(self, fun, path_length, breakpoints=[]):
        # Assumption: fun takes in p, the path length
        # and returns the point on the path
        self.fun = fun
        self.path_length = path_length
        self.breakpoints = breakpoints
    
    def from_irregular_function(fun, N=100, breakpoints=[]):
        # Construct a path from a function that
        # is of the form u -> point, where 0 <= u <= 1.
        # We need to regularize it such that it correctly accepts
        # the path length and returns a point on the path.
        #
        # To regularize we integrate the norm of the derivative.
        # path length = integrate |f'(x)|
        def derivative(u, tol=1e-4):
            assert 0 <= u <= 1
             
            if u - tol < 0.: # Use simply finite difference formula to approx. the derivative
                f1, f2, f3 = fun(u), fun(u+tol), fun(u+2*tol)
                return (-3/2*f1 + 2*f2 - 1/2*f3)/tol
            elif u + tol > 1:
                f1, f2, f3 = fun(u-2*tol), fun(u-tol), fun(u)
                return (1/2*f1 -2*f2 + 3/2*f3)/tol
            else:
                return (-1/2*fun(u-tol) + 1/2*fun(u+tol))/tol
            
        u = np.linspace(0, 1, N)
        samples = [np.linalg.norm(derivative(u_)) for u_ in u]
        cum_sum = cumulative_simpson(samples, dx=u[1]-u[0], initial=0.)
        path_length = cum_sum[-1]
        interpolation = CubicSpline(cum_sum, u) # Path length to u

        return Path(lambda pl: fun(interpolation(pl)), path_length, breakpoints=breakpoints)
    
    def map_points(self, fun):
        return Path(lambda u: fun(self(u)), self.path_length, self.breakpoints)
     
    def __call__(self, t):
        assert 0 <= t <= self.path_length
        return self.fun(t)
     
    def is_closed(self):
        return _points_close(self.starting_point(), self.final_point())
    
    def add_phase(self, l):
        assert self.is_closed()
        
        def fun(u):
            return self( (l + u) % self.path_length )
        
        return Path(fun, self.path_length, sorted([(b-l)%self.path_length for b in self.breakpoints + [0.]]))
     
    def __rshift__(self, other):
        assert isinstance(other, Path), "Exteding path with object that is not actually a Path"

        assert _points_close(self.final_point(), other.starting_point())

        total = self.path_length + other.path_length
         
        def f(t):
            assert 0 <= t <= total
            
            if t <= self.path_length:
                return self(t)
            else:
                return other(t - self.path_length)
        
        return Path(f, total, self.breakpoints + [self.path_length] + other.breakpoints)

    def starting_point(self):
        return self(0.)
    def middle_point(self):
        return self(self.path_length/2)
    def final_point(self):
        return self(self.path_length)
    
    def line_to(self, point):
        point = np.array(point)
        l = Path.line(self.final_point(), point)
        return self >> l
     
    def revolve_x(self, angle=2*pi):
        pstart, pmiddle, pfinal = self.starting_point(), self.middle_point(), self.final_point()
        rstart = sqrt(pstart[1]**2 + pstart[2]**2)
        rmiddle = sqrt(pmiddle[1]**2 + pmiddle[2]**2)
        rfinal = sqrt(pfinal[1]**2 + pfinal[2]**2)
        
        length2 = 2*pi*max([rstart, rmiddle, rfinal])
         
        def f(u, v):
            p = self(u)
            theta = atan2(p[2], p[1])
            r = sqrt(p[1]**2 + p[2]**2)
            return np.array([p[0], r*cos(theta + v/length2*angle), r*sin(theta + v/length2*angle)])
         
        return Surface(f, self.path_length, length2, self.breakpoints)
    
    def revolve_y(self, angle=2*pi):
        pstart, pfinal = self.starting_point(), self.final_point()
        rstart = sqrt(pstart[0]**2 + pstart[2]**2)
        rfinal = sqrt(pfinal[0]**2 + pfinal[2]**2)
        
        length2 = 2*pi*max(rstart, rfinal)
         
        def f(u, v):
            p = self(u)
            theta = atan2(p[2], p[0])
            r = sqrt(p[0]*p[0] + p[2]*p[2])
            return np.array([r*cos(theta + v/length2*angle), p[1], r*sin(theta + v/length2*angle)])
         
        return Surface(f, self.path_length, length2, self.breakpoints)
    
    def revolve_z(self, angle=2*pi):
        pstart, pfinal = self.starting_point(), self.final_point()
        rstart = sqrt(pstart[0]**2 + pstart[1]**2)
        rfinal = sqrt(pfinal[0]**2 + pfinal[1]**2)

        length2 = 2*pi*max(rstart, rfinal)
        
        def f(u, v):
            p = self(u)
            theta = atan2(p[1], p[0])
            r = sqrt(p[0]*p[0] + p[1]*p[1])
            return np.array([r*cos(theta + v/length2*angle), r*sin(theta + v/length2*angle), p[2]])
        
        return Surface(f, self.path_length, length2, self.breakpoints)
     
    def extrude(self, vector):
        vector = np.array(vector)
        length = np.linalg.norm(vector)
         
        def f(u, v):
            return self(u) + v/length*vector
        
        return Surface(f, self.path_length, length, self.breakpoints)
    
    def extrude_by_path(self, p2):
        p0 = p2.starting_point()
         
        def f(u, v):
            return self(u) + p2(v) - p0

        return Surface(f, self.path_length, p2.path_length, self.breakpoints, p2.breakpoints)

        
    def close(self):
        return self.line_to(self.starting_point())
    
    def ellipse(major, minor):
        # Crazy enough there is no closed formula
        # to go from path length to a point on the ellipse.
        # So we have to use `from_irregular_function`
        def f(u):
            return np.array([major*cos(2*pi*u), minor*sin(2*pi*u), 0.])
        return Path.from_irregular_function(f)
    
    def circle(radius):
        def f(u):
            theta = u / radius 
            return np.array([radius*cos(theta), radius*sin(theta), 0.])
        
        return Path(f, 2*pi*radius)
    
    def line(from_, to):
        from_, to = np.array(from_), np.array(to)
        length = np.linalg.norm(from_ - to)
        return Path(lambda pl: (1-pl/length)*from_ + pl/length*to, length)

    def cut(self, length):
        return Path(self.fun, length, [b for b in self.breakpoints if b <= length])

    def rectangle_xz(xmin, xmax, zmin, zmax):
        return Path.line([xmin, 0., zmin], [xmin, 0, zmax]) \
            .line_to([xmax, 0, zmax]).line_to([xmax, 0., zmin]).close()
     
    def rectangle_yz(ymin, ymax, zmin, zmax):
        return Path.line([0., ymin, zmin], [0, ymax, zmin]) \
            .line_to([0., ymax, zmax]).line_to([0., ymin, zmax]).close()
     
    def rectangle_xy(xmin, xmax, ymin, ymax):
        return Path.line([xmin, ymin, 0.], [xmax, ymin, 0.]) \
            .line_to([xmax, ymax, 0.]).line_to([xmin, ymax, 0.]).close()


class Surface(GeometricObject):
    def __init__(self, fun, path_length1, path_length2, breakpoints1=[], breakpoints2=[]):
        self.fun = fun
        self.path_length1 = path_length1
        self.path_length2 = path_length2
        self.breakpoints1 = breakpoints1
        self.breakpoints2 = breakpoints2
     
    def __call__(self, u, v):
        assert 0 <= u <= self.path_length1
        assert 0 <= v <= self.path_length2
        return self.fun(u, v)

    def map_points(self, fun):
        return Surface(lambda u, v: fun(self(u, v)),
            self.path_length1, self.path_length2,
            self.breakpoints1, self.breakpoints2)
     
    def spanned_by_paths(path1, path2):
        length1 = max(path1.path_length, path2.path_length)
        
        length_start = np.linalg.norm(path1.starting_point() - path2.starting_point())
        length_final = np.linalg.norm(path1.final_point() - path2.final_point())
        length2 = (length_start + length_final)/2
         
        def f(u, v):
            p1 = path1(u/length1*path1.path_length) # u/l*p = b, u = l*b/p
            p2 = path2(u/length1*path2.path_length)
            return (1-v/length2)*p1 + v/length2*p2

        breakpoints = sorted([length1*b/path1.path_length for b in path1.breakpoints] + \
                                [length1*b/path2.path_length for b in path2.breakpoints])
         
        return Surface(f, length1, length2, breakpoints)

    def sphere(radius):
        
        length1 = 2*pi*radius
        length2 = pi*radius
         
        def f(u, v):
            phi = u/radius
            theta = v/radius
            
            return np.array([
                radius*sin(theta)*cos(phi),
                radius*sin(theta)*sin(phi),
                radius*cos(theta)]) 
        
        return Surface(f, length1, length2)

    
    def _from_boundary_paths(path1, path2, path3, path4):
        pl1, pl2, pl3, pl4 = path1.path_length, path2.path_length, \
            path3.path_length, path4.path_length
         
        length1 = (pl1+pl3)/2
        length2 = (pl2+pl4)/2
        
        def f(u, v):
            k = u/length1
            l = v/length2
            p1 = path1(k*pl1)
            p2 = path2(l*pl2)
            p3 = path3((1-k)*pl3)
            p4 = path4((1-l)*pl4)
             
            sum_ = (k**2 - k)*( (1-l) + l) + (l**2-l)*((1-k) + k)
            return ( (k**2 - k)*((1-l)*p1 + l*p3) + (l**2-l)*((1-k)*p2 + k*p4) )/sum_
            
            return (1-k)*p2 + k*p4 #+ (1-l)*p1 + l*p3
        
        breakpoints1 = path1.breakpoints + path3.breakpoints
        breakpoints2 = path2.breakpoints + path4.breakpoints
         
        return Surface(f, length1, length2, sorted(breakpoints1), sorted(breakpoints2))
     
    def _discretize_path(path_length, breakpoints, mesh_size):
        # Return the arguments to use to breakup the path
        # in a 'nice' way
        
        # Points that have to be in, in any case
        points = [0.] + breakpoints +  [path_length]

        subdivision = []
         
        for (u0, u1) in zip(points, points[1:]):
            if u0 == u1:
                continue
                  
            N = max( ceil((u1-u0)/mesh_size), 2)
            subdivision.append(np.linspace(u0, u1, N, endpoint=False))
          
        subdivision.append( [path_length] )
         
        return np.concatenate(subdivision)
    
    def mesh(self, mesh_size=None, name=None):
        
        if mesh_size is None:
            mesh_size = min(self.path_length1, self.path_length2)/10
        
        u = Surface._discretize_path(self.path_length1, self.breakpoints1, mesh_size)
        v = Surface._discretize_path(self.path_length2, self.breakpoints2, mesh_size)
        
        Nu, Nv = len(u), len(v)
        
        points = np.zeros( (Nu, Nv, 3) )
         
        for i in range(Nu):
            for j in range(Nv):
                points[i, j] = self(u[i], v[j])
         
        ru, rv = np.arange(Nu), np.arange(Nv)
        a = (ru[:-1], rv[:-1])
        b = (ru[1:], rv[:-1])
        c = (ru[1:], rv[1:])
        to_linear = lambda i, j: [i_*Nv + j_ for i_ in i for j_ in j]
        lower_indices = np.array([to_linear(*a), to_linear(*b), to_linear(*c)]).T
        assert np.all(lower_indices < Nu*Nv)
        
        # Upper triangles
        a = (ru[:-1], rv[:-1])
        b = (ru[:-1], rv[1:])
        c = (ru[1:], rv[1:])
        upper_indices = np.array([to_linear(*a), to_linear(*b), to_linear(*c)]).T
        assert np.all(upper_indices < Nu*Nv)
        
        points = np.reshape(points, (Nu*Nv, 3))
        triangles = np.concatenate( (lower_indices, upper_indices), axis=0)

        assert triangles.dtype == np.int64
        
        if name is not None:
            physical_to_triangles = {name:np.arange(len(triangles))}
        else:
            physical_to_triangles = {}

        return G.Mesh(G.Symmetry.THREE_D, points=points, triangles=triangles, physical_to_triangles=physical_to_triangles)


def aperture(height, radius, extent, name=None, mesh_size=None):
    l = Path.line([extent, 0., -height/2], [radius, 0., -height/2])\
            .line_to([radius, 0., height/2]).line_to([extent, 0., height/2])
    return l.revolve_z().mesh(name=name, mesh_size=mesh_size)




def _concat_arrays(arr1, arr2):
    if not len(arr1):
        return np.copy(arr2)
    if not len(arr2):
        return np.copy(arr1)
      
    assert arr1.shape[1:] == arr2.shape[1:], "Cannot add meshes if one is higher order and the other is not"
    
    return np.concatenate( (arr1, arr2), axis=0)

class Mesh(Saveable, GeometricObject):
    """Class containing a mesh.
    For now, to make things manageable only lines and triangles are supported.
    Lines and triangles can be higher order (curved) or not. But a mesh cannot contain
    both curved and simple elements at the same time.
    
    When the elements are higher order (curved), triangles consists of 6 points and lines of four points.
    These correspond with the GMSH line4 and triangle6 types."""
     
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
            self.lines = np.array(lines, dtype=np.uint64)
        else:
            self.lines = np.empty((0,2), dtype=np.uint64)
    
        if len(triangles):
            self.triangles = np.array(triangles, dtype=np.uint64)
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
    
    def map_points(self, fun):
        new_points = np.vectorize(fun)(self.points)
        return Mesh(self.symmetry, new_points, self.lines, self.triangles, self.physical_to_lines, self.physical_to_triangles)
    
    def _merge_dicts(dict1, dict2):
        dict_ = {}
        
        for (k, v) in chain(dict1.items(), dict2.items()):
            if k in dict_:
                dict_[k] = np.concatenate( (dict_[k], v), axis=0)
            else:
                dict_[k] = v

        return dict_
     
    def __add__(self, other):
        assert isinstance(other, Mesh)
        assert self.symmetry == other.symmetry, "Cannot add meshes with different symmetries"
         
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
        elif 'triangle6' in mesh.cells_dict:
            triangles, physical_triangles = extract('triangle6')
        
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


    def _triangles_to_higher_order(points, elements):
        N_elements = len(elements)
        N_points = len(points)
         
        v0, v1, v2 = elements.T
        p3 = (points[v0] + points[v1])/2
        p4 = (points[v1] + points[v2])/2
        p5 = (points[v2] + points[v0])/2
         
        assert all(p.shape == (N_elements, points.shape[1]) for p in [p3,p4,p5])
          
        points = np.concatenate( (points, p3, p4, p5), axis=0)
          
        elements = np.array([
            elements[:, 0], elements[:, 1], elements[:, 2],
            np.arange(N_points, N_points + N_elements, dtype=np.uint64),
            np.arange(N_points + N_elements, N_points + 2*N_elements, dtype=np.uint64),
            np.arange(N_points + 2*N_elements, N_points + 3*N_elements, dtype=np.uint64)]).T
         
        assert np.allclose(p3, points[elements[:, 3]])
        assert np.allclose(p4, points[elements[:, 4]])
        assert np.allclose(p5, points[elements[:, 5]])
        
        return points, elements

    def _to_higher_order_mesh(self):
        # The matrix solver currently only works with higher order meshes.
        # We can however convert a simple mesh easily to a higher order mesh, and solve that.
        
        points, lines, triangles = self.points, self.lines, self.triangles

        if len(lines) and lines.shape[1] == 2:
            points, lines = Mesh._lines_to_higher_order(points, lines)
        if len(triangles) and triangles.shape[1] == 3:
            points, triangles = Mesh._triangles_to_higher_order(points, triangles) 
         
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
 
    
        

