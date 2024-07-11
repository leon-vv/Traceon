"""The geometry module allows the creation of general geometries in 2D and 3D and generate
the resulting meshes. The heavy lifting is done by the powerful [GMSH](https://gmsh.info/) library, and we access this library
through the convenient [pygmsh](https://github.com/nschloe/pygmsh) library.

The GMSH library has the concept of _physical groups_. These are simply elements inside your geometry which
are assigned a given name. When using Traceon, usually every electrode gets assigned its
own name (or physical group) with which it can be referenced when later specifying the excitation
of this electrode.
"""
from math import pi, sqrt, sin, cos, atan2, ceil

import numpy as np
from scipy.integrate import cumulative_simpson, quad
from scipy.interpolate import CubicSpline

from .mesher import GeometricObject, mesh

__pdoc__ = {}
__pdoc__['discteize_path'] = False


def _points_close(p1, p2, tolerance=1e-8):
    return np.allclose(p1, p2, atol=tolerance)

def discretize_path(path_length, breakpoints, mesh_size, N_factor=1):
    # Return the arguments to use to breakup the path
    # in a 'nice' way
    
    # Points that have to be in, in any case
    points = [0.] + breakpoints +  [path_length]

    subdivision = []
        
    for (u0, u1) in zip(points, points[1:]):
        if u0 == u1:
            continue
         
        N = max( ceil((u1-u0)/mesh_size), 3)
        # When using higher order, we splice extra points
        # between two points in the descretization. This
        # ensures that the number of elements stays the same
        # as in the non-higher order case.
        # N_factor = 1  normal flat elements
        # N_factor = 2  extra points for curved triangles (triangle6 in GMSH terminology)
        # N_factor = 3  extra points for curved line elements (line4 in GMSH terminology)
        subdivision.append(np.linspace(u0, u1, N_factor*N, endpoint=False))
    
    subdivision.append( [path_length] )
    
    return np.concatenate(subdivision)


class Path(GeometricObject):
    
    def __init__(self, fun, path_length, breakpoints=[], name=None):
        # Assumption: fun takes in p, the path length
        # and returns the point on the path
        self.fun = fun
        self.path_length = path_length
        assert self.path_length > 0
        self.breakpoints = breakpoints
        self.name = name
    
    def from_irregular_function(to_point, N=100, breakpoints=[]):
        # Construct a path from a function that
        # is of the form u -> point, where 0 <= u <= 1.
        # We need to regularize it such that it correctly accepts
        # the path length and returns a point on the path.
        #
        # To regularize we integrate the norm of the derivative.
        # path length = integrate |f'(x)|
        fun = lambda u: np.array(to_point(u))
        
        def derivative(u, tol=1e-4):
            assert 0 <= u <= 1
             
            if u - tol < 0.: # Use simply finite difference formula to approx. the derivative
                f1, f2, f3 = fun(u), fun(u+tol), fun(u+2*tol)
                assert isinstance(f1, np.ndarray), "Function should return a (3,) np.ndarray"
                return (-3/2*f1 + 2*f2 - 1/2*f3)/tol
            elif u + tol > 1:
                f1, f2, f3 = fun(u-2*tol), fun(u-tol), fun(u)
                assert isinstance(f1, np.ndarray), "Function should return a (3,) np.ndarray"
                return (1/2*f1 -2*f2 + 3/2*f3)/tol
            else:
                f1, f2 = fun(u-tol), fun(u+tol)
                assert isinstance(f1, np.ndarray), "Function should return a (3,) np.ndarray"
                return (-1/2*f1 + 1/2*f2)/tol
            
        u = np.linspace(0, 1, N)
        samples = [np.linalg.norm(derivative(u_)) for u_ in u]
        cum_sum = cumulative_simpson(samples, dx=u[1]-u[0], initial=0.)
        path_length = cum_sum[-1]
        interpolation = CubicSpline(cum_sum, u) # Path length to u

        return Path(lambda pl: fun(interpolation(pl)), path_length, breakpoints=breakpoints)
     
    def average(self, fun):
        return quad(lambda s: fun(self(s)), 0, self.path_length, points=self.breakpoints)[0]/self.path_length
     
    def map_points(self, fun):
        return Path(lambda u: fun(self(u)), self.path_length, self.breakpoints, name=self.name)
     
    def __call__(self, t):
        return self.fun(t)
     
    def is_closed(self):
        return _points_close(self.starting_point(), self.final_point())
    
    def add_phase(self, l):
        assert self.is_closed()
        
        def fun(u):
            return self( (l + u) % self.path_length )
        
        return Path(fun, self.path_length, sorted([(b-l)%self.path_length for b in self.breakpoints + [0.]]), name=self.name)
     
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
        
        return Path(f, total, self.breakpoints + [self.path_length] + other.breakpoints, name=self.name)

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
     
    def circle(radius, angle=2*pi):
        def f(u):
            theta = u / radius 
            return np.array([radius*cos(theta), radius*sin(theta), 0.])
        
        return Path(f, angle*radius)
    
    def arc_to(self, center, end, reverse=False):
        start = self.final_point()
        return self >> Path.arc(center, start, end, reverse=reverse)
    
    def arc(center, start, end, reverse=False):
        start, center, end = np.array(start), np.array(center), np.array(end)
         
        x_unit = start - center
        x_unit /= np.linalg.norm(x_unit)

        vector = end - center
         
        y_unit = vector - np.dot(vector, x_unit) * x_unit
        y_unit /= np.linalg.norm(y_unit)

        radius = np.linalg.norm(start - center) 
        theta_max = atan2(np.dot(vector, y_unit), np.dot(vector, x_unit))

        if reverse:
            theta_max = theta_max - 2*pi

        path_length = abs(theta_max * radius)
          
        def f(l):
            theta = l/path_length * theta_max
            return center + radius*cos(theta)*x_unit + radius*sin(theta)*y_unit
        
        return Path(f, path_length)
     
    def revolve_x(self, angle=2*pi):
        pstart, pmiddle, pfinal = self.starting_point(), self.middle_point(), self.final_point()
        r_avg = self.average(lambda p: sqrt(p[1]**2 + p[2]**2))
        length2 = 2*pi*r_avg
         
        def f(u, v):
            p = self(u)
            theta = atan2(p[2], p[1])
            r = sqrt(p[1]**2 + p[2]**2)
            return np.array([p[0], r*cos(theta + v/length2*angle), r*sin(theta + v/length2*angle)])
         
        return Surface(f, self.path_length, length2, self.breakpoints, name=self.name)
    
    def revolve_y(self, angle=2*pi):
        pstart, pfinal = self.starting_point(), self.final_point()
        r_avg = self.average(lambda p: sqrt(p[0]**2 + p[2]**2))
        length2 = 2*pi*r_avg
         
        def f(u, v):
            p = self(u)
            theta = atan2(p[2], p[0])
            r = sqrt(p[0]*p[0] + p[2]*p[2])
            return np.array([r*cos(theta + v/length2*angle), p[1], r*sin(theta + v/length2*angle)])
         
        return Surface(f, self.path_length, length2, self.breakpoints, name=self.name)
    
    def revolve_z(self, angle=2*pi):
        pstart, pfinal = self.starting_point(), self.final_point()
        r_avg = self.average(lambda p: sqrt(p[0]**2 + p[1]**2))
        length2 = 2*pi*r_avg
        
        def f(u, v):
            p = self(u)
            theta = atan2(p[1], p[0])
            r = sqrt(p[0]*p[0] + p[1]*p[1])
            return np.array([r*cos(theta + v/length2*angle), r*sin(theta + v/length2*angle), p[2]])
        
        return Surface(f, self.path_length, length2, self.breakpoints, name=self.name)
     
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
    
        
    def line(from_, to):
        from_, to = np.array(from_), np.array(to)
        length = np.linalg.norm(from_ - to)
        return Path(lambda pl: (1-pl/length)*from_ + pl/length*to, length)

    def cut(self, length):
        return Path(self.fun, length, [b for b in self.breakpoints if b <= length], name=self.name)

    def rectangle_xz(xmin, xmax, zmin, zmax):
        return Path.line([xmin, 0., zmin], [xmin, 0, zmax]) \
            .line_to([xmax, 0, zmax]).line_to([xmax, 0., zmin]).close()
     
    def rectangle_yz(ymin, ymax, zmin, zmax):
        return Path.line([0., ymin, zmin], [0, ymax, zmin]) \
            .line_to([0., ymax, zmax]).line_to([0., ymin, zmax]).close()
     
    def rectangle_xy(xmin, xmax, ymin, ymax):
        return Path.line([xmin, ymin, 0.], [xmax, ymin, 0.]) \
            .line_to([xmax, ymax, 0.]).line_to([xmin, ymax, 0.]).close()
    
    def aperture(height, radius, extent, z=0.):
        return Path.line([extent, 0., -height/2], [radius, 0., -height/2])\
                .line_to([radius, 0., height/2]).line_to([extent, 0., height/2]).move(dz=z)
    
    def __add__(self, other):
        if not isinstance(other, Path) and not isinstance(other, PathCollection):
            return NotImplemented
        
        if isinstance(other, Path):
            return PathCollection([self, other])
        elif isinstance(other, PathCollection):
            return PathCollection([self] + [other.paths])
     
    def mesh(self, mesh_size=None, higher_order=False):
        
        if mesh_size is None:
            mesh_size = self.path_length/10
        
        u = discretize_path(self.path_length, self.breakpoints, mesh_size, N_factor=3 if higher_order else 1)
        
        N = len(u) 
        points = np.zeros( (N, 3) )
         
        for i in range(N):
            points[i] = self(u[i])
         
        if not higher_order:
            lines = np.array([np.arange(N-1), np.arange(1, N)]).T
        else:
            assert N % 3 == 1
            r = np.arange(N)
            p0 = r[0:-1:3]
            p1 = r[3::3]
            p2 = r[1::3]
            p3 = r[2::3]
            lines = np.array([p0, p1, p2, p3]).T
          
        assert lines.dtype == np.int64
         
        if self.name is not None:
            physical_to_lines = {self.name:np.arange(len(lines))}
        else:
            physical_to_lines = {}
        
        return Mesh(points=points, lines=lines, physical_to_lines=physical_to_lines)


class PathCollection(GeometricObject):
    
    def __init__(self, paths):
        assert all([isinstance(p, Path) for p in paths])
        self.paths = paths
    
    def map_points(self, fun):
        return PathCollection([p.map_points(fun) for p in self.paths])
     
    def mesh(self, mesh_size=None, higher_order=False):
        mesh = Mesh()
        
        for p in self.paths:
            mesh = mesh + p.mesh(mesh_size=mesh_size, higher_order=higher_order)

        return mesh

    def _map_to_surfaces(self, f, *args, **kwargs):
        surfaces = []

        for p in self.paths:
            surfaces.append(f(p, *args, **kwargs))

        return SurfaceCollection(surfaces)
    
    def __add__(self, other):
        if not isinstance(other, Path) and not isinstance(other, PathCollection):
            return NotImplemented
        
        if isinstance(other, Path):
            return PathCollection(self.paths+[other])
        else:
            return PathCollection(self.paths+other.paths)
      
    def __iadd__(self, other):
        assert isinstance(other, PathCollection) or isinstance(other, Path)

        if isinstance(other, Path):
            self.paths.append(other)
        else:
            self.paths.extend(other.paths)
       
    def revolve_x(self, angle=2*pi):
        return self._map_to_surfaces(Path.revolve_x, angle=angle)
    def revolve_y(self, angle=2*pi):
        return self._map_to_surfaces(Path.revolve_y, angle=angle)
    def revolve_z(self, angle=2*pi):
        return self._map_to_surfaces(Path.revolve_z, angle=angle)
    def extrude(self, vector):
        return self._map_to_surface(Path.extrude, vector)
    def extrude_by_path(self, p2):
        return self._map_to_surface(Path.extrude_by_path, p2)
     


class Surface(GeometricObject):
    def __init__(self, fun, path_length1, path_length2, breakpoints1=[], breakpoints2=[], name=None):
        self.fun = fun
        self.path_length1 = path_length1
        self.path_length2 = path_length2
        assert self.path_length1 > 0 and self.path_length2 > 0
        self.breakpoints1 = breakpoints1
        self.breakpoints2 = breakpoints2
        self.name = name

    def sections(self): 
        # Iterate over the sections defined by
        # the breakpoints
        b1 = [0.] + self.breakpoints1 + [self.path_length1]
        b2 = [0.] + self.breakpoints2 + [self.path_length2]

        for u0, u1 in zip(b1[:-1], b1[1:]):
            for v0, v1 in zip(b2[:-1], b2[1:]):
                def fun(u, v, u0_=u0, v0_=v0):
                    return self(u0_+u, v0_+v)
                yield Surface(fun, u1-u0, v1-v0, [], [])
       
    def __call__(self, u, v):
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

    def from_boundary_paths(p1, p2, p3, p4):
        path_length_p1_and_p3 = (p1.path_length + p3.path_length)/2
        path_length_p2_and_p4 = (p2.path_length + p4.path_length)/2

        def f(u, v):
            u /= path_length_p1_and_p3
            v /= path_length_p2_and_p4
            
            a = (1-v)
            b = (1-u)
             
            c = v
            d = u
            
            return 1/2*(a*p1(u*p1.path_length) + \
                        b*p4((1-v)*p4.path_length) + \
                        c*p3((1-u)*p3.path_length) + \
                        d*p2(v*p2.path_length))
        
        # Scale the breakpoints appropriately
        b1 = sorted([b/p1.path_length * path_length_p1_and_p3 for b in p1.breakpoints] + \
                [b/p3.path_length * path_length_p1_and_p3 for b in p3.breakpoints])
        b2 = sorted([b/p2.path_length * path_length_p2_and_p4 for b in p2.breakpoints] + \
                [b/p4.path_length * path_length_p2_and_p4 for b in p4.breakpoints])
        
        return Surface(f, path_length_p1_and_p3, path_length_p2_and_p4, b1, b2)
     
    def aperture(height, radius, extent, z=0.):
        return Path.aperture(height, radius, extent, z=z).revolve_z()
     
    def __add__(self, other):
        if not isinstance(other, Surface) and not isinstance(other, SurfaceCollection):
            return NotImplemented

        if isinstance(other, Surface):
            return SurfaceCollection([self, other])
        else:
            return SurfaceCollection([self] + other.surfaces)
     
    def mesh(self, mesh_size=None):
          
        if mesh_size is None:
            mesh_size = min(self.path_length1, self.path_length2)/10
         
        return mesh(self, mesh_size, name=self.name)



class SurfaceCollection(GeometricObject):
     
    def __init__(self, surfaces):
        assert all([isinstance(s, Surface) for s in surfaces])
        self.surfaces = surfaces
     
    def map_points(self, fun):
        return SurfaceCollection([s.map_points(fun) for s in self.surfaces])
     
    def mesh(self, mesh_size=None, name=None):
        mesh = Mesh()
        
        for s in self.surfaces:
            mesh = mesh + s.mesh(mesh_size=mesh_size)
         
        return mesh
     
    def __add__(self, other):
        if not isinstance(other, Surface) and not isinstance(other, SurfaceCollection):
            return NotImplemented
              
        if isinstance(other, Surface):
            return SurfaceCollection(self.surfaces+[other])
        else:
            return SurfaceCollection(self.surfaces+other.surfaces)
     
    def __iadd__(self, other):
        assert isinstance(other, SurfaceCollection) or isinstance(other, Surface)
        
        if isinstance(other, Surface):
            self.surfaces.append(other)
        else:
            self.surfaces.extend(other.surfaces)
    







