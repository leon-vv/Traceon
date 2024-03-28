import numpy as np
import time

import traceon.geometry as G
import traceon.plotting as P


class PointStack:
    def __init__(self, surface):
        self.path_length1 = surface.path_length1
        self.path_length2 = surface.path_length2
        
        self.surf = surf
         
        self.points = []
        self.indices = []
    
    def index_to_u(self, depth, i):
        return self.path_length1/(self.get_number_of_indices(depth) - 1) * i
     
    def index_to_v(self, depth, j):
        return self.path_length2/(self.get_number_of_indices(depth) - 1) * j
    
    def index_to_point(self, depth, i, j):
        u = self.index_to_u(depth, i)
        v = self.index_to_v(depth, j)
        return surf(u, v)
    
    def get_number_of_indices(self, depth):
        return 2**depth + 1
    
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

    def take_points_subset(self, indices):
        indices = np.array(indices, dtype=np.int64)
        
        assert np.all( (0 <= indices) & (indices < len(self.points)) )
        assert len(indices) and indices.dtype == np.int64, indices.dtype
        
        indices_flat = np.ndarray.flatten(indices)
        
        inactive = np.full(len(self.points), True, dtype=bool)
        inactive[indices_flat] = False
        
        map_index = np.arange(len(self.points)) - np.cumsum(inactive)
        
        new_points = np.array(self.points)[~inactive]
        new_indices = map_index[indices]
        
        assert np.all( (0 <= new_indices) & (new_indices < len(new_points)) )

        return new_points, new_indices

class Mesher:
    def __init__(self, surface, mesh_size, start_depth=3):
        self.mesh_size = mesh_size
        self.pstack = PointStack(surface)
        self.start_depth = start_depth

    def mesh_to_quad_indices(self):
        quads = []
        
        for i in range(self.pstack.get_number_of_indices(self.start_depth) - 1):
            for j in range(self.pstack.get_number_of_indices(self.start_depth) - 1):
                self.subdivide_quad(self.start_depth, i, i+1, j, j+1, quads=quads)

        return quads
    
    def mesh_to_triangles(self):
        triangles = []
        
        for quad in self.mesh_to_quad_indices():
            idx0, idx1, idx2, idx3 = quad 
            
            triangles.append(
                [self.pstack.to_point_index(*idx0),
                self.pstack.to_point_index(*idx1),
                self.pstack.to_point_index(*idx3)])
            
            triangles.append(
                [self.pstack.to_point_index(*idx0),
                self.pstack.to_point_index(*idx3),
                self.pstack.to_point_index(*idx2)])
        
        return self.pstack.take_points_subset(triangles)
     
    def should_split(self, depth, i0, i1, j0, j1):
        p1 = self.pstack[depth, i0, j0]
        p2 = self.pstack[depth, i0, j1]
        p3 = self.pstack[depth, i1, j0]
        p4 = self.pstack[depth, i1, j1]

        horizontal = max(np.linalg.norm(p1-p2), np.linalg.norm(p3-p4))
        vertical = max(np.linalg.norm(p1-p3), np.linalg.norm(p2-p4))
        
        return (horizontal > self.mesh_size) or horizontal > 3*vertical, vertical > self.mesh_size or vertical > 3*horizontal
     
    def subdivide_quad(self, depth, i0, i1, j0, j1, quads=[]): 
        h, v = self.should_split(depth, i0, i1, j0, j1)
        
        if h and v: # Split both horizontally and vertically
            self.subdivide_quad(depth+1, 2*i0, 2*i0+1, 2*j0, 2*j0+1, quads=quads)
            self.subdivide_quad(depth+1, 2*i0, 2*i0+1, 2*j0+1, 2*j0+2, quads=quads)
            self.subdivide_quad(depth+1, 2*i0+1, 2*i0+2, 2*j0, 2*j0+1, quads=quads)
            self.subdivide_quad(depth+1, 2*i0+1, 2*i0+2, 2*j0+1, 2*j0+2, quads=quads)
        elif h and not v: # Split only horizontally
            self.subdivide_quad(depth+1, 2*i0, 2*i1, 2*j0, 2*j0+1, quads=quads)
            self.subdivide_quad(depth+1, 2*i0, 2*i1, 2*j0+1, 2*j0+2, quads=quads) 
        elif v and not h: # Split only vertically
            self.subdivide_quad(depth+1, 2*i0, 2*i0+1, 2*j0, 2*j1, quads=quads)
            self.subdivide_quad(depth+1, 2*i0+1, 2*i0+2, 2*j0, 2*j1, quads=quads)
        else: # We are done, both sides are within mesh size limits
            quads.append([(depth, i0, j0),
                          (depth, i0, j1),
                          (depth, i1, j0),
                          (depth, i1, j1)])

        return quads


def mesh(surf, mesh_size, start_depth=3):
    return Mesher(surf, mesh_size, start_depth).mesh_to_triangles()

l1 = G.Path.line([0.2, 0, 0], [1., 0, 0])

surf = l1.revolve_z()

start = time.time()
points, triangles = mesh(surf, 0.08, start_depth=3)
end = time.time()

print(end-start, (end-start)/len(triangles), len(triangles))

P.plot_mesh(G.Mesh(points, triangles=triangles))











