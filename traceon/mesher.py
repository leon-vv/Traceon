from math import *
import numpy as np
import time

# TODO: fix circular dependency
import traceon.geometry as G
import traceon.plotting as P

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
        
        for quad in self.quads:
            depth, i0, i1, j0, j1 = quad 
            assert depth == self.depth
            
            triangles.append([
                self.indices[i0, j0],
                self.indices[i0, j1],
                self.indices[i1, j1]])
            
            triangles.append([
                self.indices[i0, j0],
                self.indices[i1, j1],
                self.indices[i1, j0]])
        
        assert not (-1 in np.array(triangles))
        return triangles
            
    def __getitem__(self, *args, **kwargs):
        self.indices.__getitem__(*args, **kwargs)
    
    def __setitem__(self, *args, **kwargs):
        self.indices.__setitem__(*args, **kwargs)


def subdivide_quads(pstack, mesh_size, to_subdivide=[], quads=[]): 
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

def mesh_subsections_to_quads(surface, mesh_size, start_depth):
    all_pstacks = []
    all_quads = []
    points = []
    
    for s in surface.sections():
        quads = []
        pstack = PointStack(s, points=points)
        
        for i in range(pstack.get_number_of_indices(start_depth) - 1):
            for j in range(pstack.get_number_of_indices(start_depth) - 1):
                subdivide_quads(pstack, mesh_size, to_subdivide=[(start_depth, i, i+1, j, j+1)], quads=quads)
        
        all_pstacks.append(pstack)
        all_quads.append(quads)
        points = pstack.points

    return points, all_pstacks, all_quads
    

def mesh(surface, mesh_size, start_depth=3, name=None):
    # Create a point stack for each subsection
    points, point_stacks, quads = mesh_subsections_to_quads(surface, mesh_size, start_depth)
     
    max_depth = max([p.depth() for p in point_stacks])
    print('Normalizing to depth: ', max_depth)
     
    # Normalize all the point stacks to the max depth of all sections 
    point_with_quads = [p.normalize_to_depth(max_depth, q, start_depth) for p, q in zip(point_stacks, quads)]
    
    # TODO: copy over edges
    
    points = np.array(points)
    triangles = np.concatenate([pq.to_triangles() for pq in point_with_quads], axis=0)
    
    assert points.shape == (len(points), 3)
    assert triangles.shape == (len(triangles), 3)
    assert np.all( (0 <= triangles) & (triangles < len(points)) )
     
    if name is not None:
        physical_to_triangles = {name:np.arange(len(triangles))}
    else:
        physical_to_triangles = {}
    
    return G.Mesh(points=points, triangles=triangles, physical_to_triangles=physical_to_triangles)












