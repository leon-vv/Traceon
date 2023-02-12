import numpy as np
from pygmsh import *

import pickle


def revolve_around_optical_axis(geom, elements, factor=1.0):
    revolved = []
    
    for e in elements:
        
        top = e
        for i in range(4):
            top, extruded, lateral = geom.revolve(top, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], factor*0.5*np.pi)
            revolved.append(extruded)
        
    return revolved


class Geometry:
    """Class containing a mesh and related metadata.
    
    Attributes:
        mesh: [meshio](https://github.com/nschloe/meshio) object containing the mesh
        metadata: dictionary containing arbitrary metadata
    """
    
    def __init__(self, mesh, N, zmin=None, zmax=None, metadata={}, symmetry='radial'):
        """ Args: """
        self.mesh = mesh
        self.metadata = metadata
        self.N = N
        self._symmetry = symmetry
        
        self.zmin = zmin if zmin != None else np.min(mesh.points[:, 1]) - 1
        self.zmax = zmax if zmax != None else np.max(mesh.points[:, 1]) + 1
     
    def get_z_bounds(self):
        if hasattr(self, 'zmin'):
            return self.zmin, self.zmax
         
        return np.min(self.mesh.points[:, 1]) - 1, np.max(self.mesh.points[:, 1]) + 2
     
    def write(self, filename):
        """Write a mesh to a file. The pickle module will be used
        to save the Geometry object.

        Args:
            filename: name of the file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def get_mesh_size(self):
        line_points = self.mesh.points[ self.mesh.cells_dict['line'] ]
        return np.mean(np.linalg.norm(line_points[:, 1] - line_points[:, 0], axis=1))
    
    @property
    def symmetry(self):
        if hasattr(self, '_symmetry'):
            return self._symmetry
        else:
            return 'radial'

    def get_electrodes(self):
        """Get the names of all the electrodes in the geometry.

        Returns:
            List of electrode names"""
        return list(self.mesh.cell_sets_dict.keys())
    
    def read(filename):
        """Read a geometry from disk (previously saved with the write method)
        
        Args:
            filename: the name of the file.
        """
        with open(filename, 'rb') as f:
            object_ = pickle.load(f)
        
        if isinstance(object_, dict): 
            # Backwards compatibility
            return Geometry(object_['mesh'], object_['N'], metadata=object_['metadata'])
        else:
            return object_
    
    def __str__(self):
        return str(self.mesh) + ' (metadata: ' + str(self.metadata) + ')'

class MEMSStack:
    """Create a MEMS geometry. A MEMS geometry is a stack of rectangular
    electrodes with spacer between them. The stack is built using the add_spacer,
    add_electrode and add_custom methods. The stack is built from bottom to top."""

    def __init__(self):
        # Stack elements are either ('spacer', radius, thickess)
        # or ('electrode', radius, thickness, name)
        self.stack_elements = []

    def add_spacer(self, thickness):
        """Add a spacer to the stack.

        Args:
            thickness: the thickness in millimeters.
        """

        self.stack_elements.append( ('spacer', thickness) )
     
    def add_electrode(self, radius, thickness, name):
        """Add an electorde to the stack.

        Args:
            radius: distance between the optical axis and the leftmost part
                of the rectangle making up the electrode (in millimeters).
            thickness: the thickness in millimeters.
            name: the name that will be used to refer to this electrode.
        """
        self.stack_elements.append( ('electrode', radius, thickness, name) )

    def add_custom(self, fun, thickness, name):
        """Add an custom electrode to the stack. A custom electrode need not be rectangular.

        Args:
            fun: function which will be called as fun(r, z) where r and z are the (bottom right)
                coordinates at which the electrode starts. The function should return a list of points
                making up the electrode (starting from (r,z) ).
            thickness: the thickness the electrode will be in millimeters.
            name: the name that will be used to refer to this electrode.
        """

        self.stack_elements.append( ('custom', fun, thickness, name) )

    def _get_distance_to_name(self, z_zero_name, center_zero):
        # Distance to z_zero
        distance_until_zero = 0.0

        for element in self.stack_elements:
            if element[0] in ['electrode', 'custom']:
                _, _, thickness, name = element
            
                distance_until_zero += thickness
                
                if name == z_zero_name:

                    if center_zero:
                        distance_until_zero -= thickness/2
                    
                    break
            else:
                distance_until_zero += element[1]

        return distance_until_zero
     
    def build_geometry(self, z_zero_name, N, enclose_right=True, electrode_width=2, margin_right=0.5, zero_in_center=False, **kwargs):
        """Generate a mesh from the current MEMSStack.The mesh will be returned as an instance of the Geometry
        class. 
        
        Args:
            z_zero_name: name of the electrode that will have z=0 at the top of the electrode. 
            N: the number of mesh points. A line with length equal to the height of the stack will have N
                mesh points.
            enclose_right: whether to connect the topmost and bottommost electrodes at the right side of the
                stack. For example, when the top and bottom electrodes are grounded connecting them on the right
                will make sure that V=0 at every point outside the geometry.
            electrode_width: the width of the electrodes (in millimeters).
            margin_right: distance between the enclosure on the right and the electrodes.
        """
         
        assert len(self.stack_elements) > 0, "Cannot build mesh for empty MEMSStack"
         
        # Build point list and physicals
        bottom = -self._get_distance_to_name(z_zero_name, zero_in_center)
        top = sum(e[2] if e[0] != 'spacer' else e[1] for e in self.stack_elements)

        lcar = (top-bottom)/N
        z = bottom
        
        to_physical = {e[-1]:[] for e in self.stack_elements if e[0] != 'spacer'}
        
        with occ.Geometry() as geom:
            for element in self.stack_elements:
                if element[0] == 'electrode':
                    _, radius, thickness, name = element
                    x0 = [radius, z]
                    points = [x0, [x0[0], x0[1]+thickness], [electrode_width, x0[1]+thickness], [electrode_width, x0[1]]]
                    poly = geom.add_polygon(points, lcar, make_surface=False)
                    if np.isclose(radius, 0.0):
                        to_physical[name].extend(poly.lines[1:]) # Don't add lines on the optical axis, leads to singular matrices
                    else:
                        to_physical[name].extend(poly.lines)
                elif element[0] == 'custom':
                    raise NotImplementedError
                    #_, fun, thickness, name = element
                    #generated_points = fun(electrode_width, z)
                else:
                    _, thickness = element
                
                z += thickness
             
            if enclose_right:
                p1 = geom.add_point([electrode_width+margin_right, z], lcar)
                p2 = geom.add_point([electrode_width+margin_right, bottom], lcar)
                line = geom.add_line(p1, p2)
                to_physical[name].append(line)
            
            for k, v in to_physical.items():
                geom.add_physical(v, k)
             
            return Geometry(geom.generate_mesh(dim=1), N, **kwargs)

def create_two_cylinder_lens(N=200, S=0.2, R=1, wall_thickness=1, boundary_length=20, gap_at_zero=False, include_boundary=True, **kwargs):
    """Generate lens consisting of two concentric cylinders. For example studied in
    David Edwards, Jr. Accurate Potential Calculations For The Two Tube Electrostatic Lens Using FDM A Multiregion Method.  2007.
    
    D. Cubric, B. Lencova, F.H. Read, J. Zlamal. Comparison of FDM, FEM and BEM for electrostatic charged particle optics. 1999.
     
    Args:
        S: spacing between the cylinders.
        R: radius of the cylinders.
        wall_thickness: thickness of the cylinders. 0.0 is a valid input.
        boundary_length: length of the entire lens.
    """
 
    with occ.Geometry() as geom:
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
         
        lcar = boundary_length/N
        poly = geom.add_polygon(points, lcar)
         
        for key, indices in physicals:
            lines = [poly.curves[idx] for idx in indices]
            geom.add_physical(lines, key)

        mesh = geom.generate_mesh(dim=1)
        
        if gap_at_zero:
            mesh.points[:, 1] -= cylinder_length + S/2
         
        return Geometry(mesh, N, **kwargs)



