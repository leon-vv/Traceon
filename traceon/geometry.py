import numpy as np
from pygmsh import *

import pickle

class Geometry:
    """Class containing a mesh and related metadata.
    
    Attributes:
        mesh: [meshio](https://github.com/nschloe/meshio) object containing the mesh
        metadata: dictionary containing arbitrary metadata
    """
    
    def __init__(self, mesh, N, metadata={}):
        """ Args: """
        self.mesh = mesh
        self.metadata = metadata
        self.N = N
    
    def write(self, filename):
        """Write a mesh to a file. The pickle module will be used
        to save the Geometry object.

        Args:
            filename: name of the file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
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
            return Geometry(object_['mesh'], object_['N'], object_['metadata'])
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

    def _get_distance_to_name(self, z_zero_name):
        # Distance to z_zero
        distance_until_zero = 0.0

        for element in self.stack_elements:
            if element[0] in ['electrode', 'custom']:
                _, _, thickness, name = element
            
                distance_until_zero += thickness
                
                if name == z_zero_name:
                    break
            else:
                distance_until_zero += element[1]

        return distance_until_zero
     
    def build_geometry(self, z_zero_name, N, enclose_right=True, electrode_width=2, margin_right=0.5):
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
        bottom = -self._get_distance_to_name(z_zero_name)
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
             
            return Geometry(geom.generate_mesh(dim=1), N)

if __name__ == '__main__':
    # Some tests/examples with MEMSstack.
    # Run this file directly 'python3 ./geometry.py' to see it in action.
    import plotting as P
    stack_one = MEMSStack()
    stack_one.add_electrode(1, 2, 'one')
    P.show_line_mesh(stack_one.build_geometry('one', 100).mesh, one='green')
    P.show()
     
    P.show_line_mesh(stack_one.build_geometry('one', 100, enclose_right=False).mesh, one='green')
    P.show()
    
    stack_two = MEMSStack()
    stack_two.add_electrode(1, 2, 'first')
    stack_two.add_spacer(1)
    stack_two.add_electrode(1.5, 2, 'second')
    
    P.show_line_mesh(stack_two.build_geometry('first', 100, enclose_right=False).mesh, first='blue', second='red')
    P.show()
    
    P.show_line_mesh(stack_two.build_geometry('first', 100, enclose_right=True).mesh, first='blue', second='red')
    P.show()
    
    stack_tree = MEMSStack()
    stack_tree.add_electrode(1, 2, 'first')
    stack_tree.add_spacer(1)
    stack_tree.add_electrode(1.5, 2, 'second')
    stack_tree.add_spacer(0.5)
    stack_tree.add_electrode(1.5, 2, 'first')
    
    P.show_line_mesh(stack_tree.build_geometry('second', 100, enclose_right=False).mesh, first='blue', second='red')
    P.show()
    
    P.show_line_mesh(stack_tree.build_geometry('second', 100, enclose_right=True).mesh, first='blue', second='red')
    P.show()
    


# Correction properties of electron mirrors.
# D. Preikszas and H. Rose.
# 1997.
def create_preikszas_mirror(N=100): # Radius is 5
    """Generate the mirror studied in the paper:
    D. Preikszas and H. Rose. Correction properties of electron mirrors. 1997.
    
    Args:
        N: number of mesh points. The size of the mesh elements will be chosen such that a line that is
            as long as the geometry is tall will have N points."""
     
    with occ.Geometry() as geom:
        
        points = [
            [0, 0],   #0
            [3, 0],   #1
            [5, -2],  #2
            [5, -3],  #3
            [7, -5],  #4
            [25, -5], #5
            [25, -10],#6
            [7, -10], #7
            [5, -12], #8
            [5, -45], #9
            [0, -45], #10
        ] 

        centers = [[3, -2], [7, -3], [7, -12]]
        
        lcar = 45/N
        points = [geom.add_point(p, lcar) for p in points]
        centers = [geom.add_point(p, lcar) for p in centers]
             
        l1 = geom.add_line(points[0], points[1])
        l2 = geom.add_circle_arc(points[1], centers[0], points[2])
        l3 = geom.add_line(points[2], points[3])
        l4 = geom.add_circle_arc(points[3], centers[1], points[4])
        l5 = geom.add_line(points[4], points[5])
        l6 = geom.add_line(points[5], points[6])
        l7 = geom.add_line(points[6], points[7])
        l8 = geom.add_circle_arc(points[7], centers[2], points[8])
        l9 = geom.add_line(points[8], points[9])
        l10 = geom.add_line(points[9], points[10])
        l11 = geom.add_line(points[10], points[0])

        geom.add_physical([l11], 'axis')
        geom.add_physical([l1, l2, l3, l4, l5], 'mirror')
        geom.add_physical([l7, l8, l9, l10], 'corrector')

        cl = geom.add_curve_loop([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11])
        
        return geom.generate_mesh(dim=1)


def create_two_cylinder_lens(S=0.2, R=1, N=200, wall_thickness=1, boundary_length=20):
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
            physicals = [('v1', [0, 1, 2]), ('v2', [4, 5, 6])]
        else:
            points = [ [0, 0],
                [R, 0],
                [R, cylinder_length],
                [R, cylinder_length + S],
                [R, boundary_length],
                [0, boundary_length]]
            physicals = [('v1', [0, 1]), ('v2', [3, 4])]
         
        lcar = boundary_length/N
        poly = geom.add_polygon(points, lcar)
         
        for key, indices in physicals:
            lines = [poly.curves[idx] for idx in indices]
            geom.add_physical(lines, key)
         
        return geom.generate_mesh(dim=1)


def create_dohi_mirror(N=1500):
    """Generate the mirror studied in the following paper:
       H. Dohi, P. Kruit. Design for an aberration corrected scanning electron microscope using miniature electron mirrors. 2018.
        
        N: number of mesh points. The size of the mesh elements will be chosen such that a line that is
            as long as the geometry is tall will have N points.
    """
    
    with occ.Geometry() as geom:
        
        boundary_width = 3
        r = 0.15/2 
        t = 0.2
        s = 0.5
        margin_top = 0.5
        margin_right = 0.5
        margin_bottom = 0.5

        points = [
            [0, 0], # Mirror
            [r, 0],
            [r, t],
            [r + 20*r, t],
            [r + 20*r, t+s], # lens, 4
            [r, t+s],
            [r, 2*t+s],
            [r + 20*r, 2*t+s],
            [r + 20*r, 2*t+2*s], # Ground
            [r, 2*t+2*s],
            [r, 3*t+2*s+margin_top],
            [boundary_width+margin_right, 3*t+2*s+margin_top],
            [boundary_width+margin_right, 0.0],
            [boundary_width+margin_right, -margin_bottom],
            [0.0, -margin_bottom],
        ]

        lcar = (3*t+2*s)/N
        poly = geom.add_polygon(points, lcar)
          
        for key, indices in [('mirror', [0, 1, 2]), ('lens', [4, 5, 6]), ('ground', [8, 9, 10, 11, 12, 13])]:
            geom.add_physical([poly.curves[i] for i in indices], key)

        return Geometry(geom.generate_mesh(dim=1), N)


def create_edwards2007(N):
    """Create the geometry g5 (figure 2) from the following paper:
    D. Edwards. High precision electrostatic potential calculations for cylindrically
    symmetric lenses. 2007.
    """
    with occ.Geometry() as geom:
        points = [
            [0, 0],
            [0, 5],
            [12, 5],
            [12, 15],
            [0, 15],
            [0, 20],
            [20, 20],
            [20, 0]
        ]
        
        lcar = 20/N
        poly = geom.add_polygon(points, lcar)
        
        for key, indices in [('inner', [1, 2, 3]), ('boundary', [5,6,7])]:
            geom.add_physical([poly.curves[i] for i in indices], key)
        
        return Geometry(geom.generate_mesh(dim=1), N)

def create_spherical_capacitor(N):
    """Create the spherical deflection analyzer from the following paper

    D. Cubric, B. Lencova, F.H. Read, J. Zlamal
    Comparison of FDM, FEM and BEM for electrostatic charged particle optics.
    1999.
    """
    with occ.Geometry() as geom:
        
        r1 = 7.5
        r2 = 12.5

        points = [
            [0, -r2],
            [0, -r1],
            [r1, 0],
            [0, r1],
            [0, r2],
            [r2, 0]
        ]
        
        lcar = r2/N
        points = [geom.add_point(p, lcar) for p in points]
        center = geom.add_point([0, 0], lcar)
         
        l1 = geom.add_line(points[0], points[1])
        l2 = geom.add_circle_arc(points[1], center, points[2])
        l3 = geom.add_circle_arc(points[2], center, points[3])
        
        l4 = geom.add_line(points[3], points[4])
        l5 = geom.add_circle_arc(points[4], center, points[5])
        l6 = geom.add_circle_arc(points[5], center, points[0])
        
        geom.add_physical([l2, l3], 'inner')
        geom.add_physical([l5, l6], 'outer')
        
        cl = geom.add_curve_loop([l1, l2, l3, l4, l5, l6])

        return Geometry(geom.generate_mesh(dim=1), N)




