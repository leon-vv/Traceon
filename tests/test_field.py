import os.path as path

import unittest
from math import *

import traceon.mesher as M
from traceon.geometry import *
import traceon.excitation as E
import traceon.solver as S
from traceon.field import *
import traceon.logging as logging

logging.set_log_level(logging.LogLevel.SILENT)

class FieldGeometryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # simple field; details are not important for geometric tests
        pos = Path.rectangle_xz(0.1,1,1, 1.5)
        neg = Path.rectangle_xz(0.1,1,-1.5, -1)
        neg.name='neg'
        pos.name='pos'

        mesh = (neg + pos).mesh(mesh_size=1)

        excitation = E.Excitation(mesh, E.Symmetry.RADIAL)
        excitation.add_voltage(pos=1)
        excitation.add_magnetostatic_potential(neg=-1)
        cls.field = S.solve_direct(excitation)
        cls.field_axial = FieldRadialAxial(cls.field, -2, 2, 100)

    def test_map_points(self):
        # translation
        field_trans = self.field.move(dx=1)
        # origin should shift
        assert np.allclose(field_trans.get_origin(), [1.,0.,0.]) 
        # basis vectors should remain invariant
        assert np.allclose(field_trans.get_basis(), self.field.get_basis()) 
        # in homogenous coords the tranformation matrix is [[R, t], [0,1]]
        assert np.allclose(
            np.linalg.inv(field_trans._inverse_transformation_matrix), np.array([[1,0,0,1], 
                                                                                [0,1,0,0],
                                                                                [0,0,1,0],
                                                                                [0,0,0,1]]))
        #rotation after translation
        field_trans_rot = field_trans.rotate(Ry=np.pi/2)
        # local origin [1,0,0] rotates as well in global coordinate system
        assert np.allclose(field_trans_rot.get_origin(), [0,0,-1]) 
        # x-> -z, y-> y, z-> -x
        assert np.allclose(field_trans_rot.get_basis(), np.array([[0,0,-1],[0,1,0],[1,0,0]])) 
        #T_total = T_rot @ T_trans
        assert np.allclose(
            np.linalg.inv(field_trans_rot._inverse_transformation_matrix), np.array([[0,0,-1,0], 
                                                                                    [0,1,0,0],
                                                                                    [1,0,0,-1],
                                                                                    [0,0,0,1]]))
   
    def test_map_points_to_local(self):
        field_trans = self.field.move(dx=1)
        point = np.array([1,1,1])
        point_trans = point + np.array([1,0,0]) # point translated over same distance as field
        #translated point should be original point in translated system
        assert np.allclose(field_trans.map_points_to_local(point_trans), point)

        # origin should locally always coincide with standard origin
        field_trans_rot = field_trans.rotate(Ry=np.pi/2)
        assert np.allclose(field_trans_rot.map_points_to_local(field_trans_rot.get_origin()), self.field.get_origin())
        # NOTE: map_points_to_local does not work for basis as they are direction vectors not points
    
    def test_field_axial_coordinate_system(self):
        field_axial_trans_rot1 = self.field_axial.move(dz=1).rotate(Rz=np.pi)

        field_trans_rot = self.field.move(dz=1).rotate(Rz=np.pi)
        field_axial_trans_rot2= FieldRadialAxial(field_trans_rot, -2, 2, 100)
        
        # it should not matter wheter we interpolate or transform first
        # FieldAxial should inherit the coordinate system of its base field
        assert np.allclose(field_axial_trans_rot1._inverse_transformation_matrix, 
                           field_axial_trans_rot2._inverse_transformation_matrix)
    
    def test_potential_at_point(self):
        field_trans = self.field.move(dx=1)
        field_axial_trans = self.field_axial.move(dx=1)

        pg = np.array([1,1,1])
        pl = field_trans.map_points_to_local(pg)

        assert np.allclose(field_trans.electrostatic_potential_at_point(pg),
                        field_trans.electrostatic_potential_at_local_point(pl))
        
        assert np.allclose(field_trans.magnetostatic_potential_at_point(pg),
                        field_trans.magnetostatic_potential_at_local_point(pl))
        
        assert np.allclose(field_axial_trans.electrostatic_potential_at_point(pg),
                        field_axial_trans.electrostatic_potential_at_local_point(pl))
        
        assert np.allclose(field_axial_trans.magnetostatic_potential_at_point(pg),
                        field_axial_trans.magnetostatic_potential_at_local_point(pl))
        
    def test_field_at_point(self):
        field_trans = self.field.move(dx=1)
        field_axial_trans = self.field_axial.move(dx=1)

        pg = np.array([1,1,1])
        pl = field_trans.map_points_to_local(pg)

        assert np.allclose(field_trans.electrostatic_field_at_point(pg),
                        field_trans.electrostatic_field_at_local_point(pl))
        
        assert np.allclose(field_trans.magnetostatic_field_at_point(pg),
                        field_trans.magnetostatic_field_at_local_point(pl))
        
        assert np.allclose(field_axial_trans.electrostatic_field_at_point(pg),
                        field_axial_trans.electrostatic_field_at_local_point(pl))
        
        assert np.allclose(field_axial_trans.magnetostatic_field_at_point(pg),
                        field_axial_trans.magnetostatic_field_at_local_point(pl))
    
    def test_field_bounds(self):
        field_mirr = self.field.mirror_xy()
        field_mirr.set_bounds([[0,1], [0,1], [0,1]])
        self.field.set_bounds([[0,1],[0,1], [0,1]])

        # field bounds are local by default so global orientation should not matter
        assert np.allclose(field_mirr.field_bounds, self.field.field_bounds)

        # if set in global coordinates, field bounds should be transformed to local coordinates
        field_mirr.set_bounds([[0,1], [0,1], [0,1]], global_coordinates=True)
        assert np.allclose(field_mirr.field_bounds, np.array([[0,1], [0,1], [-1,0]]))

        # point is outside bounds if its global, but inside if its local
        
        point = [0.5, 0.5, -0.5]
        assert field_mirr.electrostatic_potential_at_point(point) == 0.
        assert field_mirr.electrostatic_potential_at_local_point(point) != 0.


    def test_field_superposition(self):
        field_scaled = 2 * self.field
        field_trans  = self.field.move(dx=1)
        field_magnetic_only = FieldRadialBEM(self.field.magnetostatic_point_charges)

        field_axial_scaled = 2 * self.field_axial
        field_axial_trans = self.field.move(dx=1)
        field_axial_diff_z = FieldRadialAxial(self.field, -2, 2, 200)

        # fields with same underlying geometry are added directly
        assert isinstance(self.field + field_scaled, FieldRadialBEM)
        assert isinstance(self.field_axial + field_axial_scaled, FieldRadialAxial)
        assert isinstance(self.field + field_magnetic_only, FieldRadialBEM)

        # fields with different geometry or type become a superposition
        assert isinstance(self.field + field_trans, FieldSuperposition)
        assert isinstance(self.field_axial + field_axial_trans, FieldSuperposition)
        assert isinstance(self.field + self.field_axial, FieldSuperposition)
        assert isinstance(self.field_axial + field_axial_diff_z, FieldSuperposition)

        field_sup = FieldSuperposition([self.field, field_trans])
        field_sup2 = field_sup + field_trans
        field_sup3 = field_sup + self.field_axial

        assert len(field_sup2) == 3 and len(field_sup3) == 3
        
