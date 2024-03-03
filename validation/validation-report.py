
import traceon.geometry as G

from capacitance_sphere import CapacitanceSphere
from dohi import DohiMirror
from edwards2007 import Edwards2007
from einzel_lens import EinzelLens
from simple_mirror import SimpleMirror
from spherical_capacitor import SphericalCapacitor
from two_cylinder_edwards import TwoCylinderEdwards
from magnetic_einzel_lens import MagneticEinzelLens
from rectangular_coil import RectangularCoil
from rectangular_coil_with_circle import RectangularCoilWithCircle

validations = [CapacitanceSphere,
               DohiMirror,
               Edwards2007,
               EinzelLens,
               SimpleMirror,
               SphericalCapacitor,
               TwoCylinderEdwards,
               MagneticEinzelLens,
               RectangularCoil,
               RectangularCoilWithCircle]

report = []

def run_all(symmetry,higher_order, use_fmm):
    
    for v in validations:
        print('='*80, ' ', v.__name__, ' ', '='*10, str(symmetry))
        validation = v()
        MSF = validation.default_MSF(symmetry)[1]
        
        if use_fmm and not validation.supports_fmm():
            print('Validation does not support FMM')
            continue
        if symmetry.is_3d() and not validation.supports_3d():
            print('Validation does not support 3D')
            continue
         
        dur, err = validation.print_accuracy(MSF, symmetry, higher_order, use_fmm=use_fmm)
        report.append( (v.__name__, symmetry, higher_order, use_fmm, dur, err) )

run_all(G.Symmetry.RADIAL, True, False)
run_all(G.Symmetry.RADIAL, False, False)

run_all(G.Symmetry.THREE_D, False, False)
run_all(G.Symmetry.THREE_D, True, False)
run_all(G.Symmetry.THREE_D, False, True)

print(f'Solver symmetry higher order method duration error')
for (name, sym, higher_order, fmm, duration, err) in report:
    s = '3D' if sym.is_3d() else '2D'
    h = 'simple' if not higher_order else 'higher order'
    f = 'FMM' if fmm else 'Direct'
    print(f'{name:<30} | {s:<3} | {h:<15} | {f:<10} | {duration:>8.0f} ms | {err:>10.1e}')
     


