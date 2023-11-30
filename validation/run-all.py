
import traceon.geometry as G

from capacitance_sphere import CapacitanceSphere
from dohi import DohiMirror
from edwards2007 import Edwards2007
from einzel_lens import EinzelLens
from simple_mirror import SimpleMirror
from spherical_capacitor import SphericalCapacitor
from spherical_capacitor_floating_conductor import SphericalFloatingConductor
from two_cylinder_edwards import TwoCylinderEdwards

validations = [CapacitanceSphere,
               DohiMirror,
               Edwards2007,
               EinzelLens,
               SimpleMirror,
               SphericalCapacitor,
               SphericalFloatingConductor,
               TwoCylinderEdwards]

def run_all(symmetry, higher_order, use_fmm):
    for v in validations:
        print('='*80, ' ', v.__name__, ' ', '='*10, str(symmetry))
        validation = v()
        MSF = validation.default_MSF(symmetry)[1]
        
        if use_fmm and not validation.supports_fmm():
            print('Validation does not support FMM')
            continue
        
        validation.print_accuracy(MSF, symmetry, higher_order=higher_order, use_fmm=use_fmm)

#run_all(G.Symmetry.RADIAL, True, False)
run_all(G.Symmetry.RADIAL, False, False)

run_all(G.Symmetry.THREE_D, False, False)
run_all(G.Symmetry.THREE_D, True, False)
run_all(G.Symmetry.THREE_D, False, True)


