import time
import argparse

from pygmsh import *
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as S
import traceon.plotting as P

def print_info(MSFs, Nlines, duration, correct, computed, accuracy):
    print('\n%-25s %-25s %-25s %-25s %-25s %-25s' % ('Mesh size factor', 'Number of elements', 'Computation time (ms)',
        'Correct value', 'Computed value', 'Relative error'))
    
    for m, N, d, corr, comp, a in zip(MSFs, Nlines, duration, correct, computed, accuracy):
        print('%-25d %-25d %-25.1f %-25.8f %-25.8f %-25.1e' % (m, N, d, corr, comp, a))

default_MSF ={'radial':[10, 25, 50, 100, 150], '3d':[20, 50, 100, 200, 400]}

class Validation:
    
    def __init__(self, description=''):
        self.description = description
        self.plot_colors = {}
     
    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

        if self.description != '':
            parser.description = self.description
        
        parser.add_argument('-MSF', '--mesh-size-factor', dest='MSF', default=None, type=int, help='Mesh size factor')
        parser.add_argument('--symmetry', choices=['3d', 'radial'], default='radial', help='Choose the symmetry to use for the geometry (3d or radially symmetric)')
        parser.add_argument('--plot-accuracy', action='store_true', help='Plot the accuracy as a function of time and number of elements')
        parser.add_argument('--plot-geometry', action='store_true', help='Plot the geometry')
        parser.add_argument('--plot-normals', action='store_true', help='When plotting geometry, show normals')
        parser.add_argument('--plot-charge-density', action='store_true', help='When plotting geometry, base the colors on the computed charge density')
        parser.add_argument('--plot-charges', action='store_true', help='When plotting geometry, base the colors on the charge on each element')
        parser.add_argument('--use-fmm', action='store_true', help='Use fast multipole method to solve 3D geometry')
        return parser.parse_args()

   
    def is_3d(self, mesh):
        return mesh.symmetry in [G.Symmetry.THREE_D_HIGHER_ORDER, G.Symmetry.THREE_D]
    
    def plot_geometry(self, MSF, symmetry, use_fmm=False, plot_charges=False, plot_charge_density=False, plot_normals=False):
        geom = self.create_mesh(MSF, symmetry)
        assert geom.symmetry == symmetry 
         
        if plot_charges or plot_charge_density:
            exc, field = self.compute_field(geom, use_fmm=use_fmm)
            P.plot_charge_density(exc, field, density=plot_charge_density)
        else:
            P.plot_mesh(geom, show_normals=plot_normals, **self.plot_colors) 
    
    def plot_accuracy(self, MSFs, symmetry, use_fmm=False):
        num_lines = []
        times = []
        correct = []
        computed = []
        errors = []

        for n in MSFs:
            print('-'*81, f' MSF={n}')
            st = time.time()
            geom = self.create_mesh(n, symmetry)
            exc, field = self.compute_field(geom, use_fmm=use_fmm)
            
            corr = self.correct_value_of_interest()  
            comp = self.compute_value_of_interest(geom, field)
            err = self.compute_accuracy(comp, corr)
             
            num_lines.append(exc.get_number_of_matrix_elements())
            times.append( (time.time() - st)*1000)
            correct.append(corr)
            computed.append(comp)
            errors.append(err)
        
        print_info(MSFs, num_lines, times, correct, computed, errors)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.sca(ax1)
        plt.xlabel('Number of elements')
        plt.ylabel('Accuracy (relative error)')
        plt.yscale('log')
        plt.xscale('log')
        plt.plot(num_lines, errors)
        plt.scatter(num_lines, errors)

        plt.sca(ax2)
        plt.plot(times, errors)
        plt.scatter(times, errors)
        plt.xlabel('Computation time (ms)')
        plt.ylabel('Accuracy (relative error)')
        plt.yscale('log')
        
        plt.sca(ax3)
        plt.plot(num_lines, times)
        plt.scatter(num_lines, times)
        plt.xlabel('Number of elements')
        plt.ylabel('Computation time (ms)')
        plt.show()

    def print_accuracy(self, MSF, symmetry, use_fmm=False):
        st = time.time()
        geom = self.create_mesh(MSF, symmetry)
        exc, field = self.compute_field(geom, use_fmm=use_fmm)
        
        correct = self.correct_value_of_interest()  
        computed = self.compute_value_of_interest(geom, field)
        err = self.compute_accuracy(computed, correct)
         
        duration = (time.time() - st)*1000
        print_info([MSF], [exc.get_number_of_matrix_elements()], [duration], [correct], [computed], [err])
     
    def args_to_symmetry(args):
        if args.symmetry == 'radial':
            assert not args.use_fmm, "Fast Multipole Method not supported for radial geometries"
            return G.Symmetry.RADIAL
        elif args.symmetry == '3d' and not args.use_fmm:
            return G.Symmetry.THREE_D_HIGHER_ORDER
        elif args.symmetry == '3d' and args.use_fmm:
            return G.Symmetry.THREE_D
         
        return G.Symmetry.RADIAL
    
    def run_validation(self):
        args = self.parse_args()
        plot = args.plot_geometry or args.plot_normals or args.plot_charge_density or args.plot_charges
        MSF = args.MSF if args.MSF != None else default_MSF[args.symmetry][1]
        symmetry = Validation.args_to_symmetry(args)
         
        if plot:
            self.plot_geometry(MSF, symmetry, plot_charges=args.plot_charges, \
                                            plot_charge_density=args.plot_charge_density,
                                            plot_normals=args.plot_normals) 
        elif args.plot_accuracy:
            self.plot_accuracy(default_MSF[args.symmetry], symmetry, use_fmm=args.use_fmm)
        else:
            self.print_accuracy(MSF, symmetry, use_fmm=args.use_fmm)

    # Should be implemented by each of the validations
    def create_mesh(self, MSF, symmetry):
        pass
    
    def get_excitation(self, geometry):
        pass

    def compute_field(self, geometry, use_fmm=False):
        exc = self.get_excitation(geometry)
        return exc, S.solve_bem(exc, use_fmm=use_fmm)
     
    def compute_value_of_interest(self, geometry, field):
        pass

    def correct_value_of_interest(self):
        pass

    def compute_accuracy(self, computed, correct):
        return abs(computed/correct - 1)
     










