import time
import argparse

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

class Validation:
    
    def __init__(self, description=''):
        self.description = description
        self.plot_colors = {}
        self.args = self.parse_args()
     
    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

        if self.description != '':
            parser.description = self.description
        
        parser.add_argument('-MSF', '--mesh-size-factor', dest='MSF', default=None, type=int, help='Mesh size factor')
        parser.add_argument('--symmetry', choices=['3d', 'radial'], default='radial', help='Choose the symmetry to use for the geometry (3d or radially symmetric)')
        parser.add_argument('--higher-order', action='store_true', help='Use higher order (curved) elements')
        parser.add_argument('--plot-accuracy', action='store_true', help='Plot the accuracy as a function of time and number of elements')
        parser.add_argument('--plot-geometry', action='store_true', help='Plot the geometry')
        parser.add_argument('--plot-normals', action='store_true', help='When plotting geometry, show normals')
        parser.add_argument('--plot-charge-density', action='store_true', help='When plotting geometry, base the colors on the computed charge density')
        parser.add_argument('--plot-charges', action='store_true', help='When plotting geometry, base the colors on the charge on each element')
        parser.add_argument('--use-fmm', action='store_true', help='Use fast multipole method to solve 3D geometry')
        parser.add_argument('--fmm-precision', type=int, choices=[-2, -1,0,1,2,3], default=0, help='Use fast multipole method to solve 3D geometry')
        
        return parser.parse_args()
    
    def default_MSF(self, symmetry):
        return [2,4,8,16,32]
    
    def supports_fmm(self):
        return True

    def supports_3d(self):
        return True
    
    def plot_geometry(self, MSF, symmetry, higher_order=False, plot_charges=False, plot_charge_density=False, plot_normals=False):
        geom = self.create_mesh(MSF, symmetry, higher_order)
         
        if plot_charges or plot_charge_density:
            exc, field = self.compute_field(geom, symmetry, use_fmm=use_fmm)
            P.plot_charge_density(exc, field, density=plot_charge_density)
        else:
            P.plot_mesh(geom, show_normals=plot_normals, **self.plot_colors) 
    
    def plot_accuracy(self, MSFs, symmetry, higher_order):
        num_lines = []
        times = []
        correct = []
        computed = []
        errors = []

        for n in MSFs:
            print('-'*81, f' MSF={n}')
            st = time.time()
            geom = self.create_mesh(n, symmetry, higher_order)
            exc, field = self.compute_field(geom, symmetry, use_fmm=use_fmm)
             
            corr = self.correct_value_of_interest()  
            comp = self.compute_value_of_interest(geom, field)
            err = self.compute_accuracy(comp, corr)
             
            num_lines.append(len(exc.get_electrostatic_active_elements()[0]))
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

    def print_accuracy(self, MSF, symmetry, higher_order=True, use_fmm=False):
        st = time.time()
        geom = self.create_mesh(MSF, symmetry, higher_order)
        exc, field = self.compute_field(geom, symmetry, use_fmm=use_fmm)
        correct = self.correct_value_of_interest()  
        computed = self.compute_value_of_interest(geom, field)
        err = self.compute_accuracy(computed, correct)
         
        duration = (time.time() - st)*1000
        print_info([MSF], [len(exc.get_electrostatic_active_elements()[0])], [duration], [correct], [computed], [err])
        return duration, err
     
    def args_to_symmetry(args):
        if args.symmetry == 'radial':
            assert not args.use_fmm, "Fast Multipole Method not supported for radial geometries"
            return E.Symmetry.RADIAL
        elif args.symmetry == '3d':
            assert not (args.use_fmm and args.higher_order), "Fast Multipole Method not supported for higher order elements"
            return E.Symmetry.THREE_D
          
        return E.Symmetry.RADIAL
    
    def run_validation(self):
        args = self.parse_args()
        
        assert args.symmetry != '3d' or not args.higher_order, "Higher order meshes not supported in 3D"
        
        plot = args.plot_geometry or args.plot_normals or args.plot_charge_density or args.plot_charges
        symmetry = Validation.args_to_symmetry(args)
        MSF = args.MSF if args.MSF != None else self.default_MSF(symmetry)[1]
         
        if plot:
            self.plot_geometry(MSF, symmetry, higher_order=args.higher_order,
                                            plot_charges=args.plot_charges,
                                            plot_charge_density=args.plot_charge_density,
                                            plot_normals=args.plot_normals) 
        elif args.plot_accuracy:
            self.plot_accuracy(self.default_MSF(symmetry), symmetry)
        else:
            self.print_accuracy(MSF, symmetry, args.higher_order, args.use_fmm)

    # Should be implemented by each of the validations
    def create_mesh(self, MSF, symmetry, higher_order):
        pass
    
    def get_excitation(self, geometry):
        pass
    
    def compute_field(self, geometry, symmetry, use_fmm=False):
        exc = self.get_excitation(geometry, symmetry)
        return exc, S.solve_bem(exc, use_fmm=use_fmm, fmm_precision=self.args.fmm_precision)
     
    def compute_value_of_interest(self, geometry, field):
        pass

    def correct_value_of_interest(self):
        pass

    def compute_accuracy(self, computed, correct):
        return abs(computed/correct - 1)
     










