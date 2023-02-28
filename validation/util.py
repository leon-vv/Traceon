import time
import argparse

from pygmsh import *
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as S
import traceon.plotting as P

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-N', default=None, type=int, help='Mesh size will be taken as 1/N')
parser.add_argument('--symmetry', choices=['3d', 'radial'], default='radial', help='Choose the symmetry to use for the geometry (3d or radially symmetric)')
parser.add_argument('--plot-accuracy', action='store_true', help='Plot the accuracy as a function of time and number of elements')
parser.add_argument('--plot-geometry', action='store_true', help='Plot the geometry')

def print_info(Nlines, duration, accuracy):
    print('Number of elements\t\tComputation time (ms)\t\tAccuracy')
    
    for N, d, a in zip(Nlines, duration, accuracy):
        print('%d\t\t\t\t%.1f\t\t\t\t%.1e' % (N, d, a))

def parse_validation_args(create_geometry, compute_error, N={'radial': [10,50,100,300,500,700]}, **colors):
     
    args = parser.parse_args()
    Ndefault = args.N if args.N != None else N[args.symmetry][1]
    
    if args.plot_geometry:
        geom = create_geometry(Ndefault, args.symmetry, True)
        assert geom.symmetry == args.symmetry
        if geom.symmetry != '3d':
            P.show_line_mesh(geom.mesh, **colors) 
        else:
            P.show_triangle_mesh(geom.mesh, **colors)
        P.show()
    elif args.plot_accuracy:
        num_lines = []
        times = []
        errors = []

        for n in N[args.symmetry]:
            st = time.time()
            geom = create_geometry(n, args.symmetry, False)
            N, err = compute_error(geom)
            num_lines.append(N)
            times.append( (time.time() - st)*1000)
            errors.append(err)
        
        print_info(num_lines, times, errors)

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

    else:
        st = time.time()
        geom = create_geometry(Ndefault, args.symmetry, False)
        N, err = compute_error(geom)
        duration = (time.time() - st)*1000
        print_info([N], [duration], [err])



