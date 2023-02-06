import time
import argparse

from pygmsh import *
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as S
import traceon.plotting as P

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-N', default=100, type=int, help='Mesh size will be taken as 1/N')
parser.add_argument('--plot-accuracy', action='store_true', help='Plot the accuracy as a function of time and number of line elements')
parser.add_argument('--plot-geometry', action='store_true', help='Plot the geometry')

def print_info(Nlines, duration, accuracy):
    print('Number of lines\t\tComputation time (ms)\t\tAccuracy')
    
    for N, d, a in zip(Nlines, duration, accuracy):
        print('%d\t\t\t%.1f\t\t\t\t%.1e' % (N, d, a))

def parse_validation_args(create_geometry, compute_error, N=[10,50,100,300,500,700], **colors):
    
    args = parser.parse_args()

    if args.plot_geometry:
        geom = create_geometry(args.N)
        P.show_line_mesh(geom.mesh, **colors) 
        P.show()
    elif args.plot_accuracy:
        num_lines = []
        times = []
        errors = []

        for n in N:
            st = time.time()
            N, err = compute_error(n)
            num_lines.append(N)
            times.append( (time.time() - st)*1000)
            errors.append(err)
        
        print_info(num_lines, times, errors)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.sca(ax1)
        plt.xlabel('Number of line elements')
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
        plt.xlabel('Number of line elements')
        plt.ylabel('Computation time (ms)')
        plt.show()

    else:
        st = time.time()
        N, err = compute_error(args.N)
        duration = (time.time() - st)*1000
        print_info([N], [duration], [err])



