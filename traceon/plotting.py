import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import *
import numpy as np

def _create_point_to_physical_dict(mesh):
    d = {}
    
    for k, v in mesh.cell_sets_dict.items():
        
        if 'triangle' in v: 
            for p in mesh.cells_dict['triangle'][v['triangle']]:
                a, b, c = p
                d[a], d[b], d[c] = k, k, k
        
        if 'line' in v:
            for l in mesh.cells_dict['line'][v['line']]:
                a, b = l
                d[a], d[b] = k, k
     
    return d

def show_line_mesh(mesh, trajectory=None, show_legend=True, **colors):
    plt.figure(figsize=(10, 13))
    plt.gca().set_aspect('equal')
     
    dict_ = _create_point_to_physical_dict(mesh)
    lines = mesh.cells_dict['line']

    to_plot_x = []
    to_plot_y = []
    colors_ = []
    
    for (A, B) in lines:
        color = '#CCC'

        if A in dict_ and B in dict_:
            phys1, phys2 = dict_[A], dict_[B]
            if phys1 == phys2 and phys1 in colors:
                color = colors[phys1]
         
        p1, p2 = mesh.points[A], mesh.points[B]
        to_plot_x.append( [p1[0], p2[0]] )
        to_plot_y.append( [p1[1], p2[1]] )
        colors_.append(color)
     
    colors_ = np.array(colors_)
     
    for c in set(colors_):
        mask = colors_ == c
        plt.plot(np.array(to_plot_x)[mask].T, np.array(to_plot_y)[mask].T, color=c, linewidth=3)
        plt.scatter(np.array(to_plot_x)[mask].T, np.array(to_plot_y)[mask].T, color=c, s=11)

    if show_legend:
        for l, c in colors.items():
            plt.plot([], [], label=l, color=c)
        plt.legend()
     
    plt.xlabel('r (mm)')
    plt.ylabel('z (mm)')
    plt.axvline(0, color='black', linestyle='dashed')

    if trajectory is not None:
        plt.plot(trajectory[:, 0], trajectory[:, 1])

def show_charge_density(lines, charges):
    # See https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    assert len(lines) == len(charges)

    plt.figure()
    segments = lines[:, :, :2] # Remove z value
    
    amplitude = np.mean(np.abs(charges))
    norm = plt.Normalize(-3*amplitude, 3*amplitude)
    
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    lc.set_array(charges)
    lc.set_linewidth(4)
    line = plt.gca().add_collection(lc)
    plt.xlim(np.min(lines[:, :, 0])-0.2, np.max(lines[:, :, 0])+0.2)
    plt.ylim(np.min(lines[:, :, 1])-0.2, np.max(lines[:, :, 1])+0.2)
    plt.xlabel('r (mm)')
    plt.ylabel('z (mm)')
    plt.show()
    

def show(legend=False):
    if legend:
        plt.legend()
    plt.show()





