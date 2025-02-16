"""The `voltrace.plotting` module uses the `vedo` plotting library to provide some convenience functions
to show the line and triangle meshes generated by Voltrace.

To show a mesh, for example use:
```Python3
plt.new_figure()
plt.plot_mesh(mesh)
plt.show()
```
Where mesh is created using the `voltrace.geometry` module.
"""
from __future__ import annotations

from math import sqrt
import numpy as np
import vedo
import vedo.shapes
import vedo.colors

from . import backend

from .typing import *

_current_figures = []

class Figure:
    def __init__(self, show_legend: bool = True) -> None:
        self.show_legend = show_legend
        self.is_2d = True
        self.legend_entries: list[Any] = []
        self.to_plot: list[Any] = []
     
    def plot_mesh(self, mesh: Mesh, show_normals: bool = False, **colors: str) -> None:
        """Plot mesh using the Vedo library. Optionally showing normal vectors.

        Parameters
        ---------------------
        mesh: `voltrace.mesher.Mesh`
            The mesh to plot
        show_normals: bool
            Whether to show the normal vectors at every element
        colors: dict of (string, string)
            Use keyword arguments to specify colors, for example `plot_mesh(mesh, lens='blue', ground='green')`
        """
        if not len(mesh.triangles) and not len(mesh.lines):
            raise RuntimeError("Trying to plot empty mesh.")

        triangle_normals, line_normals = None, None
        
        if len(mesh.triangles):
            meshes, triangle_normals = _get_vedo_triangles_and_normals(mesh, **colors)
            self.legend_entries.extend(meshes)
            self.to_plot.append(meshes)
        
        if len(mesh.lines):
            lines, line_normals = _get_vedo_lines_and_normals(mesh, **colors)
            self.legend_entries.extend(lines)
            self.to_plot.append(lines)
         
        if show_normals:
            if triangle_normals is not None:
                self.to_plot.append(triangle_normals)
            if line_normals is not None:
                self.to_plot.append(line_normals)
        
        self.is_2d &= mesh.is_2d()

    def plot_equipotential_lines(self,
        field: Field,
        surface: Surface,
        N0: int = 75,
        N1: int = 75,
        color_map: str = 'coolwarm',
        N_isolines: int = 40,
        isolines_width: int = 1,
        isolines_color: str = '#444444') -> None:
        """Make potential color plot including equipotential lines.

        Parameters
        -------------------------------------
        field: `voltrace.field.Field`
            The field used to compute the potential values (note that any field returned from the solver can be used)
        surface: `voltrace.geometry.Surface`
            The surface in 3D space which will be 'colored in'
        N0: int
            Number of pixels to use along the first 'axis' of the surface
        N1: int
            Number of pixels to use along the second 'axis' of the surface
        color_map: str
            Color map to use to color in the surface
        N_isolines: int
            Number of equipotential lines to plot
        isolines_width: int
            The width to use for the isolines. Pass in 0 to disable the isolines.
        isolines_color: str
            Color to use for the isolines"""
        grid = _get_vedo_grid(field, surface, N0, N1)
        isolines = grid.isolines(n=N_isolines).color(isolines_color).lw(isolines_width) # type: ignore
        grid.cmap(color_map)
        self.to_plot.append(grid)
        self.to_plot.append(isolines)
    
    def plot_trajectories(self, trajectories: list[Path], 
                xmin: float | None = None, xmax: float | None = None,
                ymin: float | None = None, ymax: float | None = None,
                zmin: float | None = None, zmax: float | None = None,
                color: str = '#00AA00', line_width: int = 1, N=1000) -> None:
        """Plot particle trajectories.

        Parameters
        ------------------------------------
        trajectories: list of numpy.ndarray
            List of positions as returned by `voltrace.tracing.Tracer.__call__`
        xmin, xmax: float
            Only plot trajectory points for which xmin <= x <= xmax
        ymin, ymax: float
            Only plot trajectory points for which ymin <= y <= ymax
        zmin, zmax: float
            Only plot trajectory points for which zmin <= z <= zmax
        color: str
            Color to use for the particle trajectories
        line_width: int
            Width of the trajectory lines
        """
        
        for trajectory in trajectories:
            
            _, t = trajectory.sample(N=N)
            
            if not len(t):
                continue
            
            mask = np.full(len(t), True)

            if xmin is not None:
                mask &= t[:, 0] >= xmin
            if xmax is not None:
                mask &= t[:, 0] <= xmax
            if ymin is not None:
                mask &= t[:, 1] >= ymin
            if ymax is not None:
                mask &= t[:, 1] <= ymax
            if zmin is not None:
                mask &= t[:, 2] >= zmin
            if zmax is not None:
                mask &= t[:, 2] <= zmax
            
            t = t[mask]
            
            if not len(t):
                continue
            
            lines = vedo.shapes.Lines(start_pts=t[:-1, :3], end_pts=t[1:, :3], c=color, lw=line_width)
            self.to_plot.append(lines)

    def plot_charge_density(self, excitation: Excitation, field: FieldBEM, color_map: str = 'coolwarm') -> None:
        """Plot charge density using the Vedo library.
        
        Parameters
        ---------------------
        excitation: `voltrace.excitation.Excitation`
            Excitation applied
        field: `voltrace.field.FieldBEM`
            Field that resulted after solving for the applied excitation
        color_map: str
            Name of the color map to use to color the charge density values
        """
        mesh = excitation.mesh
        
        if not len(mesh.triangles) and not len(mesh.lines):
            raise RuntimeError("Trying to plot empty mesh.")
        
        if len(mesh.triangles):
            assert isinstance(field, Field3D_BEM)
            meshes = _get_vedo_charge_density_3d(excitation, field, color_map)
            self.to_plot.append(meshes)
            
        if len(mesh.lines):
            assert isinstance(field, FieldRadialBEM)
            lines = _get_vedo_charge_density_2d(excitation, field, color_map)
            self.to_plot.append(lines)
         
        self.is_2d &= mesh.is_2d()
    
    def show(self) -> None:
        """Show the figure."""
        plotter = vedo.Plotter() 

        for t in self.to_plot:
            plotter += t
        
        if self.show_legend:
            lb = vedo.LegendBox(self.legend_entries)
            plotter += lb
        
        if self.is_2d:
            plotter.add_global_axes(dict(number_of_divisions=[12, 0, 12], zxgrid=True, xaxis_rotation=90))
        else:
            plotter.add_global_axes(dict(number_of_divisions=[10, 10, 10]))
        
        plotter.look_at(plane='xz')
        plotter.show()

def new_figure(show_legend: bool = True) -> Figure:
    """Create a new figure and make it the current figure.
    
    Parameters
    ----------------------
    show_legend: bool
        Whether to show the legend in the corner of the figure

    Returns
    ----------------------
    `Figure`"""
    global _current_figures
    f = Figure(show_legend=show_legend)
    _current_figures.append(f)
    return f

def get_current_figure() -> Figure:
    """Get the currently active figure. If no figure has been created yet
    a new figure will be returned.

    Returns
    --------------------
    `Figure`"""
    if len(_current_figures):
        return _current_figures[-1]
    
    return new_figure()

def plot_mesh(*args: Any, **kwargs: Any) -> None:
    """Calls `Figure.plot_mesh` on the current `Figure`"""
    get_current_figure().plot_mesh(*args, **kwargs)

def plot_charge_density(*args: Any, **kwargs: Any) -> None:
    """Calls `Figure.plot_charge_density` on the current `Figure`"""
    get_current_figure().plot_charge_density(*args, **kwargs)

def plot_equipotential_lines(*args: Any, **kwargs: Any) -> None:
    """Calls `Figure.plot_equipotential_lines` on the current `Figure`"""
    get_current_figure().plot_equipotential_lines(*args, **kwargs)

def plot_trajectories(*args: Any, **kwargs: Any) -> None:
    """Calls `Figure.plot_trajectories` on the current `Figure`"""
    get_current_figure().plot_trajectories(*args, **kwargs)

def show() -> None:
    """Calls `Figure.show` on the current `Figure`"""
    global _current_figures
        
    for f in _current_figures:
        f.show()

    _current_figures = []

def _get_vedo_grid(field: Field, surface: Surface, N0: int, N1: int) -> vedo.Grid:
    x = np.linspace(0, surface.parameter_range1, N0)
    y = np.linspace(0, surface.parameter_range2, N1)

    grid = vedo.Grid(s=(x, y))
    points = np.array([surface(x_, y_) for x_, y_, _ in grid.vertices])
    grid.vertices = points
    grid.pointdata['z'] = np.array([field.potential_at_point(p) for p in points])
    grid.lw(0)

    return grid
    
def _create_point_to_physical_dict(mesh: Mesh) -> dict[int, str]:
    d: dict[int, str] = {}
    
    for physical, elements in [(mesh.physical_to_lines, mesh.lines), (mesh.physical_to_triangles, mesh.triangles)]:
        for k, v in physical.items():
            for element_index in v:
                for p in elements[element_index]:
                    d[p] = k
    return d

def _get_vedo_triangles_and_normals(mesh: Mesh, **phys_colors: str) -> tuple[list[vedo.Mesh], vedo.shapes.Arrows]:
    triangles = mesh.triangles[:, :3]
    normals = np.array([backend.normal_3d(mesh.points[t]) for t in triangles])
    
    colors = np.full(len(triangles), '#CCCCCC')
    dict_ = _create_point_to_physical_dict(mesh)
    
    for i, (A, B, C) in enumerate(cast(List[Tuple[int,int,int]], triangles)):
        if A in dict_ and B in dict_ and C in dict_:
            phys1, phys2, phys3 = dict_[A], dict_[B], dict_[C]
            if phys1 == phys2 and phys2 == phys3 and phys1 in phys_colors:
                colors[i] = phys_colors[phys1]
     
    meshes = []
    
    for c in set(colors):
        mask = colors == c
        vm = vedo.Mesh([mesh.points, triangles[mask]], c)
        vm.linecolor('black').linewidth(2) # type: ignore
        
        key = [k for k, col in phys_colors.items() if c==col]
        if len(key):
            vm.legend(key[0])
        
        meshes.append(vm)
    
    start_to_end = np.zeros( (len(triangles), 6) )
    for i, t in enumerate(triangles):
        v1, v2, v3 = mesh.points[t]
        middle = (v1 + v2 + v3)/3
        area = 1/2*np.linalg.norm(np.cross(v2-v1, v3-v1))
        side_length = sqrt( (4*area) / sqrt(3) ) # Equilateral triangle, side length with equal area
        normal = 0.75*side_length*normals[i]
        start_to_end[i] = [*middle, *(middle+normal)]
        
    arrows = vedo.shapes.Arrows(start_to_end[:, :3], start_to_end[:, 3:], res=20, c='black')
     
    return meshes, arrows

# TODO: Move to Voltrace Pro
def _get_vedo_charge_density_3d(excitation: Excitation, field: Field3D_BEM, color_map: str) -> list[vedo.Mesh]:
    
    if excitation.is_electrostatic():
        all_vertices, name = excitation.get_electrostatic_active_elements()
        all_charges = field.electrostatic_point_charges.charges
    else:
        all_vertices, name = excitation.get_magnetostatic_active_elements()
        all_charges = field.magnetostatic_point_charges.charges
    
    charge_min, charge_max = np.min(all_charges), np.max(all_charges)
    
    meshes = []
    
    for _, indices in name.items():
        vertices = all_vertices[indices, :3]
        
        points = np.reshape(vertices, (3*len(vertices), 3))
        p_indices = np.arange(3*len(vertices)).reshape( (len(vertices), 3) )
        vm = vedo.Mesh([points, p_indices])
        vm.linewidth(0)
        vm.cellcolors = 255*vedo.colors.color_map(all_charges[indices], name=color_map, vmin=charge_min, vmax=charge_max)
        meshes.append(vm)
    
    return meshes

def _get_vedo_charge_density_2d(excitation: Excitation, field: FieldRadialBEM, color_map: str) -> list[vedo.Lines]:
    if excitation.is_electrostatic():
        all_vertices, name = excitation.get_electrostatic_active_elements()
        all_charges = field.electrostatic_point_charges.charges
    else:
        all_vertices, name = excitation.get_magnetostatic_active_elements()
        all_charges = field.magnetostatic_point_charges.charges
    
    assert len(all_vertices) == len(all_charges)
    charge_min, charge_max = np.min(all_charges), np.max(all_charges)
     
    lines = []
    
    for _, indices in name.items():
        vertices = all_vertices[indices]
        start_points = vertices[:, 0]
        end_points = vertices[:, 1]
        l = vedo.Lines(start_points, end_points, lw=3)
        colors = np.repeat(all_charges[indices], 2)
        l.cmap(color_map, colors, vmin=charge_min, vmax=charge_max)
        lines.append(l)

    return lines
    

def _get_vedo_lines_and_normals(mesh: Mesh, **phys_colors: str) -> tuple[list[vedo.Lines], vedo.shapes.Arrows]:
    lines = mesh.lines[:, :2]
    start = np.zeros( (len(lines), 3) )
    end = np.zeros( (len(lines), 3) )
    colors_ = np.zeros(len(lines), dtype=object) 
    normals = []
    dict_ = _create_point_to_physical_dict(mesh)
    points = mesh.points 
     
    for i, (A, B) in enumerate(cast(List[Tuple[int, int]], lines)):
            color = '#CCCCC'
            
            if A in dict_ and B in dict_:
                phys1, phys2 = dict_[A], dict_[B]
                if phys1 == phys2 and phys1 in phys_colors:
                    color = phys_colors[phys1]
            
            p1, p2 = points[A], points[B]
            start[i] = p1
            end[i] = p2
            colors_[i] = color

            normal = backend.normal_2d(np.array([p1[0], p1[2]]), np.array([p2[0], p2[2]])) 
            normals.append(np.array([normal[0], 0.0, normal[1]]))
     
    vedo_lines = []
     
    for c in set(colors_):
        mask = colors_ == c
        l = vedo.Lines(start[mask], end[mask], lw=3, c=c)
        
        key = [k for k, col in phys_colors.items() if c==col]
        if len(key):
            l.legend(key[0])
         
        vedo_lines.append(l)
     
    arrows_to_plot = np.zeros( (len(normals), 6) )
    
    for i, (v1, v2) in enumerate(zip(start, end)):
        middle = (v1 + v2)/2
        length = np.linalg.norm(v2-v1)
        normal = 3*length*normals[i]
        arrows_to_plot[i] = [*middle, *(middle+normal)]
    
    arrows = vedo.shapes.Arrows(arrows_to_plot[:, :3], arrows_to_plot[:, 3:], c='black')
     
    return vedo_lines, arrows
    

