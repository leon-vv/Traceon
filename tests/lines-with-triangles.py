import traceon.geometry as G
import traceon.plotting as P


with G.Geometry(G.Symmetry.RADIAL) as geom:
    
    points = [ [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.5, 0.5] ]
    points = [geom.add_point(p) for p in points]
    
    l1 = geom.add_line(points[0], points[1])
    l2 = geom.add_line(points[1], points[2])
    l3 = geom.add_line(points[2], points[3])

    geom.add_physical([l1,l2,l3], 'lines')

    mesh1 = geom.generate_line_mesh(True)

with G.Geometry(G.Symmetry.RADIAL) as geom:
    points = [ [3.0, 4.0], [4.0, 4.0], [4.0, 2.0], [2.0, 2.0] ]
    p = geom.add_polygon(points)
    geom.add_physical(p.surface, 'triangles')
    mesh2 = geom.generate_triangle_mesh(False)

total_mesh = mesh1.remove_triangles() + mesh2.remove_lines()
P.plot_mesh(total_mesh, lines='blue', triangles='purple')
