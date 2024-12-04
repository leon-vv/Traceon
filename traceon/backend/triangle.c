#include <assert.h>
#include <time.h>
#include <math.h>

// To efficiently compute the double integrals we define
// a coordinate system as follows.
// Let v0, v1, v2 be the vertices of the source triangle
// Let p be the target point at which the potential (or flux)
// needs to be calculated.
// The x-axis and y-axis are orthogonal and lie in the plane
// of the triangle. The x-axis is aligned with v1-v0.
// The z-axis is perpendicular to the triangle and forms a right
// handed coordinate system with x,y.
// The origin of the coordinate system is the projection of the
// p on the plane of the triangle, and in the new coordinate
// system p can therefore be written as (0, 0, z).
// v0 can be written as (x0, y0, 0)
// v1 can be written as (x0 + a, y0, 0)
// v2 can be written as (x0 + b, y0 + c, 0)
// Therefore the whole problem can be expressed in x0,y0,a,b,c,z

struct _normalized_triangle {
	double x0;
	double y0;

	double a;
	double b;
	double c;
	double z;
	
	double *normal;
};

double _potential_integrand(double y, void *args_p) {
	struct _normalized_triangle args = *(struct _normalized_triangle*)args_p;
	double xmin = args.x0 + y/args.c*args.b;
	double xmax = args.x0 + args.a + y/args.c*(args.b-args.a);
		
	double denom = sqrt((y+args.y0)*(y+args.y0) + args.z*args.z);

	if(denom < 1e-12) {
		// The asinh(xmax/denom) - asinh(xmin/denom) is numerical 
		// unstable when denom is small. Taking the taylor expansion
		// of denom -> 0 we find
		return log(fabs(xmax)) - log(fabs(xmin));
	}
    return asinh(xmax/denom) - asinh(xmin/denom);
}

EXPORT double
potential_triangle(double v0[3], double v1[3], double v2[3], double target[3]) {
	double vec1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
	double vec2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};

	double x_normal[3] = {vec1[0], vec1[1], vec1[2]};
	double a = norm_3d(x_normal[0], x_normal[1], x_normal[2]);
	normalize_3d(x_normal);
	
	double z_normal[3];
	cross_product_3d(vec1, vec2, z_normal);
	normalize_3d(z_normal);

	double y_normal[3];
	cross_product_3d(z_normal, x_normal, y_normal);

	double to_v0[3] = {v0[0]-target[0], v0[1]-target[1], v0[2]-target[2]};
		
	double x0 = dot_3d(to_v0, x_normal);
    double y0 = dot_3d(to_v0, y_normal);
    double b = dot_3d(vec2, x_normal);
    double c = dot_3d(vec2, y_normal);
    double z = -dot_3d(z_normal, to_v0);
	
	struct _normalized_triangle tri = {x0, y0, a,b,c,z};

	return kronrod_adaptive(_potential_integrand, 0, c, (void*) &tri, 1e-9, 1e-9);
}

EXPORT double self_potential_triangle_v0(double v0[3], double v1[3], double v2[3]) {
	double vec1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
	double vec2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};

	double x_normal[3] = {vec1[0], vec1[1], vec1[2]};
	double a = norm_3d(x_normal[0], x_normal[1], x_normal[2]);
	normalize_3d(x_normal);
	
	double z_normal[3];
	cross_product_3d(vec1, vec2, z_normal);
	normalize_3d(z_normal);

	double y_normal[3];
	cross_product_3d(z_normal, x_normal, y_normal);

    double b = dot_3d(vec2, x_normal);
    double c = dot_3d(vec2, y_normal);
		
    double alpha = b / c;
    double beta = (b - a) / c;

    return -((-a * asinh((beta * beta * c + c + beta * a) / a)) +
             sqrt(beta * beta + 1) * (c * asinh((beta * c + a) / c) - asinh(alpha) * c) +
             asinh(beta) * a) / sqrt(beta * beta + 1);
}

EXPORT double self_potential_triangle(double v0[3], double v1[3], double v2[3], double target[3]) {

	return 
		self_potential_triangle_v0(target, v0, v1) +
		self_potential_triangle_v0(target, v1, v2) +
		self_potential_triangle_v0(target, v2, v0);
}

double _flux_integrand(double y, void *args_p) {
	struct _normalized_triangle args = *(struct _normalized_triangle*)args_p;
	
	double x0 = args.x0;
	double y0 = args.y0;
	double z = args.z;
	
	double xmin = x0 + y/args.c*args.b;
	double xmax = x0 + args.a + y/args.c*(args.b-args.a);
	
	double z2 = z*z;
	double xmin2 = xmin*xmin;
	double yy02 = (y+y0)*(y+y0);
	double xmax2 = xmax*xmax;
	double r2 = z2 + yy02;

	double flux[3];

	flux[0] = 1 / sqrt(r2 + xmax2) - 1 / sqrt(r2 + xmin2);

	// Singularity when r2 is small...
	if (fabs(r2) < 1e-9) {
		flux[1] = ((xmin2 - xmax2) * y0 + (xmin2 - xmax2) * y) / (2.0 * xmax2 * xmin2);
		flux[2] = -((xmin2 - xmax2) * z) / (2.0 * xmax2 * xmin2);
	} else {
		double denom_max = r2 * sqrt(r2 + xmax2);
		double denom_min = r2 * sqrt(r2 + xmin2);

		flux[1] = -((xmax * (y + y0)) / denom_max) + (xmin * (y + y0)) / denom_min;
		flux[2] = (xmax * z) / denom_max - (xmin * z) / denom_min;
	}
	
	return dot_3d(args.normal, flux);
}



EXPORT double
flux_triangle(double v0[3], double v1[3], double v2[3], double target[3], double normal[3]) {
	double vec1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
	double vec2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};

	double x_normal[3] = {vec1[0], vec1[1], vec1[2]};
	double a = norm_3d(x_normal[0], x_normal[1], x_normal[2]);
	normalize_3d(x_normal);
	
	double z_normal[3];
	cross_product_3d(vec1, vec2, z_normal);
	normalize_3d(z_normal);

	double y_normal[3];
	cross_product_3d(z_normal, x_normal, y_normal);

	double to_v0[3] = {v0[0]-target[0], v0[1]-target[1], v0[2]-target[2]};
		
	double x0 = dot_3d(to_v0, x_normal);
    double y0 = dot_3d(to_v0, y_normal);
    double b = dot_3d(vec2, x_normal);
    double c = dot_3d(vec2, y_normal);
    double z = -dot_3d(z_normal, to_v0);

	// Express normal in new coordinate system
	double new_normal[3] = {dot_3d(x_normal, normal),
							dot_3d(y_normal, normal),
							dot_3d(z_normal, normal)};
		
	struct _normalized_triangle tri = {x0, y0, a, b, c, z, new_normal};
	
	return kronrod_adaptive(_flux_integrand, 0, c, (void*) &tri, 1e-9, 1e-9);
}


EXPORT int triangle_orientation_is_equal(int triangle_index1, int triangle_index2, uint64_t triangles[][3], double points[][3]) {
    // Imagine standing at a common vertex, and looking along the common edge
    // One way to define a normal, would be to draw a line from your position
    // to the midpoint, and taking a cross product with the edge.
    // In this way, we can define normals that point in the same direction.
    // To check if the normals that follow from the ordering of the triangles
    // agree, we can compare the normals as computed with the procedure explained.

    uint64_t *t1 = triangles[triangle_index1];
    uint64_t *t2 = triangles[triangle_index2];

    double triangle1[3][3] = { {points[t1[0]][0], points[t1[0]][1], points[t1[0]][2]},
                               {points[t1[1]][0], points[t1[1]][1], points[t1[1]][2]},
                               {points[t1[2]][0], points[t1[2]][1], points[t1[2]][2]} };

    double triangle2[3][3] = { {points[t2[0]][0], points[t2[0]][1], points[t2[0]][2]},
                               {points[t2[1]][0], points[t2[1]][1], points[t2[1]][2]},
                               {points[t2[2]][0], points[t2[2]][1], points[t2[2]][2]} };

    double mid1[3] = {(triangle1[0][0] + triangle1[1][0] + triangle1[2][0]) / 3.0,
                      (triangle1[0][1] + triangle1[1][1] + triangle1[2][1]) / 3.0,
                      (triangle1[0][2] + triangle1[1][2] + triangle1[2][2]) / 3.0};

    double mid2[3] = {(triangle2[0][0] + triangle2[1][0] + triangle2[2][0]) / 3.0,
                      (triangle2[0][1] + triangle2[1][1] + triangle2[2][1]) / 3.0,
                      (triangle2[0][2] + triangle2[1][2] + triangle2[2][2]) / 3.0};

    double normal1[3], normal2[3];
    normal_3d(triangle1, normal1);
    normal_3d(triangle2, normal2);

    // Find the common vertices between the two triangles
	int common_index1, common_index2;
	bool common_index_found = false;
	
    for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++) {
		if (t1[i] == t2[j]) {
			common_index1 = i;
			common_index2 = j;
			common_index_found = true;
		}
	}
	
	if(!common_index_found) return -1;
	
    double *common_vertex = points[t1[common_index1]];
	
	// Find the common edge
	double edges[4][3];
	int edge_count = 0;
	
	for(int i = 0; i < 3; i++) {
		if(i == common_index1) continue;
		edges[edge_count][0] = points[t1[i]][0] - common_vertex[0];
		edges[edge_count][1] = points[t1[i]][1] - common_vertex[1];
		edges[edge_count][2] = points[t1[i]][2] - common_vertex[2];
		edge_count += 1;
	}
	for(int j = 0; j < 3; j++) {
		if(j == common_index2) continue;
		edges[edge_count][0] = points[t2[j]][0] - common_vertex[0];
		edges[edge_count][1] = points[t2[j]][1] - common_vertex[1];
		edges[edge_count][2] = points[t2[j]][2] - common_vertex[2];
		edge_count += 1;
	}

	assert(edge_count == 4);

	for(int e = 0; e < 4; e++) normalize_3d(&edges[e][0]);
	
	// Two edges are equal if their dot product is one
	double *common_edge = NULL;

	for(int i = 0; i < 2; i++)
	for(int j = 0; j < 2; j++) {
		double d = dot_3d(edges[i], edges[2 + j]);
		if(fabs(d - 1.0) < 1e-14) common_edge = &edges[i][0];
	}

	if(common_edge == NULL) return -1;
		
    double to_mid1[3] = {mid1[0] - common_vertex[0], mid1[1] - common_vertex[1], mid1[2] - common_vertex[2]};
    double to_mid2[3] = {mid2[0] - common_vertex[0], mid2[1] - common_vertex[1], mid2[2] - common_vertex[2]};

    double upward_normal1[3], upward_normal2[3];
    cross_product_3d(to_mid1, common_edge, upward_normal1);
    cross_product_3d(common_edge, to_mid2, upward_normal2);

    double dot1 = dot_3d(upward_normal1, normal1);
    double dot2 = dot_3d(upward_normal2, normal2);

    return (dot1 > 0.0) == (dot2 > 0.0);
}












