#include <assert.h>
#include <time.h>
#include <math.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

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
		return log(xmax) - log(xmin);
	}
    return asinh(xmax/denom) - asinh(xmin/denom);
}

double
triangle_area(double v0[3], double v1[3], double v2[3]) {
	double vec1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
	double vec2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
	
	double out[3];
	cross_product_3d(vec1, vec2, out);
	return norm_3d(out[0], out[1], out[2])/2.0;
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

	gsl_function F;
	F.function = _potential_integrand;
	F.params = (void*) &tri;

	double result, error;
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(ADAPTIVE_MAX_ITERATION);
    gsl_integration_qag(&F, 0, c, 1e-9, 1e-9, ADAPTIVE_MAX_ITERATION, GSL_INTEG_GAUSS31, w, &result, &error);
	gsl_integration_workspace_free(w);

	return result;
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
	
	gsl_function F;
	F.function = _flux_integrand;
	F.params = (void*) &tri;
	
	double result, error;
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(ADAPTIVE_MAX_ITERATION);
    gsl_integration_qag(&F, 0, c, 1e-9, 1e-9, ADAPTIVE_MAX_ITERATION, GSL_INTEG_GAUSS31, w, &result, &error);
	gsl_integration_workspace_free(w);
	
	return result;
}





	












