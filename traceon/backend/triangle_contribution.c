#include <assert.h>
#include <time.h>
#include <math.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>


double _potential_normalized_triangle_integrand(double theta, void *args_p) {
	double *args = (double*)args_p;
	double a = args[0], b = args[1], c = args[2], z0 = args[3];
	assert(a > 0 && c > 0 && z0 >= 0);
		
	double rmax = -a*c/((b-a)*sin(theta) - c*cos(theta));
	assert(rmax >= 0.);
	return sqrt(z0*z0 + rmax*rmax) - fabs(z0);
}

double potential_normalized_triangle(double a, double b, double c, double z0) {
	// Triangle is defined by the vertices
	// (0, 0, 0), (a, 0, 0), (b, c, 0)
	// the target is located at (0, 0, z0)
	// this function returns the potential at the target for a triangle
	// with unit charge density. One adaptive integration is needed.
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(ADAPTIVE_MAX_ITERATION);
	
	double args[4] = {a,b,c,z0};
	
	gsl_function F;
	F.function = _potential_normalized_triangle_integrand;
	F.params = (void*)args;
	
	double result, error;
	gsl_integration_qags(&F, 0, atan2(c, b), 0, 1e-8, ADAPTIVE_MAX_ITERATION, w, &result, &error);
    gsl_integration_workspace_free(w);
	return result;
}

void _express_triangle_in_local_coordinate_system(double *v0, double *v1, double *v2, double *target, double out[4]) {
	// Define a local coordinate system.
	// The x normal is parallel to v0-v1
	// The z normal goes through v0 and the target, ensuring that the target lives at positive z0
	// The y is defined to make the CS right handed
	// 
	// we then compute a,b,c,z0 (see function above)
	
	// Check if in the resulting coordinate system z0 ends up positive, otherwise swap v1 and v2
	double z_direction[3];
	
	double p1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
	double p2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
	double p3[3] = {target[0]-v0[0], target[1]-v0[1], target[2]-v0[2]};
	
	cross_product_3d(p1, p2, z_direction);
	
	if(dot_3d(z_direction, p3) < 0.) {
		// Swap v1 and v2
		p1[0] = v2[0]-v0[0]; p1[1] = v2[1]-v0[1]; p1[2] = v2[2]-v0[2];
		p2[0] = v1[0]-v0[0]; p2[1] = v1[1]-v0[1]; p2[2] = v1[2]-v0[2];
	}
	
	double x_normal[3] = {p1[0], p1[1], p1[2]};
	
	double y_normal[3], z_normal[3];
	cross_product_3d(x_normal, p2, z_normal);
	cross_product_3d(z_normal, x_normal, y_normal);
		
	normalize_3d(x_normal); 
	normalize_3d(y_normal); 
	normalize_3d(z_normal);

	double a = dot_3d(x_normal, p1);
	double b = dot_3d(x_normal, p2);
	double c = dot_3d(y_normal, p2); 
	double z0 = dot_3d(z_normal, p3);

	out[0] = a; out[1] = b; out[2] = c; out[3] = z0;
}

double potential_triangle_target_over_v0(double *v0, double *v1, double *v2, double *target) {
	// General triangle, but the target has to lie on the line defined
	// by v0 and the normal to the triangle.
	double abcz0[4];
	_express_triangle_in_local_coordinate_system(v0, v1, v2, target, abcz0);
	double a = abcz0[0], b = abcz0[1], c = abcz0[2], z0 = abcz0[3];
	return potential_normalized_triangle(a,b,c,z0);
}

// Get the barycentric coordinates of
// the projection of the point on the plane of the triangle
void triangle_barycentric_coords(double p[3], double v0[3], double v1[3], double v2[3], double out[3]) {
	double x[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
	double y[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
	
    double vx = dot_3d(v0, x);
    double vy = dot_3d(v0, y);
    double px = dot_3d(p, x);
    double py = dot_3d(p, y);
    double x2 = dot_3d(x, x);
    double xy = dot_3d(x, y);
    double y2 = dot_3d(y, y);
		
	double denom = (x2*y2 - xy*xy);
	double a = -((vx-px)*y2+(py-vy)*xy)/denom;
    double b = ((vx-px)*xy+(py-vy)*x2)/denom;

	out[0] = 1-a-b;
	out[1] = a;
	out[2] = b;
}












