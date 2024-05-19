#include <assert.h>
#include <time.h>
#include <math.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

// Triangle is defined by the vertices
// (0, 0, 0), (a, 0, 0), (b, c, 0)
// the target is located at (0, 0, z0)
// this function returns the potential at the target for a triangle
// with unit charge density. One adaptive integration is needed.

// If the triangle has no area the code is numerically unstable.
// So we check that first

struct _normalized_triangle {
	double a;
	double b;
	double c;
	double z0;
	double normal[3];
};


double _potential_normalized_triangle_integrand(double theta, void *args_p) {
	struct _normalized_triangle *args = (struct _normalized_triangle*)args_p;

	double a = args->a; double b = args->b; double c = args->c;
	double z0 = args->z0;
	
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
	
	// If the triangle has no area the code is numerically unstable.
	// So we check that first

	struct _normalized_triangle tri = { a, b, c, z0 };

	if( fabs(tri.a*tri.c) < 1e-12 ) return 0.;
	
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(ADAPTIVE_MAX_ITERATION);
	
	gsl_function F;
	F.function = _potential_normalized_triangle_integrand;
	F.params = (void*) &tri;
	
	double result, error;
	gsl_integration_qags(&F, 0, atan2(tri.c, tri.b), 0, 1e-8, ADAPTIVE_MAX_ITERATION, w, &result, &error);
    gsl_integration_workspace_free(w);
	return result;
}

double _flux_normalized_triangle_integrand(double theta, void *args_p) {
	struct _normalized_triangle *args = (struct _normalized_triangle*)args_p;
	
	double a = args->a; double b = args->b; double c = args->c;
	double z0 = args->z0;
	
	double rmax = -a*c/((b-a)*sin(theta)-c*cos(theta));

	double z02 = z0*z0;
	double rmax2 = rmax*rmax;
		
	double dx = -(rmax*sqrt(z02+rmax2)*cos(theta)+(-asinh(rmax/z0)*z02-rmax2*asinh(rmax/z0))*cos(theta))/(z02+rmax2);
	double dy = -(rmax*sqrt(z02+rmax2)*sin(theta)+(-asinh(rmax/z0)*z02-rmax2*asinh(rmax/z0))*sin(theta))/(z02+rmax2);
	double dz = -(sqrt(z02+rmax2)-z0)/sqrt(z02+rmax2);
		
	return dx*args->normal[0] + dy*args->normal[1] + dz*args->normal[2];
}

double flux_normalized_triangle(double a, double b, double c, double z0, double normal[3]) {
	// See comment in potential_normalized_triangle

	struct _normalized_triangle tri = { a, b, c, z0, {normal[0], normal[1], normal[2]} };
	
	if( fabs(tri.a*tri.c) < 1e-12 ) return 0.;
		
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(ADAPTIVE_MAX_ITERATION);
	
	gsl_function F;
	F.function = _flux_normalized_triangle_integrand;
	F.params = (void*) &tri;
		
	double result, error;
	gsl_integration_qags(&F, 0, atan2(tri.c, tri.b), 0, 1e-8, ADAPTIVE_MAX_ITERATION, w, &result, &error);
    gsl_integration_workspace_free(w);
	return result;


}

void _express_triangle_in_local_coordinate_system(double *v0, double *v1, double *v2, double *target, double *normal, struct _normalized_triangle *out) {
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

	out->a = dot_3d(x_normal, p1);
	out->b = dot_3d(x_normal, p2);
	out->c = dot_3d(y_normal, p2); 
	out->z0 = dot_3d(z_normal, p3);

	if(normal != NULL) {
		out->normal[0] = dot_3d(x_normal, normal);
		out->normal[1] = dot_3d(y_normal, normal);
		out->normal[2] = dot_3d(z_normal, normal);
	}
}

double potential_triangle_target_over_v0(double *v0, double *v1, double *v2, double *target) {
	// General triangle, but the target has to lie on the line defined
	// by v0 and the normal to the triangle.
	struct _normalized_triangle tr;
	_express_triangle_in_local_coordinate_system(v0, v1, v2, target, NULL, &tr);
	return potential_normalized_triangle(tr.a, tr.b, tr.c, tr.z0);
}

double flux_triangle_target_over_v0(double *v0, double *v1, double *v2, double *target, double *normal) {
	// General triangle, but the target has to lie on the line defined
	// by v0 and the normal to the triangle.
	
	struct _normalized_triangle tr;
	_express_triangle_in_local_coordinate_system(v0, v1, v2, target, normal, &tr);
	return flux_normalized_triangle(tr.a, tr.b, tr.c, tr.z0, tr.normal);
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

double
triangle_area(double v0[3], double v1[3], double v2[3]) {
	double vec1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
	double vec2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
	
	double out[3];
	cross_product_3d(vec1, vec2, out);
	return norm_3d(out[0], out[1], out[2])/2.0;
}

inline int sign(double x) {
	return x > 0 ? 1 : -1;
}

double
potential_triangle(double v0[3], double v1[3], double v2[3], double target[3]) {
	double coords[3];
	triangle_barycentric_coords(target, v0, v1, v2, coords);
	
	double area = triangle_area(v0, v1, v2);
	
	// Project of point in the triangle
	// (a,b,c) = coords
    // pt = a*v0 + b*v1 + c*v2
	double pt[3] = {
		coords[0]*v0[0] + coords[1]*v1[0] + coords[2]*v2[0],
		coords[0]*v0[1] + coords[1]*v1[1] + coords[2]*v2[1],
		coords[0]*v0[2] + coords[1]*v1[2] + coords[2]*v2[2]};
		
	double pot1 = potential_triangle_target_over_v0(pt, v0, v1, target);
	double pot2 = potential_triangle_target_over_v0(pt, v1, v2, target);
	double pot3 = potential_triangle_target_over_v0(pt, v2, v0, target);
	
	return (sign(coords[2])*pot1 + sign(coords[0])*pot2 + sign(coords[1])*pot3)/(2*area);
}

double
flux_triangle(double v0[3], double v1[3], double v2[3], double target[3], double normal[3]) {
	double coords[3];
	triangle_barycentric_coords(target, v0, v1, v2, coords);
	
	double area = triangle_area(v0, v1, v2);
	
	// Project of point in the triangle
	// (a,b,c) = coords
    // pt = a*v0 + b*v1 + c*v2
	double pt[3] = {
		coords[0]*v0[0] + coords[1]*v1[0] + coords[2]*v2[0],
		coords[0]*v0[1] + coords[1]*v1[1] + coords[2]*v2[1],
		coords[0]*v0[2] + coords[1]*v1[2] + coords[2]*v2[2]};
		
	double flux1 = flux_triangle_target_over_v0(pt, v0, v1, target, normal);
	double flux2 = flux_triangle_target_over_v0(pt, v1, v2, target, normal);
	double flux3 = flux_triangle_target_over_v0(pt, v2, v0, target, normal);
	
	return (sign(coords[2])*flux1 + sign(coords[0])*flux2 + sign(coords[1])*flux3)/(2*area);
}





	












