
INLINE double
norm_2d(double x, double y) {
	return sqrt(x*x + y*y);
}

INLINE double
length_2d(double *v1, double *v2) {
	return norm_2d(v2[0]-v1[0], v2[1]-v1[1]);
}

EXPORT void
normal_2d(double *p1, double *p2, double *normal) {
	double x1 = p1[0], y1 = p1[1];
	double x2 = p2[0], y2 = p2[1];
	
	double tangent_x = x2 - x1, tangent_y = y2 - y1;
	double normal_x = tangent_y, normal_y = -tangent_x;
	double length = norm_2d(normal_x, normal_y);

	normal[0] = normal_x/length;
	normal[1] = normal_y/length;
}

EXPORT void
higher_order_normal_radial(double alpha, double *v1, double *v2, double *v3, double *v4, double *normal) {

	double v1x = v1[0], v1y = v1[1];
	double v2x = v2[0], v2y = v2[1];
	double v3x = v3[0], v3y = v3[1];
	double v4x = v4[0], v4y = v4[1];
		
	double a2 = pow(alpha, 2);
	double a3 = pow(alpha, 3);
	
	double dx = (2*alpha*(9*v4x-9*v3x-9*v2x+9*v1x)+3*a2*(9*v4x-27*v3x+27*v2x-9*v1x)-v4x+27*v3x-27*v2x+v1x)/16;
	double dy = (2*alpha*(9*v4y-9*v3y-9*v2y+9*v1y)+3*a2*(9*v4y-27*v3y+27*v2y-9*v1y)-v4y+27*v3y-27*v2y+v1y)/16;

	double zero[2] = {0., 0.};
	double vec[2] = {dx, dy};
	normal_2d(zero, vec, normal);
}

INLINE void position_and_jacobian_radial(double alpha, double *v1, double *v2, double *v3, double *v4, double *pos_out, double *jac) {

	double v1x = v1[0], v1y = v1[1];
	double v2x = v2[0], v2y = v2[1];
	double v3x = v3[0], v3y = v3[1];
	double v4x = v4[0], v4y = v4[1];
		
	double a2 = pow(alpha, 2);
	double a3 = pow(alpha, 3);
	
	// Higher order line element parametrization. 
	pos_out[0] = (a2*(9*v4x-9*v3x-9*v2x+9*v1x)+a3*(9*v4x-27*v3x+27*v2x-9*v1x)-v4x+alpha*(-v4x+27*v3x-27*v2x+v1x)+9*v3x+9*v2x-v1x)/16;
	pos_out[1] = (a2*(9*v4y-9*v3y-9*v2y+9*v1y)+a3*(9*v4y-27*v3y+27*v2y-9*v1y)-v4y+alpha*(-v4y+27*v3y-27*v2y+v1y)+9*v3y+9*v2y-v1y)/16;
	
	// Term following from the Jacobian
	*jac = 1/16. * sqrt(pow(2*alpha*(9*v4y-9*v3y-9*v2y+9*v1y)+3*a2*(9*v4y-27*v3y+27*v2y-9*v1y)-v4y+27*v3y-27*v2y+v1y, 2) +pow(2*alpha*(9*v4x-9*v3x-9*v2x+9*v1x)+3*a2*(9*v4x-27*v3x+27*v2x-9*v1x)-v4x+27*v3x-27*v2x+v1x, 2));
}
