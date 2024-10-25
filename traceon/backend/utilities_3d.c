
typedef double (triangle)[3][3];

#define N_TRIANGLE_QUAD 12
EXPORT const int N_TRIANGLE_QUAD_SYM = N_TRIANGLE_QUAD;

const double QUAD_WEIGHTS[N_TRIANGLE_QUAD] =
 {0.0254224531851035,
  0.0254224531851035,
  0.0254224531851035,
  0.0583931378631895,
  0.0583931378631895,
  0.0583931378631895,
  0.041425537809187,
  0.041425537809187,
  0.041425537809187,
  0.041425537809187,
  0.041425537809187,
  0.041425537809187};

const double QUAD_B1[N_TRIANGLE_QUAD] =
 {0.873821971016996,
  0.063089014491502,
  0.063089014491502,
  0.501426509658179,
  0.249286745170910,
  0.249286745170910,
  0.636502499121399,
  0.636502499121399,
  0.310352451033785,
  0.310352451033785,
  0.053145049844816,
  0.053145049844816};

const double QUAD_B2[N_TRIANGLE_QUAD] =
 {0.063089014491502,
  0.873821971016996,
  0.063089014491502,
  0.249286745170910,
  0.501426509658179,
  0.249286745170910,
  0.310352451033785,
  0.053145049844816,
  0.636502499121399,
  0.053145049844816,
  0.636502499121399,
  0.310352451033785};












INLINE double
dot_3d(double v1[3], double v2[3]) {
	return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

INLINE double
norm_3d(double x, double y, double z) {
	return sqrt(x*x + y*y + z*z);
}

INLINE double
distance_3d(double v0[3], double v1[3]) {
	return norm_3d(v0[0]-v1[0], v0[1]-v1[1], v0[2]-v1[2]);
}

INLINE void
normalize_3d(double *v) {
	double length = norm_3d(v[0], v[1], v[2]);
	v[0] /= length;
	v[1] /= length;
	v[2] /= length;
	assert(fabs(norm_3d(v[0], v[1], v[2])-1.) < 1e-8);
}

INLINE void
cross_product_3d(double v1[3], double v2[3], double out[3]) {
	double v1x = v1[0], v1y = v1[1], v1z = v1[2];
	double v2x = v2[0], v2y = v2[1], v2z = v2[2];

	out[0] = v1y*v2z-v1z*v2y;
	out[1] = v1z*v2x-v1x*v2z;
	out[2] = v1x*v2y-v1y*v2x;
}

double
triangle_area(double v0[3], double v1[3], double v2[3]) {
	double vec1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
	double vec2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
	
	double out[3];
	cross_product_3d(vec1, vec2, out);
	return norm_3d(out[0], out[1], out[2])/2.0;
}

EXPORT void
normal_3d(double alpha, double beta, triangle t, double *normal) {
	double x1 = t[0][0], y1 = t[0][1], z1 = t[0][2];
	double x2 = t[1][0], y2 = t[1][1], z2 = t[1][2];
	double x3 = t[2][0], y3 = t[2][1], z3 = t[2][2];

	double normal_x = (y2-y1)*(z3-z1)-(y3-y1)*(z2-z1);
	double normal_y = (x3-x1)*(z2-z1)-(x2-x1)*(z3-z1);
	double normal_z = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1);
	double length = norm_3d(normal_x, normal_y, normal_z);
	
	normal[0] = normal_x/length;
	normal[1] = normal_y/length;
	normal[2] = normal_z/length;
}

INLINE void position_and_jacobian_3d(double alpha, double beta, triangle t, double pos_out[3], double *jac) {

	double v1[3] = {t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]};
	double v2[3] = {t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]};

	double x = t[0][0] + alpha*v1[0] + beta*v2[0];
	double y = t[0][1] + alpha*v1[1] + beta*v2[1];
	double z = t[0][2] + alpha*v1[2] + beta*v2[2];


	pos_out[0] = x;
	pos_out[1] = y;
	pos_out[2] = z;
	*jac = 2*triangle_area(t[0], t[1], t[2]);
}

// Compute E + v x B, which is used in the Lorentz force law to calculate the force
// on the particle. The magnetic field produced by magnetiziation and the magnetic field
// produced by currents are passed in separately, but can simpy be summed to find the total
// magnetic field.
EXPORT void
combine_elec_magnetic_field(double velocity[3], double elec_field[3],
		double mag_field[3], double current_field[3], double result[3]) {
		
	double total_mag[3] = {0.}; // Total magnetic field, produced by charges and currents
		
	// Important: Traceon always computes the H field
	// Therefore when converting from H to B we need to multiply
	// by mu_0.
	total_mag[0] = MU_0*(mag_field[0] + current_field[0]);
	total_mag[1] = MU_0*(mag_field[1] + current_field[1]);
	total_mag[2] = MU_0*(mag_field[2] + current_field[2]);
			
	double cross[3] = {0.};
		
	// Calculate v x B
	cross_product_3d(velocity, total_mag, cross);
	
	result[0] = elec_field[0] + cross[0];
	result[1] = elec_field[1] + cross[1];
	result[2] = elec_field[2] + cross[2];
}


