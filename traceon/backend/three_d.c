
typedef double (*jacobian_buffer_3d)[N_TRIANGLE_QUAD];
typedef double (*position_buffer_3d)[N_TRIANGLE_QUAD][3];

struct effective_point_charges_3d {
	double *charges;
	jacobian_buffer_3d jacobians;
	position_buffer_3d positions;
	size_t N;
};




