
#define MIN_DISTANCE_AXIS 1e-10
#define NU_MAX 4
#define M_MAX 8

// DERIV_2D_MAX, NU_MAX_SYM and M_MAX_SYM need to be present in the .so file to
// be able to read them. We cannot call them NU_MAX and M_MAX as
// the preprocessor will substitute their names. We can also not 
// simply only use these symbols instead of the preprocessor variables
// as the length of arrays need to be a compile time constant in C...
EXPORT const int NU_MAX_SYM = NU_MAX;
EXPORT const int M_MAX_SYM = M_MAX;

typedef double (*jacobian_buffer_3d)[N_TRIANGLE_QUAD];
typedef double (*position_buffer_3d)[N_TRIANGLE_QUAD][3];
typedef double (*vertices_3d)[3][3];

struct effective_point_charges_3d {
	double *charges;
	jacobian_buffer_3d jacobians;
	position_buffer_3d positions;
	size_t N;
};

struct field_derivs_args {
	double *z_interpolation;
	double *electrostatic_axial_coeffs;
	double *magnetostatic_axial_coeffs;
	size_t N_z;
};



EXPORT double  
potential_3d(double point[3], double *charges, jacobian_buffer_3d jacobian_buffer, position_buffer_3d position_buffer, size_t N_vertices) {  
	
	double sum_ = 0.0;  
	
	for(int i = 0; i < N_vertices; i++) {  
		for(int k = 0; k < N_TRIANGLE_QUAD; k++) {
			double *pos = &position_buffer[i][k][0];
			double potential = potential_3d_point(point[0], point[1], point[2], pos[0], pos[1], pos[2], NULL);
			
			sum_ += charges[i] * jacobian_buffer[i][k] * potential;
		}
	}  
	
	return sum_;
}  

double
field_dot_normal_3d(double x0, double y0, double z0, double x, double y, double z, void* normal_p) {
	
	double Ex = -dx1_potential_3d_point(x0, y0, z0, x, y, z, NULL);
	double Ey = -dy1_potential_3d_point(x0, y0, z0, x, y, z, NULL);
	double Ez = -dz1_potential_3d_point(x0, y0, z0, x, y, z, NULL);
	
	double *normal = (double *)normal_p;
	
    return normal[0]*Ex + normal[1]*Ey + normal[2]*Ez;
}

EXPORT void
field_3d(double point[3], double result[3], double *charges,
	jacobian_buffer_3d jacobian_buffer, position_buffer_3d position_buffer, size_t N_vertices) {

	double Ex = 0.0, Ey = 0.0, Ez = 0.0;

	for(int i = 0; i < N_vertices; i++) {
		for(int k = 0; k < N_TRIANGLE_QUAD; k++) {
			double *pos = &position_buffer[i][k][0];
			double field_x = dx1_potential_3d_point(point[0], point[1], point[2], pos[0], pos[1], pos[2], NULL);
			double field_y = dy1_potential_3d_point(point[0], point[1], point[2], pos[0], pos[1], pos[2], NULL);
			double field_z = dz1_potential_3d_point(point[0], point[1], point[2], pos[0], pos[1], pos[2], NULL);
		
			Ex -= charges[i] * jacobian_buffer[i][k] * field_x;
			Ey -= charges[i] * jacobian_buffer[i][k] * field_y;
			Ez -= charges[i] * jacobian_buffer[i][k] * field_z;
		}
	}
		
	result[0] = Ex;
	result[1] = Ey;
	result[2] = Ez;
}



EXPORT void
axial_coefficients_3d(double *restrict charges,
	jacobian_buffer_3d restrict jacobian_buffer,
	position_buffer_3d restrict position_buffer,
	double *trig_cos_buffer_p, double *trig_sin_buffer_p,
	size_t N_v,
	double *restrict zs, double *restrict output_coeffs_p, size_t N_z) {
		
	double (*output_coeffs)[2][NU_MAX][M_MAX] = (double (*)[2][NU_MAX][M_MAX]) output_coeffs_p;
		
	double (*trig_cos_buffer)[N_TRIANGLE_QUAD][M_MAX] = (double (*)[N_TRIANGLE_QUAD][M_MAX]) trig_cos_buffer_p;
	double (*trig_sin_buffer)[N_TRIANGLE_QUAD][M_MAX] = (double (*)[N_TRIANGLE_QUAD][M_MAX]) trig_sin_buffer_p;

	double factorial[NU_MAX][M_MAX] = {
		{1.0,1.0,0.5,0.1666666666666666,0.04166666666666666,0.008333333333333334,0.001388888888888889,1.984126984126984E-4},
		{0.5,0.1666666666666666,0.04166666666666666,0.008333333333333334,0.001388888888888889,1.984126984126984E-4,2.48015873015873E-5,2.755731922398589E-6},
		{0.04166666666666666,0.008333333333333334,0.001388888888888889,1.984126984126984E-4,2.48015873015873E-5,2.755731922398589E-6,2.755731922398589E-7,2.505210838544172E-8},
		{0.001388888888888889,1.984126984126984E-4,2.48015873015873E-5,2.755731922398589E-6,2.755731922398589E-7,2.505210838544172E-8,2.08767569878681E-9,1.605904383682161E-10}};
		
	for(int h = 0; h < N_v; h++)
	for(int k = 0; k < N_TRIANGLE_QUAD; k++)
	for(int m = 0; m < M_MAX; m++) {
		
		double x = position_buffer[h][k][0];
		double y = position_buffer[h][k][1];
		double mu = atan2(y, x);
			
		// The integration factor needs to be adjusted for m=0, since the
		// cos(m*phi) term in the integral vanishes.
		trig_cos_buffer[h][k][m] = (1./M_PI) * cos(m*mu) * (m == 0 ? 1/2. : 1.);
		trig_sin_buffer[h][k][m] = (1./M_PI) * sin(m*mu);
	}
		
	for (int i=0; i < N_z; i++) 
	for(int h = 0; h < N_v; h++)
	for (int k=0; k < N_TRIANGLE_QUAD; k++) {
		double x = position_buffer[h][k][0];
		double y = position_buffer[h][k][1];
		double z = position_buffer[h][k][2];
		
		double r = 1/norm_3d(x, y, z-zs[i]);
		double p = (z-zs[i]) / norm_2d(x, y);
		
		double p2 = pow(p, 2);
		double p4 = pow(p, 4);
		double p6 = pow(p, 6);
		double p8 = pow(p, 8);
		double p10 = pow(p, 10);
		double p12 = pow(p, 12);
		double p14 = pow(p, 14);
		double sqrt_p2_plus1 = sqrt(p2+1);
		
		// Output base values, without cos, sin dependence
		double output_base[NU_MAX][M_MAX] = {
			{1./2.,
			1./(4*sqrt_p2_plus1),
			(3)/((8*p2+8)),
			(15)/(sqrt_p2_plus1*(16*p2+16)),
			(105)/((32*p4+64*p2+32)),
			(945*sqrt_p2_plus1)/((64*p6+192*p4+192*p2+64)),
			(10395)/((128*p6+384*p4+384*p2+128)),
			(135135*sqrt_p2_plus1)/((256*p8+1024*p6+1536*p4+1024*p2+256))},
			
			{-(2*p2-1)/((4*p2+4)),
			-(36*p2-9)/(sqrt_p2_plus1*(16*p2+16)),
			-(90*p2-15)/((8*p4+16*p2+8)),
			-(sqrt_p2_plus1*(4200*p2-525))/((64*p6+192*p4+192*p2+64)),
			-(28350*p2-2835)/((64*p6+192*p4+192*p2+64)),
			-(sqrt_p2_plus1*(873180*p2-72765))/((256*p8+1024*p6+1536*p4+1024*p2+256)),
			-(1891890*p2-135135)/((64*p8+256*p6+384*p4+256*p2+64)),
			-(291891600*p2-18243225)/(sqrt_p2_plus1*(1024*p8+4096*p6+6144*p4+4096*p2+1024))},
			
			{(72*p4-216*p2+27)/((16*p4+32*p2+16)),
			(sqrt_p2_plus1*(1800*p4-2700*p2+225))/((32*p6+96*p4+96*p2+32)),
			(75600*p4-75600*p2+4725)/((128*p6+384*p4+384*p2+128)),
			(sqrt_p2_plus1*(1587600*p4-1190700*p2+59535))/((256*p8+1024*p6+1536*p4+1024*p2+256)),
			(8731800*p4-5239080*p2+218295)/((128*p8+512*p6+768*p4+512*p2+128)),
			(204324120*p4-102162060*p2+3648645)/(sqrt_p2_plus1*(256*p8+1024*p6+1536*p4+1024*p2+256)),
			(20432412000*p4-8756748000*p2+273648375)/((2048*p10+10240*p8+20480*p6+20480*p4+10240*p2+2048)),
			(545837292000*p4-204688984500*p2+5685805125)/(sqrt_p2_plus1*(4096*p10+20480*p8+40960*p6+40960*p4+20480*p2+4096))},
			
			{-(3600*p6-27000*p4+20250*p2-1125)/((32*p6+96*p4+96*p2+32)),
			-(sqrt_p2_plus1*(705600*p6-2646000*p4+1323000*p2-55125))/((256*p8+1024*p6+1536*p4+1024*p2+256)),
			-(3175200*p6-7938000*p4+2976750*p2-99225)/((64*p8+256*p6+384*p4+256*p2+64)),
			-(209563200*p6-392931000*p4+117879300*p2-3274425)/(sqrt_p2_plus1*(256*p8+1024*p6+1536*p4+1024*p2+256)),
			-(3405402000*p6-5108103000*p4+1277025750*p2-30405375)/((256*p10+1280*p8+2560*p6+2560*p4+1280*p2+256)),
			-(899026128000*p6-1123782660000*p4+240810570000*p2-5016886875)/(sqrt_p2_plus1*(4096*p10+20480*p8+40960*p6+40960*p4+20480*p2+4096)),
			-(7641722088000*p6-8187559380000*p4+1535167383750*p2-28429025625)/((2048*p12+12288*p10+30720*p8+40960*p6+30720*p4+12288*p2+2048)),
			-(sqrt_p2_plus1*(539287244496000*p6-505581791715000*p4+84263631952500*p2-1404393865875))/(8192*p14+57344*p12+172032*p10+286720*p8+286720*p6+172032*p4+57344*p2+8192)} };
		
		UNROLL
		for (int nu=0; nu < NU_MAX; nu++) {

			double r_dependence = pow(r, 2*nu + 1);
			
			UNROLL
			for (int m=0; m < M_MAX; m++) {
				double base = output_base[nu][m];
					
				double jac = jacobian_buffer[h][k];
				double C = trig_cos_buffer[h][k][m], S = trig_sin_buffer[h][k][m];
				
				output_coeffs[i][0][nu][m] += charges[h]*jac*base*C*r_dependence * factorial[nu][m];
				output_coeffs[i][1][nu][m] += charges[h]*jac*base*S*r_dependence * factorial[nu][m];
				
				r_dependence *= r;
			}
		}
	}
}

EXPORT double
potential_3d_derivs(double point[3], double *zs, double *coeffs_p, size_t N_z) {

	double (*coeffs)[2][NU_MAX][M_MAX][4] = (double (*)[2][NU_MAX][M_MAX][4]) coeffs_p;
	
	double xp = point[0], yp = point[1], zp = point[2];

	if (!(zs[0] < zp && zp < zs[N_z-1])) return 0.0;

	double dz = zs[1] - zs[0];
	int index = (int) ((zp-zs[0])/dz);
	
	double z_ = zp - zs[index];

	double A[NU_MAX][M_MAX], B[NU_MAX][M_MAX];
	double (*C)[NU_MAX][M_MAX][4] = &coeffs[index][0];
		
	for (int nu=0; nu < NU_MAX; nu++)
	for (int m=0; m < M_MAX; m++) {
		A[nu][m] = pow(z_, 3)*C[0][nu][m][0] + pow(z_, 2)*C[0][nu][m][1] + z_*C[0][nu][m][2] + C[0][nu][m][3];
		B[nu][m] = pow(z_, 3)*C[1][nu][m][0] + pow(z_, 2)*C[1][nu][m][1] + z_*C[1][nu][m][2] + C[1][nu][m][3];
	}

	double r = norm_2d(xp, yp);
	double phi = atan2(yp, xp);
	
	double sum_ = 0.0;
	
	for (int nu=0; nu < NU_MAX; nu++)
	for (int m=0; m < M_MAX; m++)
		sum_ += (A[nu][m]*cos(m*phi) + B[nu][m]*sin(m*phi))*pow(r, (m+2*nu));
	
	return sum_;
}

EXPORT void
field_3d_derivs(double point[3], double field[3], double *restrict zs, double *restrict coeffs_p, size_t N_z) {
	
	double (*coeffs)[2][NU_MAX][M_MAX][4] = (double (*)[2][NU_MAX][M_MAX][4]) coeffs_p;

	double xp = point[0], yp = point[1], zp = point[2];

	field[0] = 0.0, field[1] = 0.0, field[2] = 0.0;
	
	if (!(zs[0] < zp && zp < zs[N_z-1])) return;
		
	double dz = zs[1] - zs[0];
	int index = (int) ((zp-zs[0])/dz);
	
	double z_ = zp - zs[index];

	double A[NU_MAX][M_MAX], B[NU_MAX][M_MAX];
	double Adiff[NU_MAX][M_MAX], Bdiff[NU_MAX][M_MAX];
	
	double (*C)[NU_MAX][M_MAX][4] = &coeffs[index][0];
		
	UNROLL
	for (int nu=0; nu < NU_MAX; nu++)
	UNROLL
	for (int m=0; m < M_MAX; m++) {
		A[nu][m] = pow(z_, 3)*C[0][nu][m][0] + pow(z_, 2)*C[0][nu][m][1] + z_*C[0][nu][m][2] + C[0][nu][m][3];
		B[nu][m] = pow(z_, 3)*C[1][nu][m][0] + pow(z_, 2)*C[1][nu][m][1] + z_*C[1][nu][m][2] + C[1][nu][m][3];
		
		Adiff[nu][m] = 3*pow(z_, 2)*C[0][nu][m][0] + 2*z_*C[0][nu][m][1]+ C[0][nu][m][2];
		Bdiff[nu][m] = 3*pow(z_, 2)*C[1][nu][m][0] + 2*z_*C[1][nu][m][1]+ C[1][nu][m][2];
	}
		
	double r = norm_2d(xp, yp);
	double phi = atan2(yp, xp);
	
	if(r < MIN_DISTANCE_AXIS) {
		field[0] = -A[0][1];
		field[1] = -B[0][1];
		field[2] = -Adiff[0][0];
		return;
	}
	
	
	UNROLL
	for (int nu=0; nu < NU_MAX; nu++)
	UNROLL
	for (int m=0; m < M_MAX; m++) {
		int exp = 2*nu + m;

		double diff_r = (A[nu][m]*cos(m*phi) + B[nu][m]*sin(m*phi)) * exp*pow(r, exp-1);
		double diff_theta = m*(-A[nu][m]*sin(m*phi) + B[nu][m]*cos(m*phi)) * pow(r, exp);
		
		field[0] -= diff_r * xp/r + diff_theta * -yp/pow(r,2);
		field[1] -= diff_r * yp/r + diff_theta * xp/pow(r,2);
		field[2] -= (Adiff[nu][m]*cos(m*phi) + Bdiff[nu][m]*sin(m*phi)) * pow(r, exp);
	}
}

EXPORT void triangle_areas(vertices_3d triangles, double *out, size_t N) {
	
	for(int i = 0; i < N; i++) {
		double v1[3] = {
			triangles[i][1][0] - triangles[i][0][0],
			triangles[i][1][1] - triangles[i][0][1],
			triangles[i][1][2] - triangles[i][0][2]
		};
		
		double v2[3] = {
			triangles[i][2][0] - triangles[i][0][0],
			triangles[i][2][1] - triangles[i][0][1],
			triangles[i][2][2] - triangles[i][0][2]
		};
		
		double cross[3];
		cross_product_3d(v1, v2, cross);
		out[i] = 0.5*norm_3d(cross[0], cross[1], cross[2]);
	}
}



