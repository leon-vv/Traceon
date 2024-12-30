
typedef double (*positions_2d)[4];
typedef double (*positions_3d)[6];

#define TRACING_STEP_MAX 0.01

EXPORT const size_t TRACING_BLOCK_SIZE = (size_t) 1e5;

const double A[]  = {0.0, 2./9., 1./3., 3./4., 1., 5./6.};	// https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
const double B6[] = {65./432., -5./16., 13./16., 4./27., 5./144.};
const double B5[] = {-17./12., 27./4., -27./5., 16./15.};
const double B4[] = {69./128., -243./128., 135./64.};
const double B3[] = {1./12., 1./4.};
const double B2[] = {2./9.};
const double CH[] = {47./450., 0., 12./25., 32./225., 1./30., 6./25.};
const double CT[] = {-1./150., 0., 3./100., -16./75., -1./20., 6./25.};

typedef void (*field_fun)(double position[3], double velocity[3], void* args, double elec_out[3], double mag_out[3]);

void
produce_new_y(double y[6], double ys[6][6], double ks[6][6], size_t index) {
	
	const double* coefficients[] = {NULL, B2, B3, B4, B5, B6};
	
	for(int i = 0; i < 6; i++) {
		
		ys[index][i] = y[i];
		
		for(int j = 0; j < index; j++) 
			ys[index][i] += coefficients[index][j]*ks[j][i];
	}
}

void
produce_new_k(double ys[6][6], double ks[6][6], size_t index, double h, double charge_over_mass, field_fun ff, void *args) {
	double elec[3] = { 0. };
	double mag[3] = { 0. };
	
	ff(&ys[index][0], &ys[index][3], args, elec, mag);
	
	// Convert to acceleration using Lorentz force law
	double cross[3] = { 0. }; 
	cross_product_3d(&ys[index][3], mag, cross); // Compute v x H

	double field[3] = { // Compute E + v x B
		elec[0] + MU_0*cross[0],
		elec[1] + MU_0*cross[1],
		elec[2] + MU_0*cross[2]
	};

	ks[index][0] = h*ys[index][3];
	ks[index][1] = h*ys[index][4];
	ks[index][2] = h*ys[index][5];
	ks[index][3] = h*charge_over_mass*field[0];
	ks[index][4] = h*charge_over_mass*field[1];
	ks[index][5] = h*charge_over_mass*field[2];
}


EXPORT size_t
trace_particle(double *times_array, double *pos_array, double charge_over_mass, field_fun field, double bounds[3][2], double atol, void *args) {
	
	double (*positions)[6] = (double (*)[6]) pos_array;
	
	double y[6];
	for(int i = 0; i < 6; i++) y[i] = positions[0][i];
	
    double V = norm_3d(y[3], y[4], y[5]);
    double hmax = TRACING_STEP_MAX/V;
    double h = hmax;
	
    int N = 1;
		
    double xmin = bounds[0][0], xmax = bounds[0][1];
	double ymin = bounds[1][0], ymax = bounds[1][1];
	double zmin = bounds[2][0], zmax = bounds[2][1];

	 
    while( (xmin <= y[0]) && (y[0] <= xmax) &&
		   (ymin <= y[1]) && (y[1] <= ymax) &&
		   (zmin <= y[2]) && (y[2] <= zmax) ) {
		
		double k[6][6] = { {0.} };
		double ys[6][6] = { {0.} };
		
		for(int index = 0; index < 6; index++) {
			produce_new_y(y, ys, k, index);
			produce_new_k(ys, k, index, h, charge_over_mass, field, args);
		}
		
		double max_position_error = 0.0;
		double max_velocity_error = 0.0;

		for(int i = 0; i < 3; i++) {
			double err = 0.0;
			for(int j = 0; j < 6; j++) err += CT[j]*k[j][i];
			if(fabs(err) > max_position_error) max_position_error = fabs(err);
		}
		
		for(int i = 3; i < 6; i++) {
			double err = 0.0;
			for(int j = 0; j < 6; j++) err += CT[j]*k[j][i];
			if(fabs(err) > max_velocity_error) max_velocity_error = fabs(err);
		}
		
		double error = max_position_error + h*max_velocity_error;
			
		if(error <= atol) {
			for(int i = 0; i < 6; i++) {
				y[i] += CH[0]*k[0][i] + CH[1]*k[1][i] + CH[2]*k[2][i] + CH[3]*k[3][i] + CH[4]*k[4][i] + CH[5]*k[5][i];
				positions[N][i] = y[i];
				times_array[N] = times_array[N-1] + h;
			}
				
			N += 1;
			if(N==TRACING_BLOCK_SIZE) return N;
		}
		
		h = fmin(0.9 * h * pow(atol / error, 0.2), hmax);
	}
		
	return N;
}

EXPORT void
field_radial_traceable(double position[3], double velocity[3], void *args_p, double elec_out[3], double mag_out[3]) {
	
	struct field_evaluation_args *args = (struct field_evaluation_args*) args_p;

	struct effective_point_charges_2d *elec_charges = (struct effective_point_charges_2d*) args->elec_charges;
	struct effective_point_charges_2d *mag_charges = (struct effective_point_charges_2d*) args->mag_charges;
	struct effective_point_charges_3d *current_charges = (struct effective_point_charges_3d*) args->currents;
	
	double (*bounds)[2] = (double (*)[2]) args->bounds;
	
	if(args->bounds == NULL || ((bounds[0][0] < position[0]) && (position[0] < bounds[0][1])
						        && (bounds[1][0] < position[1]) && (position[1] < bounds[1][1])
						        && (bounds[2][0] < position[2]) && (position[2] < bounds[2][1]))) {
		
		
		field_radial(position, elec_out,
			elec_charges->charges, elec_charges->jacobians, elec_charges->positions, elec_charges->N);
		
		field_radial(position, mag_out,
			mag_charges->charges, mag_charges->jacobians, mag_charges->positions, mag_charges->N);
			
		double curr_field[3] = {0.};
		
		current_field_radial(position, curr_field,
			current_charges->charges, current_charges->jacobians, current_charges->positions, current_charges->N);
		
		mag_out[0] += curr_field[0];
		mag_out[1] += curr_field[1];
		mag_out[2] += curr_field[2];
	}
	else {
		elec_out[0] = 0.;
		elec_out[1] = 0.;
		elec_out[2] = 0.;
		
		mag_out[0] = 0.;
		mag_out[1] = 0.;
		mag_out[2] = 0.;
	}
}


EXPORT void
field_radial_derivs_traceable(double position[3], double velocity[3], void *args_p, double elec_out[3], double mag_out[3]) {
	struct field_derivs_args *args = (struct field_derivs_args*) args_p;
	
	field_radial_derivs(position, elec_out, args->z_interpolation, args->electrostatic_axial_coeffs, args->N_z);
	field_radial_derivs(position, mag_out, args->z_interpolation, args->magnetostatic_axial_coeffs, args->N_z);
}

void
field_3d_derivs_traceable(double position[3], double velocity[3], void *args_p, double elec_out[3], double mag_out[3]) {
	struct field_derivs_args *args = (struct field_derivs_args*) args_p;
	
	field_3d_derivs(position, elec_out, args->z_interpolation, args->electrostatic_axial_coeffs, args->N_z);
	field_3d_derivs(position, mag_out, args->z_interpolation, args->magnetostatic_axial_coeffs, args->N_z);
}

EXPORT bool
plane_intersection(double p0[3], double normal[3], positions_3d positions, size_t N_p, double result[6]) {
	
	assert(N_p > 1);
		
	double xp = p0[0], yp = p0[1], zp = p0[2];
	double xn = normal[0], yn = normal[1], zn = normal[2];

	// Initial sign
	int i = N_p-1;
	
	double x = positions[i][0], y = positions[i][1], z = positions[i][2];	
	double prev_kappa = (zn*zp-z*zn+yn*yp-y*yn+xn*xp-x*xn)/norm_3d(xn, yn, zn);
		
	i -= 1;
			
	for(; i >= 0; i--) {
		double x = positions[i][0], y = positions[i][1], z = positions[i][2];	
		double kappa = (zn*zp-z*zn+yn*yp-y*yn+xn*xp-x*xn)/norm_3d(xn, yn, zn);
		
		int sign_kappa = kappa > 0 ? 1 : -1;
		int sign_prev = prev_kappa > 0 ? 1 : -1;
		
		if(sign_kappa != sign_prev) {
			double diff = kappa - prev_kappa;
			
			double factor = -prev_kappa / diff;
			double prev_factor = kappa / diff;
			
			for(int k = 0; k < 6; k++)
				result[k] = prev_factor*positions[i+1][k] + factor*positions[i][k];

			return true;
		}
		
		prev_kappa = kappa;
	}
	
	return false;
}
