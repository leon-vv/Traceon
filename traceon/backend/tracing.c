
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

typedef void (*field_fun)(double pos[6], double field[3], void* args);

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
produce_new_k(double ys[6][6], double ks[6][6], size_t index, double h, field_fun ff, void *args) {
	
	double field[3] = { 0. };
	ff(ys[index], field, args);
	
	ks[index][0] = h*ys[index][3];
	ks[index][1] = h*ys[index][4];
	ks[index][2] = h*ys[index][5];
	ks[index][3] = h*EM*field[0];
	ks[index][4] = h*EM*field[1];
	ks[index][5] = h*EM*field[2];
}


EXPORT size_t
trace_particle(double *times_array, double *pos_array, field_fun field, double bounds[3][2], double atol, void *args) {
	
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
			produce_new_k(ys, k, index, h, field, args);
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

void
field_radial_traceable(double point[6], double result[3], void *args_p) {
	
	struct field_evaluation_args *args = (struct field_evaluation_args*) args_p;

	struct effective_point_charges_2d *elec_charges = (struct effective_point_charges_2d*) args->elec_charges;
	struct effective_point_charges_2d *mag_charges = (struct effective_point_charges_2d*) args->mag_charges;
	struct effective_point_charges_3d *current_charges = (struct effective_point_charges_3d*) args->current_charges;
	
	double (*bounds)[2] = (double (*)[2]) args->bounds;
	
	if(args->bounds == NULL || ((bounds[0][0] < point[0]) && (point[0] < bounds[0][1])
						 && (bounds[1][0] < point[1]) && (point[1] < bounds[1][1]))) {
		
		double elec_field[3] = {0.};
		double mag_field[3] = {0.};
		double curr_field[3] = {0.};
		
		field_radial(point, elec_field,
			elec_charges->charges, elec_charges->jacobians, elec_charges->positions, elec_charges->N);
		
		field_radial(point, mag_field,
			mag_charges->charges, mag_charges->jacobians, mag_charges->positions, mag_charges->N);
			
		current_field(point, curr_field,
			current_charges->charges, current_charges->jacobians, current_charges->positions, current_charges->N);
			
		combine_elec_magnetic_field(point + 3, elec_field, mag_field, curr_field, result);
	}
	else {
		result[0] = 0.;
		result[1] = 0.;
		result[2] = 0.;
	}
}



EXPORT size_t
trace_particle_radial(double *times_array, double *pos_array, double tracer_bounds[3][2], double atol, double *field_bounds,
		struct effective_point_charges_2d eff_elec,
		struct effective_point_charges_2d eff_mag,
		struct effective_point_charges_3d eff_current) {
	
	struct field_evaluation_args args = {
		.elec_charges = (void*) &eff_elec,
		.mag_charges = (void*) &eff_mag,
		.current_charges = (void*) &eff_current,
		.bounds = field_bounds
	};
		
	return trace_particle(times_array, pos_array, field_radial_traceable, tracer_bounds, atol, (void*) &args);
}

void
field_radial_derivs_traceable(double point[6], double field[3], void *args_p) {
	struct field_derivs_args *args = (struct field_derivs_args*) args_p;

	double elec_field[3];
	field_radial_derivs(point, elec_field, args->z_interpolation, args->electrostatic_axial_coeffs, args->N_z);
	
	double mag_field[3];
	field_radial_derivs(point, mag_field, args->z_interpolation, args->magnetostatic_axial_coeffs, args->N_z);

	double curr_field[3] = {0., 0., 0.};
	combine_elec_magnetic_field(point + 3, elec_field, mag_field, curr_field, field);
}

EXPORT size_t
trace_particle_radial_derivs(double *times_array, double *pos_array, double bounds[3][2], double atol,
	double *z_interpolation, double *electrostatic_coeffs, double *magnetostatic_coeffs, size_t N_z) {

	struct field_derivs_args args = { z_interpolation, electrostatic_coeffs, magnetostatic_coeffs, N_z };
		
	return trace_particle(times_array, pos_array, field_radial_derivs_traceable, bounds, atol, (void*) &args);
}

void
field_3d_traceable(double point[6], double result[3], void *args_p) {
	struct field_evaluation_args *args = (struct field_evaluation_args*)args_p;
	struct effective_point_charges_3d *elec_charges = (struct effective_point_charges_3d*) args->elec_charges;
	struct effective_point_charges_3d *mag_charges = (struct effective_point_charges_3d*) args->mag_charges;
	
	double (*bounds)[2] = (double (*)[2]) args->bounds;
	
	if(	bounds == NULL || ((bounds[0][0] < point[0]) && (point[0] < bounds[0][1])
		&& (bounds[1][0] < point[1]) && (point[1] < bounds[1][1])
		&& (bounds[2][0] < point[2]) && (point[2] < bounds[2][1])) ) {

		double elec_field[3] = {0.};
		double mag_field[3] = {0.};
		double curr_field[3] = {0.};
			
		field_3d(point, elec_field, elec_charges->charges, elec_charges->jacobians, elec_charges->positions, elec_charges->N);
		field_3d(point, mag_field, mag_charges->charges, mag_charges->jacobians, mag_charges->positions, mag_charges->N);
		combine_elec_magnetic_field(point + 3, elec_field, mag_field, curr_field, result);
	}
	else {
		result[0] = 0.0;
		result[1] = 0.0;
		result[2] = 0.0;
	}
}

EXPORT size_t
trace_particle_3d(double *times_array, double *pos_array, double tracer_bounds[3][2], double atol,
		struct effective_point_charges_3d eff_elec, struct effective_point_charges_3d eff_mag, double *field_bounds) {
	
	struct field_evaluation_args args = {.elec_charges = (void*) &eff_elec, .mag_charges = (void*) &eff_mag, .bounds = field_bounds};
	
	return trace_particle(times_array, pos_array, field_3d_traceable, tracer_bounds, atol, (void*) &args);
}



void
field_3d_derivs_traceable(double point[6], double field[3], void *args_p) {
	struct field_derivs_args *args = (struct field_derivs_args*) args_p;
	
	double elec_field[3];
	field_3d_derivs(point, elec_field, args->z_interpolation, args->electrostatic_axial_coeffs, args->N_z);
	
	double mag_field[3];
	field_3d_derivs(point, mag_field, args->z_interpolation, args->magnetostatic_axial_coeffs, args->N_z);
	
	double curr_field[3] = {0., 0., 0.};
	combine_elec_magnetic_field(point + 3, elec_field, mag_field, curr_field, field);
}

EXPORT size_t
trace_particle_3d_derivs(double *times_array, double *pos_array, double bounds[3][2], double atol,
	double *z_interpolation, double *electrostatic_coeffs, double *magnetostatic_coeffs, size_t N_z) {

	struct field_derivs_args args = { z_interpolation, electrostatic_coeffs, magnetostatic_coeffs, N_z };
	
	return trace_particle(times_array, pos_array, field_3d_derivs_traceable, bounds, atol, (void*) &args);
}


EXPORT void fill_jacobian_buffer_3d(
	jacobian_buffer_3d jacobian_buffer,
	position_buffer_3d pos_buffer,
    vertices_3d t,
    size_t N_triangles) {
		
    for(int i = 0; i < N_triangles; i++) {  
		
		double x1 = t[i][0][0], y1 = t[i][0][1], z1 = t[i][0][2];
		double x2 = t[i][1][0], y2 = t[i][1][1], z2 = t[i][1][2];
		double x3 = t[i][2][0], y3 = t[i][2][1], z3 = t[i][2][2];
				
		double area = 0.5*sqrt(
			pow((y2-y1)*(z3-z1)-(y3-y1)*(z2-z1), 2) +
			pow((x3-x1)*(z2-z1)-(x2-x1)*(z3-z1), 2) +
			pow((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1), 2));
		
        for (int k=0; k < N_TRIANGLE_QUAD; k++) {  
            double b1_ = QUAD_B1[k];  
            double b2_ = QUAD_B2[k];  
            double w = QUAD_WEIGHTS[k];  
			
            jacobian_buffer[i][k] = 2 * w * area;
            pos_buffer[i][k][0] = x1 + b1_*(x2 - x1) + b2_*(x3 - x1);
            pos_buffer[i][k][1] = y1 + b1_*(y2 - y1) + b2_*(y3 - y1);
            pos_buffer[i][k][2] = z1 + b1_*(z2 - z1) + b2_*(z3 - z1);
        }
    }
}

EXPORT void fill_matrix_3d(double *restrict matrix, 
                    vertices_3d triangle_points, 
                    uint8_t *excitation_types, 
                    double *excitation_values, 
					jacobian_buffer_3d jacobian_buffer,
					position_buffer_3d pos_buffer,
					size_t N_lines,
					size_t N_matrix,
                    int lines_range_start, 
                    int lines_range_end) {
	
	assert(lines_range_start < N_lines && lines_range_end < N_lines);
		
    for (int i = lines_range_start; i <= lines_range_end; i++) {
		double target[3], jac;
		position_and_jacobian_3d(1/3., 1/3., &triangle_points[i][0], target, &jac);
			
        enum ExcitationType type_ = excitation_types[i];
		 
        if (type_ == VOLTAGE_FIXED || type_ == VOLTAGE_FUN || type_ == MAGNETOSTATIC_POT) {
            for (int j = 0; j < N_lines; j++) {

				// Position of first integration point. Check if 
				// close to the target triangle.
				double distance = distance_3d(triangle_points[j][0], target);
				double characteristic_length = distance_3d(triangle_points[j][0], triangle_points[j][1]);
				
				if(i == j) {
					matrix[i*N_matrix + j] = self_potential_triangle(&triangle_points[j][0][0], &triangle_points[j][1][0], &triangle_points[j][2][0], target);
				}
				if(i != j && distance > 5*characteristic_length) {
					for(int k = 0; k < N_TRIANGLE_QUAD; k++) {
							
						double *pos = pos_buffer[j][k];
						double jac = jacobian_buffer[j][k];
						matrix[i*N_matrix + j] += jac * potential_3d_point(target[0], target[1], target[2], pos[0], pos[1], pos[2], NULL);
					}
					
				}
				else {
					matrix[i*N_matrix + j] = potential_triangle(triangle_points[j][0], triangle_points[j][1], triangle_points[j][2], target) / (4*M_PI);
				}
				
            }
        } 
		else if (type_ == DIELECTRIC || type_ == MAGNETIZABLE) {  
			
			double normal[3];  
			normal_3d(1/3., 1/3., &triangle_points[i][0], normal);
			double K = excitation_values[i];  
			
			// This factor is hard to derive. It takes into account that the field
			// calculated at the edge of the dielectric is basically the average of the
			// field at either side of the surface of the dielecric (the field makes a jump).
			double factor = flux_density_to_charge_factor(K);
				
			for (int j = 0; j < N_lines; j++) {  
					
				double distance = distance_3d(triangle_points[j][0], target);
				double characteristic_length = distance_3d(triangle_points[j][0], triangle_points[j][1]);

				if(i == j) {
					matrix[i*N_matrix + j] = -1.0;
				}
				else if(distance > 5*characteristic_length) {
					for(int k = 0; k < N_TRIANGLE_QUAD; k++) {
						double *pos = pos_buffer[j][k];  
						double jac = jacobian_buffer[j][k];  
						
						matrix[i*N_matrix + j] += factor * jac * field_dot_normal_3d(target[0], target[1], target[2], pos[0], pos[1], pos[2], normal);  
					}
				}
				else {
					double a = triangle_area(triangle_points[j][0], triangle_points[j][1], triangle_points[j][2]);
					matrix[i*N_matrix + j] = factor * flux_triangle(triangle_points[j][0], triangle_points[j][1], triangle_points[j][2], target, normal) / (4*M_PI);
				}
			}  
		}  
        else {
            printf("ExcitationType unknown\n");
            exit(1);
        }
    }
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

EXPORT bool
line_intersection(double p0[2], double tangent[2], positions_2d positions, size_t N_p, double result[4]) {
	
	assert(N_p > 1);
		
	double xp = p0[0], yp = p0[1];
	// Normal components, perpendicular to tangent
	double xn = tangent[1], yn = -tangent[0];

	// Initial sign
	int i = N_p-1;
	
	double x = positions[i][0], y = positions[i][1];
	double prev_kappa = (yn*yp-y*yn+xn*xp-x*xn)/norm_2d(xn, yn);
		
	i -= 1;
			
	for(; i >= 0; i--) {
		double x = positions[i][0], y = positions[i][1];
		double kappa = (yn*yp-y*yn+xn*xp-x*xn)/norm_2d(xn, yn);
			
		int sign_kappa = kappa > 0 ? 1 : -1;
		int sign_prev = prev_kappa > 0 ? 1 : -1;
		
		if(sign_kappa != sign_prev) {
			double diff = kappa - prev_kappa;
			
			double factor = -prev_kappa / diff;
			double prev_factor = kappa / diff;
			
			for(int k = 0; k < 4; k++)
				result[k] = prev_factor*positions[i+1][k] + factor*positions[i][k];

			return true;
		}
		
		prev_kappa = kappa;
	}
	
	return false;
}






