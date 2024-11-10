

EXPORT const int DERIV_2D_MAX_SYM = DERIV_2D_MAX;

typedef double (*jacobian_buffer_2d)[N_QUAD_2D];
typedef double (*position_buffer_2d)[N_QUAD_2D][2];

typedef double (*vertices_2d)[4][3];

struct effective_point_charges_2d {
	double *charges;
	jacobian_buffer_2d jacobians;
	position_buffer_2d positions;
	size_t N;
};


EXPORT void
axial_derivatives_radial(double *derivs_p, double *charges, jacobian_buffer_2d jac_buffer, position_buffer_2d pos_buffer, size_t N_lines, double *z, size_t N_z) {

	double (*derivs)[DERIV_2D_MAX] = (double (*)[DERIV_2D_MAX]) derivs_p;	
		
	for(int i = 0; i < N_z; i++) 
	for(int j = 0; j < N_lines; j++)
	for(int k = 0; k < N_QUAD_2D; k++) {
		double z0 = z[i];
		double r = pos_buffer[j][k][0];
		double z = pos_buffer[j][k][1];

		double D[DERIV_2D_MAX];

		axial_derivatives_radial_ring(z0, r, z, D);
		
		for(int l = 0; l < DERIV_2D_MAX; l++) derivs[i][l] += jac_buffer[j][k] * charges[j] * D[l];
	}
}

EXPORT double
current_potential_axial(double z0, double *currents,
	jacobian_buffer_3d jacobian_buffer, position_buffer_3d position_buffer, size_t N_vertices) {

	double result = 0.;

	for(int i = 0; i < N_vertices; i++) 
	for(int k = 0; k < N_TRIANGLE_QUAD; k++) {
		double *pos = &position_buffer[i][k][0];
		assert(pos[2] == 0.);
			
		result += currents[i] * jacobian_buffer[i][k] * current_potential_axial_radial_ring(z0, pos[0], pos[1]);
	}

	return result;
}

EXPORT void
current_field(double point[3], double result[3], double *currents,
	jacobian_buffer_3d jacobian_buffer, position_buffer_3d position_buffer, size_t N_vertices) {

	double Br = 0., Bz = 0.;
	double r = norm_2d(point[0], point[1]);
	
	for(int i = 0; i < N_vertices; i++) {
		for(int k = 0; k < N_TRIANGLE_QUAD; k++) {
			double *pos = &position_buffer[i][k][0];
			double field[2];
			current_field_radial_ring(r, point[2], pos[0], pos[2], field);
			assert(pos[1] == 0.);
				
			Br += currents[i] * jacobian_buffer[i][k] * field[0];
			Bz += currents[i] * jacobian_buffer[i][k] * field[1];
		}
	}
		
	if(r >= MIN_DISTANCE_AXIS) {
		result[0] = point[0]/r * Br;
		result[1] = point[1]/r * Br;
	}
	else {
		result[0] = 0.;
		result[1] = 0.;
	}
	result[2] = Bz;
}

EXPORT void
current_axial_derivatives_radial(double *derivs_p,
		double *currents, jacobian_buffer_3d jac_buffer, position_buffer_3d pos_buffer, size_t N_vertices, double *z, size_t N_z) {

	double (*derivs)[DERIV_2D_MAX] = (double (*)[DERIV_2D_MAX]) derivs_p;	
		
	for(int i = 0; i < N_z; i++) 
	for(int j = 0; j < N_vertices; j++)
	for(int k = 0; k < N_TRIANGLE_QUAD; k++) {
		double z0 = z[i];
		double r = pos_buffer[j][k][0], z = pos_buffer[j][k][1];

		double D[DERIV_2D_MAX];
		
		current_axial_derivatives_radial_ring(z0, r, z, D);
			
		for(int l = 0; l < DERIV_2D_MAX; l++) derivs[i][l] += jac_buffer[j][k] * currents[j] * D[l];
	}
}

EXPORT double
potential_radial(double point[3], double* charges, jacobian_buffer_2d jacobian_buffer, position_buffer_2d position_buffer, size_t N_vertices) {

	double sum_ = 0.0;  
	double r0 = norm_2d(point[0], point[1]);
	double z0 = point[2];
	
	for(int i = 0; i < N_vertices; i++) {  
		for(int k = 0; k < N_QUAD_2D; k++) {
			double *pos = &position_buffer[i][k][0];
			double potential = potential_radial_ring(r0, z0, pos[0], pos[1], NULL);
			sum_ += charges[i] * jacobian_buffer[i][k] * potential;
		}
	}  
	
	return sum_;
}

EXPORT double
charge_radial(double *vertices_p, double charge) {

	double (*vertices)[3] = (double (*)[3]) vertices_p;
		
	double *v1 = &vertices[0][0];
	double *v2 = &vertices[2][0]; // Strange ordering following from GMSH line4 element
	double *v3 = &vertices[3][0];
	double *v4 = &vertices[1][0];
		
	double sum_ = 0.0;
	
	for(int k = 0; k < N_QUAD_2D; k++) {
				
		double pos[2], jac;
		position_and_jacobian_radial(GAUSS_QUAD_POINTS[k], v1, v2, v3, v4, pos, &jac);
		
		// Surface area is 2pi*r * charge_integral
		// charge_integral is charge integrated over line element
		// charge_integral is weight*dl*charge
		// where dl is the jacobian
		sum_ += 2*M_PI*pos[0]*GAUSS_QUAD_WEIGHTS[k]*jac*charge;
	}

	return sum_;
}

EXPORT void
field_radial(double point[3], double result[3], double* charges, jacobian_buffer_2d jacobian_buffer, position_buffer_2d position_buffer, size_t N_vertices) {
	
	double Er = 0.0, Ez = 0.0;
	double r = norm_2d(point[0], point[1]);
	
	for(int i = 0; i < N_vertices; i++) {  
		for(int k = 0; k < N_QUAD_2D; k++) {
			double *pos = &position_buffer[i][k][0];
			Er -= charges[i] * jacobian_buffer[i][k] * dr1_potential_radial_ring(r, point[2], pos[0], pos[1], NULL);
			Ez -= charges[i] * jacobian_buffer[i][k] * dz1_potential_radial_ring(r, point[2], pos[0], pos[1], NULL);
		}
	}
				
	if(r >= MIN_DISTANCE_AXIS) {
		result[0] = point[0]/r * Er;
		result[1] = point[1]/r * Er;
	}
	else {
		result[0] = 0.;
		result[1] = 0.;
	}
	result[2] = Ez;
}

struct field_evaluation_args {
	void *elec_charges;
	void *mag_charges;
	void *current_charges;
	double *bounds;
};

EXPORT double self_potential_radial(double alpha, double line_points[4][3]) {

	double *v1 = line_points[0];
	double *v2 = line_points[2];
	double *v3 = line_points[3];
	double *v4 = line_points[1];
	
	double pos[2], jac;
	position_and_jacobian_radial(alpha, v1, v2, v3, v4, pos, &jac);
	
	double target[2], jac2;
	position_and_jacobian_radial(0, v1, v2, v3, v4, target, &jac2);

	return jac*potential_radial_ring(target[0], target[1], pos[0], pos[1], NULL);
}

struct self_field_dot_normal_radial_args {
	double (*line_points)[3];
	double K;
};

EXPORT double self_field_dot_normal_radial(double alpha, struct self_field_dot_normal_radial_args* args) {
	
	double *v1 = args->line_points[0];
	double *v2 = args->line_points[2];
	double *v3 = args->line_points[3];
	double *v4 = args->line_points[1];
	
	double pos[2], jac;
	position_and_jacobian_radial(alpha, v1, v2, v3, v4, pos, &jac);
	
	double target[2], jac2;
	position_and_jacobian_radial(0, v1, v2, v3, v4, target, &jac2);
	
	double normal[2];
	higher_order_normal_radial(0.0, v1, v2, v3, v4, normal);
	
	struct {double *normal; double K;} cb_args = {normal, args->K};

	return jac*field_dot_normal_radial(target[0], target[1], pos[0], pos[1], (void*) &cb_args);
}

EXPORT void fill_jacobian_buffer_radial(
	jacobian_buffer_2d jacobian_buffer,
	position_buffer_2d pos_buffer,
    vertices_2d line_points,
    size_t N_lines) {
	
    for(int i = 0; i < N_lines; i++) {  
        for (int k=0; k < N_QUAD_2D; k++) {  
			double *v1 = &line_points[i][0][0];
			double *v2 = &line_points[i][2][0];
			double *v3 = &line_points[i][3][0];
			double *v4 = &line_points[i][1][0];
				
            double pos[2], jac;  
			
            position_and_jacobian_radial(GAUSS_QUAD_POINTS[k], v1, v2, v3, v4, pos, &jac);  
			
            jacobian_buffer[i][k] = GAUSS_QUAD_WEIGHTS[k]*jac;  
            pos_buffer[i][k][0] = pos[0];  
            pos_buffer[i][k][1] = pos[1];  
        }  
    }  
}


EXPORT void fill_matrix_radial(double *matrix, 
						vertices_2d line_points,
                        uint8_t *excitation_types, 
                        double *excitation_values, 
						jacobian_buffer_2d jacobian_buffer,
						position_buffer_2d pos_buffer,
						size_t N_lines,
						size_t N_matrix,
                        int lines_range_start, 
                        int lines_range_end) {
    
	assert(lines_range_start < N_lines && lines_range_end < N_lines);
	assert(N_matrix >= N_lines);
		
    for (int i = lines_range_start; i <= lines_range_end; i++) {
		
		double *target_v1 = &line_points[i][0][0];
		double *target_v2 = &line_points[i][2][0];
		double *target_v3 = &line_points[i][3][0];
		double *target_v4 = &line_points[i][1][0];
		
		double target[2], jac;
		position_and_jacobian_radial(0.0, target_v1, target_v2, target_v3, target_v4, target, &jac);
		
		enum ExcitationType type_ = excitation_types[i];
			
		if (type_ == VOLTAGE_FIXED || type_ == VOLTAGE_FUN || type_ == MAGNETOSTATIC_POT) {
			for (int j = 0; j < N_lines; j++) {
				
				UNROLL
				for(int k = 0; k < N_QUAD_2D; k++) {
						
					double *pos = pos_buffer[j][k];
					double jac = jacobian_buffer[j][k];
					matrix[i*N_matrix + j] += jac * potential_radial_ring(target[0], target[1], pos[0], pos[1], NULL);
				}
            }
		}
		else if(type_ == DIELECTRIC || type_ == MAGNETIZABLE) {
			for (int j = 0; j < N_lines; j++) {

				double normal[2];
				//normal_2d(target_v1, target_v2, normal);
				higher_order_normal_radial(0.0, target_v1, target_v2, target_v3, target_v4, normal);
					
				struct {double *normal; double K;} args = {normal, excitation_values[i]};
					
				UNROLL
				for(int k = 0; k < N_QUAD_2D; k++) {
						
					double *pos = pos_buffer[j][k];
					double jac = jacobian_buffer[j][k];
					matrix[i*N_matrix + j] += jac * field_dot_normal_radial(target[0], target[1], pos[0], pos[1], &args);
				}
			}
		}
		else {
		    printf("ExcitationType unknown\n");
            exit(1);
		}
	}
}

EXPORT double
potential_radial_derivs(double point[3], double *z_inter, double *coeff_p, size_t N_z) {
	
	double (*coeff)[DERIV_2D_MAX][6] = (double (*)[DERIV_2D_MAX][6]) coeff_p;
		
	double r = norm_2d(point[0], point[1]), z = point[2];
	double z0 = z_inter[0], zlast = z_inter[N_z-1];
	
	if(!(z0 < z && z < zlast)) {
		return 0.0;
	}
	
	double dz = z_inter[1] - z_inter[0];
	int index = (int) ( (z-z0)/dz );
	double diffz = z - z_inter[index];
		
	double (*C)[6] = &coeff[index][0];
		
	double derivs[DERIV_2D_MAX];

	for(int i = 0; i < DERIV_2D_MAX; i++)
		derivs[i] = C[i][0]*pow(diffz, 5) + C[i][1]*pow(diffz, 4) + C[i][2]*pow(diffz, 3)
			      +	C[i][3]*pow(diffz, 2) + C[i][4]*diffz		  + C[i][5];
		
	return derivs[0] - pow(r,2)/4*derivs[2] + pow(r,4)/64.*derivs[4] - pow(r,6)/2304.*derivs[6] + pow(r,8)/147456.*derivs[8];
}


EXPORT void
field_radial_derivs(double point[3], double field[3], double *z_inter, double *coeff_p, size_t N_z) {
	double (*coeff)[DERIV_2D_MAX][6] = (double (*)[DERIV_2D_MAX][6]) coeff_p;
	
	double r = norm_2d(point[0], point[1]), z = point[2];
	double z0 = z_inter[0], zlast = z_inter[N_z-1];
	
	if(!(z0 < z && z < zlast)) {
		field[0] = 0.0; field[1] = 0.0; field[2] = 0.0;
		return;
	}
	
	double dz = z_inter[1] - z_inter[0];
	int index = (int) ( (z-z0)/dz );
	double diffz = z - z_inter[index];
		
	double (*C)[6] = &coeff[index][0];
		
	double derivs[DERIV_2D_MAX];

	for(int i = 0; i < DERIV_2D_MAX; i++)
		derivs[i] = C[i][0]*pow(diffz, 5) + C[i][1]*pow(diffz, 4) + C[i][2]*pow(diffz, 3)
			      +	C[i][3]*pow(diffz, 2) + C[i][4]*diffz		  + C[i][5];
		
	// Field radial is already divided by r, such that x/r*field and y/r*field below do not cause divide by zero errors
	double field_radial = 0.5*(derivs[2] - pow(r,2)/8*derivs[4] + pow(r,4)/192*derivs[6] - pow(r,6)/9216*derivs[8]);
	double field_z = -derivs[1] + pow(r,2)/4*derivs[3] - pow(r,4)/64*derivs[5] + pow(r,6)/2304*derivs[7];

	field[0] = point[0]*field_radial;
	field[1] = point[1]*field_radial;
	field[2] = field_z;
}





