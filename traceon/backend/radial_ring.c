
#define DERIV_2D_MAX 9

INLINE double flux_density_to_charge_factor(double K) {
// There is quite some derivation to this factor.
// In terms of the displacement field D, we have:
// D2 - D1 = s (where s is the charge density, sigma)
// This law follows directly from Gauss' law.
// The electric field normal accross the inteface is continuous
// E1 = E2 or D1/e1 = D2/e2 which gives e2/e1 D1 = D2.
// If we let e1 be the permittivity of vacuum we have K D1 = D2 where K is 
// the relative permittivity. Now our program can only compute the average
// of the field in material 1 and the field in material 2, by the way the 
// boundary charge integration works. So we compute 
// D = (D1 + D2)/2 = (K+1)/2 D1.
// We then have s = D2 - D1 = (K - 1) D1 = 2*(K-1)/(K+1) D.
	return 2.0*(K - 1)/(1 + K);
}


EXPORT double dr1_potential_radial_ring(double r0, double z0, double r, double z, void *_) {
	
	if(r0 < MIN_DISTANCE_AXIS) {
		return 0.0;
	}
	
	double delta_r = r - r0;
	double delta_z = z - z0;
    double common_arg = (delta_z * delta_z + delta_r * delta_r) / (4 * r * r - 4 * delta_r * r + delta_z * delta_z + delta_r * delta_r);
    double denominator = ((-2 * delta_r * delta_r * r) + delta_z * delta_z * (2 * delta_r - 2 * r) + 2 * delta_r * delta_r * delta_r) * sqrt(4 * r * r - 4 * delta_r * r + delta_z * delta_z + delta_r * delta_r);
    double ellipkm1_term = (delta_z * delta_z * r + delta_r * delta_r * r) * ellipkm1(common_arg);
    double ellipem1_term = ((-2 * delta_r * r * r) - delta_z * delta_z * r + delta_r * delta_r * r) * ellipem1(common_arg);
    return 1./M_PI * (ellipkm1_term + ellipem1_term) / denominator;
}

EXPORT double potential_radial_ring(double r0, double z0, double r, double z, void *_) {
    double delta_z = z - z0;
    double delta_r = r - r0;
    double t = (pow(delta_z, 2) + pow(delta_r, 2)) / (pow(delta_z, 2) + pow(delta_r, 2) + 4 * r0 * delta_r + 4 * pow(r0, 2));
    return 1./M_PI * ellipkm1(t) * (delta_r + r0) / sqrt(pow(delta_z, 2) + pow((delta_r + 2 * r0), 2));
}

EXPORT double dz1_potential_radial_ring(double r0, double z0, double r, double z, void *_) {
	double delta_z = z - z0;
    double delta_r = r - r0;
    double common_arg = (delta_z * delta_z + delta_r * delta_r) / (4 * r0 * r0 + 4 * delta_r * r0 + delta_z * delta_z + delta_r * delta_r);
    double denominator = (delta_z * delta_z + delta_r * delta_r) * sqrt(4 * r0 * r0 + 4 * delta_r * r0 + delta_z * delta_z + delta_r * delta_r);
    double ellipem1_term = -delta_z * (r0 + delta_r) * ellipem1(common_arg);
    return 1./M_PI * -ellipem1_term / denominator;
}

double
field_dot_normal_radial(double r0, double z0, double r, double z, void* args_p) {

	struct {double *normal; double K;} *args = args_p;
	
	// This factor is hard to derive. It takes into account that the field
	// calculated at the edge of the dielectric is basically the average of the
	// field at either side of the surface of the dielecric (the field makes a jump).
	double K = args->K;
	double factor = flux_density_to_charge_factor(K);
	
	double Er = -dr1_potential_radial_ring(r0, z0, r, z, NULL);
	double Ez = -dz1_potential_radial_ring(r0, z0, r, z, NULL);
	
	return factor*(args->normal[0]*Er + args->normal[1]*Ez);

}

EXPORT double current_potential_axial_radial_ring(double z0, double r, double z) {
	double dz = z0 - z;
	return -dz / (2*sqrt(dz*dz + r*r));
}

EXPORT void
current_field_radial_ring(double x0, double y0, double x, double y, double result[2]) {
	// https://drive.google.com/file/d/0B5Hb04O3hlQSa0dGdlUtRnQ5OXM/view?resourcekey=0-VFNsHQLd7H9uMSRALuVwvw
	// https://www.grant-trebbin.com/2012/04/off-axis-magnetic-field-of-circular.html
	// https://patentimages.storage.googleapis.com/46/cb/4d/ed27deb544ce3a/EP0310212A2.pdf
	double a = x;
	double r = x0;
	double z = y0 - y;

	double A = pow(z, 2) + pow(a+r, 2);
	double B = pow(z, 2) + pow(r-a, 2);
	
	double k = 4*r*a/A;
	
	if(x < MIN_DISTANCE_AXIS) {
		// Unphysical situation, infinitely small ring
		result[0] = 0.;
		result[1] = 0.;
		return;
	}
			
	// TODO: figure out how to prevent singularity
	if(x0 < MIN_DISTANCE_AXIS) {
		result[0] = 0.;
	}
	else {
		result[0] = 1./M_PI * z/(2*r*sqrt(A)) * ( (pow(z,2) + pow(r,2) + pow(a,2))/B * ellipe(k) - ellipk(k) );
	}
	result[1] = 1./M_PI * 1/(2*sqrt(A)) * ( (pow(a,2) - pow(z,2) - pow(r,2))/B * ellipe(k) + ellipk(k) );
}

EXPORT void
axial_derivatives_radial_ring(double z0, double r, double z, double derivs[DERIV_2D_MAX]) {
	
	double R = norm_2d(z0-z, r);
	
	derivs[0] = 1/R;
	derivs[1] = -(z0-z)/pow(R, 3);
		
	for(int n = 1; n+1 < DERIV_2D_MAX; n++)
		derivs[n+1] = -1./pow(R,2) *( (2*n + 1)*(z0-z)*derivs[n] + pow(n,2)*derivs[n-1]);
	
	for(int n = 0; n < DERIV_2D_MAX; n++)
		derivs[n] *= r/2;
}

EXPORT void
current_axial_derivatives_radial_ring(double z0, double r, double z, double derivs[DERIV_2D_MAX]) {

	double dz = z0-z;	
	double R = norm_2d(dz, r);
	double mu = dz/R;
	
	derivs[0] = -dz/(2*sqrt(dz*dz + r*r));
	derivs[1] = -r*r/(2*pow(dz*dz + r*r, 1.5));
		
	for(int n = 2; n < DERIV_2D_MAX; n++)
		derivs[n] = -(2*n-1)*mu/R*derivs[n-1] - (n*n - 2*n)/(R*R)*derivs[n-2];
}

