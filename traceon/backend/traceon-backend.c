#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

#include "defs.c"
#include "utilities_3d.c"
#include "triangle_contribution.c"


#define DERIV_2D_MAX 9
#define NU_MAX 4
#define M_MAX 8

// DERIV_2D_MAX, NU_MAX_SYM and M_MAX_SYM need to be present in the .so file to
// be able to read them. We cannot call them NU_MAX and M_MAX as
// the preprocessor will substitute their names. We can also not 
// simply only use these symbols instead of the preprocessor variables
// as the length of arrays need to be a compile time constant in C...
EXPORT const int DERIV_2D_MAX_SYM = 9;
EXPORT const int NU_MAX_SYM = NU_MAX;
EXPORT const int M_MAX_SYM = M_MAX;


#define TRACING_STEP_MAX 0.01
#define MIN_DISTANCE_AXIS 1e-10

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif


EXPORT const size_t TRACING_BLOCK_SIZE = (size_t) 1e5;

#define N_QUAD_2D 16
EXPORT const int N_QUAD_2D_SYM = N_QUAD_2D;
const double GAUSS_QUAD_WEIGHTS[N_QUAD_2D] = {0.1894506104550685, 0.1894506104550685, 0.1826034150449236, 0.1826034150449236, 0.1691565193950025, 0.1691565193950025, 0.1495959888165767, 0.1495959888165767, 0.1246289712555339, 0.1246289712555339, 0.0951585116824928, 0.0951585116824928, 0.0622535239386479, 0.0622535239386479, 0.0271524594117541, 0.0271524594117541};
const double GAUSS_QUAD_POINTS[N_QUAD_2D] = {-0.0950125098376374, 0.0950125098376374, -0.2816035507792589, 0.2816035507792589, -0.4580167776572274, 0.4580167776572274, -0.6178762444026438, 0.6178762444026438, -0.7554044083550030, 0.7554044083550030, -0.8656312023878318, 0.8656312023878318, -0.9445750230732326, 0.9445750230732326, -0.9894009349916499, 0.9894009349916499};


// Triangle quadrature constants
//#define N_TRIANGLE_QUAD 33
//EXPORT const int N_TRIANGLE_QUAD_SYM = N_TRIANGLE_QUAD;
//const double QUAD_WEIGHTS[N_TRIANGLE_QUAD] = {0.03127061, 0.01424303, 0.02495917, 0.01213342, 0.00396582, 0.03127061, 0.01424303, 0.02495917, 0.01213342, 0.00396582, 0.03127061, 0.01424303, 0.02495917, 0.01213342, 0.00396582, 0.02161368, 0.00754184, 0.01089179, 0.02161368, 0.00754184, 0.01089179, 0.02161368, 0.00754184, 0.01089179, 0.02161368, 0.00754184, 0.01089179, 0.02161368, 0.00754184, 0.01089179, 0.02161368, 0.00754184, 0.01089179};
//const double QUAD_B1[N_TRIANGLE_QUAD] = {0.27146251, 0.10925783, 0.44011165, 0.48820375, 0.02464636, 0.27146251, 0.10925783, 0.44011165, 0.48820375, 0.02464636, 0.45707499, 0.78148434, 0.1197767 , 0.0235925 , 0.95070727, 0.11629602, 0.02138249, 0.02303416, 0.62824975, 0.85133779, 0.68531016, 0.25545423, 0.12727972, 0.29165568, 0.25545423, 0.12727972, 0.29165568, 0.62824975, 0.85133779, 0.68531016, 0.11629602, 0.02138249, 0.02303416};
//const double QUAD_B2[N_TRIANGLE_QUAD] = {0.27146251, 0.10925783, 0.44011165, 0.48820375, 0.02464636, 0.45707499, 0.78148434, 0.1197767 , 0.0235925 , 0.95070727, 0.27146251, 0.10925783, 0.44011165, 0.48820375, 0.02464636, 0.25545423, 0.12727972, 0.29165568, 0.11629602, 0.02138249, 0.02303416, 0.62824975, 0.85133779, 0.68531016, 0.11629602, 0.02138249, 0.02303416, 0.25545423, 0.12727972, 0.29165568, 0.62824975, 0.85133779, 0.68531016};

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



//////////////////////////////// TYPEDEFS


typedef double (*integration_cb_2d)(double, double, double, double, void*);
typedef double (*vertices_2d)[4][3];
typedef double (*charges_2d)[N_QUAD_2D];

typedef double (*integration_cb_3d)(double, double, double, double, double, double, void*);
typedef double (triangle)[3][3];
typedef double (*vertices_3d)[3][3];

typedef double (*positions_3d)[6];

typedef double (*jacobian_buffer_3d)[N_TRIANGLE_QUAD];
typedef double (*position_buffer_3d)[N_TRIANGLE_QUAD][3];

typedef double (*positions_2d)[4];

typedef double (*jacobian_buffer_2d)[N_QUAD_2D];
typedef double (*position_buffer_2d)[N_QUAD_2D][2];

//////////////////////////////// ELLIPTIC FUNCTIONS

// Chebyshev Approximations for the Complete Elliptic Integrals K and E.
// W. J. Cody. 1965.
//
// Augmented with the tricks shown on the Scipy documentation for ellipe and ellipk.
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipkm1.html#scipy.special.ellipkm1

EXPORT double ellipkm1(double p) {
	double A[] = {log(4.0),
			9.65736020516771e-2,
			3.08909633861795e-2,
			1.52618320622534e-2,
			1.25565693543211e-2,
			1.68695685967517e-2,
			1.09423810688623e-2,
			1.40704915496101e-3};
	
	double B[] = {1.0/2.0,
			1.24999998585309e-1,
			7.03114105853296e-2,
			4.87379510945218e-2,
			3.57218443007327e-2,
			2.09857677336790e-2,
			5.81807961871996e-3,
			3.42805719229748e-4};
	
	double L = log(1./p);
	double sum_ = 0.0;

	for(int i = 0; i < 8; i++)
		sum_ += (A[i] + L*B[i])*pow(p, i);
	
	return sum_;
}

EXPORT double ellipk(double k) {
	if(k > -1) return ellipkm1(1-k);
	
	return ellipkm1(1./(1-k))/sqrt(k);
}

EXPORT double ellipem1(double p) {
	double A[] = {1,
        4.43147193467733e-1,
        5.68115681053803e-2,
        2.21862206993846e-2,
        1.56847700239786e-2,
        1.92284389022977e-2,
        1.21819481486695e-2,
        1.55618744745296e-3};

    double B[] = {0,
        2.49999998448655e-1,
        9.37488062098189e-2,
        5.84950297066166e-2,
        4.09074821593164e-2,
        2.35091602564984e-2,
        6.45682247315060e-3,
        3.78886487349367e-4};
	
	double L = log(1./p);
	double sum_ = 0.0;

	for(int i = 0; i < 8; i++)
		sum_ += (A[i] + L*B[i])*pow(p, i);
		
	return sum_;
}

EXPORT double ellipe(double k) {
	if (0 <= k && k <= 1) return ellipem1(1-k);

	return ellipem1(-1/(k-1.))*sqrt(1-k);
}



typedef double (*ts_integrand)(double, void*);


EXPORT double tanh_sinh_integration(ts_integrand f, double a, double b, double epsrel, double epsabs, void *args) {
    double h = 2.7;
    double S = 0.5 * M_PI * sinh(h);
    double jacobian = (b-a)/2.;

    double sum = jacobian * M_PI/2. * f(a + (b - a)/2., args);
    double to_add = jacobian * M_PI/2. * cosh(h) / pow(cosh(S), 2) * (f(a + (b - a) * (-tanh(S) + 1) / 2., args) + f(a + (b - a) * (tanh(S) + 1) / 2., args));
		
	sum += isnormal(to_add) ? to_add : 0.0;
	
    double prev = sum;
    int level = 1;

    while (1) {
        h /= 2;


        for (int n = 0; n < level; n++) {
            S = 0.5 * M_PI * sinh((2 * n + 1) * h);
            double x = tanh(S);
            double w = jacobian * 0.5 * M_PI * cosh((2 * n + 1) * h) / pow(cosh(S), 2);
			
			double x_left = a + (b - a) * (-x + 1) / 2.;
			double x_right = a + (b - a) * (x + 1) / 2.;
            to_add = w * (f(x_left, args) + f(x_right, args));
			assert( (a <= x_left) && (x_left <= b) );
			assert( (a <= x_right) && (x_right <= b) );
			sum += isnormal(to_add) ? to_add : 0.0;
        }

        if (fabs(h * sum - 2 * h * prev) < epsabs + epsrel * fabs(h * sum)) {
			return h * sum;
		}

        
        prev = sum;
        level *= 2;
    }
}


//////////////////////////////// UTILITIES 2D


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

//////////////////////////////// UTILITIES 3D


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

INLINE double potential_3d_point(double x0, double y0, double z0, double x, double y, double z, void *_) {
	double r = norm_3d(x-x0, y-y0, z-z0);
    return 1/(4*M_PI*r);
}

//////////////////////////////// PARTICLE TRACING

// python -c "from scipy.constants import m_e, e; print(-e/m_e);"
const double EM = -175882001077.2163; // Electron charge over electron mass
// python -c "from scipy.constants import mu_0; print(mu_0);"
const double MU_0 = 1.25663706212e-06;

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



//////////////////////////////// RADIAL RING POTENTIAL (DERIVATIVES)


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


// TODO: fix name, radial_ring is not consistent with other naming
EXPORT void
axial_derivatives_radial_ring(double *derivs_p, double *charges, jacobian_buffer_2d jac_buffer, position_buffer_2d pos_buffer, size_t N_lines, double *z, size_t N_z) {

	double (*derivs)[DERIV_2D_MAX] = (double (*)[DERIV_2D_MAX]) derivs_p;	
		
	for(int i = 0; i < N_z; i++) 
	for(int j = 0; j < N_lines; j++)
	for(int k = 0; k < N_QUAD_2D; k++) {
		double z0 = z[i];
		double r = pos_buffer[j][k][0], z = pos_buffer[j][k][1];
		
		double R = norm_2d(z0-z, r);
		
		double D[DERIV_2D_MAX] = {0.}; // Derivatives of the currently considered line element.
		D[0] = 1/R;
		D[1] = -(z0-z)/pow(R, 3);
			
		for(int n = 1; n+1 < DERIV_2D_MAX; n++)
			D[n+1] = -1./pow(R,2) *( (2*n + 1)*(z0-z)*D[n] + pow(n,2)*D[n-1]);
			
		for(int l = 0; l < DERIV_2D_MAX; l++) derivs[i][l] += jac_buffer[j][k] * charges[j] * r/2 * D[l];
	}
}

//////////////////////////////// CURRENT AND MAGNETOSTATICS

EXPORT double current_potential_axial_radial_ring(double z0, double r, double z) {
	double dz = z0 - z;
	return -dz / (2*sqrt(dz*dz + r*r));
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
current_field(double point[3], double result[3], double *currents,
	jacobian_buffer_3d jacobian_buffer, position_buffer_3d position_buffer, size_t N_vertices) {

	double Br = 0., Bz = 0.;
	double r = norm_2d(point[0], point[1]);
	
	for(int i = 0; i < N_vertices; i++) {
		for(int k = 0; k < N_TRIANGLE_QUAD; k++) {
			double *pos = &position_buffer[i][k][0];
			double field[2];
			current_field_radial_ring(r, point[2], pos[0], pos[1], field);
			assert(pos[2] == 0.);
				
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

// TODO: fix name, radial_ring is not consistent with other naming
EXPORT void
current_axial_derivatives_radial_ring(double *derivs_p,
		double *currents, jacobian_buffer_3d jac_buffer, position_buffer_3d pos_buffer, size_t N_vertices, double *z, size_t N_z) {

	double (*derivs)[DERIV_2D_MAX] = (double (*)[DERIV_2D_MAX]) derivs_p;	
		
	for(int i = 0; i < N_z; i++) 
	for(int j = 0; j < N_vertices; j++)
	for(int k = 0; k < N_TRIANGLE_QUAD; k++) {
		double z0 = z[i];
		double r = pos_buffer[j][k][0], z = pos_buffer[j][k][1];

		double dz = z0-z;	
		double R = norm_2d(dz, r);
		double mu = dz/R;
		
		double D[DERIV_2D_MAX] = {0.}; // Derivatives of the currently considered line element.
		D[0] = -dz/(2*sqrt(dz*dz + r*r));
		D[1] = -r*r/(2*pow(dz*dz + r*r, 1.5));
			
		for(int n = 2; n < DERIV_2D_MAX; n++)
			D[n] = -(2*n-1)*mu/R*D[n-1] - (n*n - 2*n)/(R*R)*D[n-2];
			
		for(int l = 0; l < DERIV_2D_MAX; l++) derivs[i][l] += jac_buffer[j][k] * currents[j] * D[l];
	}
}



//////////////////////////////// RADIAL SYMMETRY POTENTIAL EVALUATION

EXPORT double
potential_radial(double point[2], double* charges, jacobian_buffer_2d jacobian_buffer, position_buffer_2d position_buffer, size_t N_vertices) {

	double sum_ = 0.0;  
	
	for(int i = 0; i < N_vertices; i++) {  
		for(int k = 0; k < N_QUAD_2D; k++) {
			double *pos = &position_buffer[i][k][0];
			double potential = potential_radial_ring(point[0], point[1], pos[0], pos[1], NULL);
			sum_ += charges[i] * jacobian_buffer[i][k] * potential;
		}
	}  
	
	return sum_;
}

EXPORT double
potential_radial_derivs(double point[2], double *z_inter, double *coeff_p, size_t N_z) {
	
	double (*coeff)[DERIV_2D_MAX][6] = (double (*)[DERIV_2D_MAX][6]) coeff_p;
	
	double r = point[0], z = point[1];
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


//////////////////////////////// RADIAL SYMMETRY FIELD EVALUATION

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

struct effective_point_charges_2d {
	double *charges;
	jacobian_buffer_2d jacobians;
	position_buffer_2d positions;
	size_t N;
};

struct effective_point_charges_3d {
	double *charges;
	jacobian_buffer_3d jacobians;
	position_buffer_3d positions;
	size_t N;
};

struct field_evaluation_args {
	void *elec_charges;
	void *mag_charges;
	void *current_charges;
	double *bounds;
};

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

struct field_derivs_args {
	double *z_interpolation;
	double *electrostatic_axial_coeffs;
	double *magnetostatic_axial_coeffs;
	size_t N_z;
};

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


//////////////////////////////// 3D POINT POTENTIAL (DERIVATIVES)

EXPORT double dx1_potential_3d_point(double x0, double y0, double z0, double x, double y, double z, void *_) {
	double r = norm_3d(x-x0, y-y0, z-z0);
    return (x-x0)/(4*M_PI*pow(r, 3));
}

EXPORT double dy1_potential_3d_point(double x0, double y0, double z0, double x, double y, double z, void *_) {
	double r = norm_3d(x-x0, y-y0, z-z0);
    return (y-y0)/(4*M_PI*pow(r, 3));
}

EXPORT double dz1_potential_3d_point(double x0, double y0, double z0, double x, double y, double z, void *_) {
	double r = norm_3d(x-x0, y-y0, z-z0);
    return (z-z0)/(4*M_PI*pow(r, 3));
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


//////////////////////////////// 3D POINT POTENTIAL EVALUATION

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

//////////////////////////////// 3D POINT FIELD EVALUATION

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


//////////////////////////////// SOLVER

enum ExcitationType{
    VOLTAGE_FIXED = 1,
    VOLTAGE_FUN = 2,
    DIELECTRIC = 3,
	CURRENT=4,
	MAGNETOSTATIC_POT=5,
	MAGNETIZABLE=6};


struct self_voltage_radial_args {
	double (*line_points)[3];
	double *target;
	integration_cb_2d cb_fun;
	double *normal;
	double K;
};

double self_voltage_radial(double alpha, void *args_p) {
	
	struct self_voltage_radial_args* args = (struct self_voltage_radial_args*) args_p;
	
	double *v1 = args->line_points[0];
	double *v2 = args->line_points[2];
	double *v3 = args->line_points[3];
	double *v4 = args->line_points[1];
	
	double pos[2], jac;
	position_and_jacobian_radial(alpha, v1, v2, v3, v4, pos, &jac);
	
	struct {double *normal; double K;} cb_args = {args->normal, args->K};
	
	return jac * args->cb_fun(args->target[0], args->target[1], pos[0], pos[1], &cb_args);
}

void fill_self_voltages_radial(double *matrix, 
                        vertices_2d line_points,
						uint8_t *excitation_types,
						double *excitation_values,
						size_t N_lines,
						size_t N_matrix,
                        int lines_range_start, 
                        int lines_range_end) {
	 
	gsl_integration_workspace * w = gsl_integration_workspace_alloc(ADAPTIVE_MAX_ITERATION);
	
	for(int i = lines_range_start; i <= lines_range_end; i++) {
		double *v1 = &line_points[i][0][0];
		double *v2 = &line_points[i][2][0];
		double *v3 = &line_points[i][3][0];
		double *v4 = &line_points[i][1][0];
		
		double target[2], jac;
		position_and_jacobian_radial(0.0, v1, v2, v3, v4, target, &jac);
		
		double normal[2];
		higher_order_normal_radial(0.0, v1, v2, v3, v4, normal);
		//normal_2d(v1, v2, normal);
			
		enum ExcitationType type_ = excitation_types[i];
			
		struct self_voltage_radial_args integration_args = {
			.target = target,
			.line_points = &line_points[i][0],
			.normal = normal,
			.K = excitation_values[i],
			.cb_fun = (type_ != DIELECTRIC && type_ != MAGNETIZABLE) ? potential_radial_ring : field_dot_normal_radial
		};
			
		gsl_function F;
		F.function = &self_voltage_radial;
		F.params = &integration_args;
			
		double result, error;
		double singular_points[3] = {-1, 0, 1};
		gsl_integration_qagp(&F, singular_points, 3, 1e-9, 1e-9, ADAPTIVE_MAX_ITERATION, w, &result, &error);

		if(type_ == DIELECTRIC || type_ == MAGNETIZABLE) {
			matrix[N_matrix*i + i] = result - 1;
		}
		else {
			matrix[N_matrix*i + i] = result;
		}
	}
	
	gsl_integration_workspace_free(w);
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
    
	gsl_set_error_handler_off();
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
	
	fill_self_voltages_radial(matrix, line_points, excitation_types, excitation_values, N_lines, N_matrix, lines_range_start, lines_range_end);
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
	
	gsl_set_error_handler_off();
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





