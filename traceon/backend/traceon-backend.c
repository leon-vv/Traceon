#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)

#include <Python.h>
PyMODINIT_FUNC PyInit_traceon_backend(void) {
	return NULL;
}

#else
#define EXPORT extern
#endif

#if defined(__clang__)
	#define UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__) || defined(__GNUG__)
	#define UNROLL _Pragma("GCC unroll 100")
#else
	#define UNROLL
#endif


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

#define N_TRIANGLE_QUAD 9

#define TRACING_STEP_MAX 0.01
#define MIN_DISTANCE_AXIS 1e-10

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif


EXPORT const size_t TRACING_BLOCK_SIZE = (size_t) 1e5;

#define N_QUAD_2D 4
EXPORT const int N_QUAD_2D_SYM = N_QUAD_2D;
const double GAUSS_QUAD_POINTS[N_QUAD_2D] = {-0.3399810435848563, 0.3399810435848563, -0.8611363115940526, 0.8611363115940526};
const double GAUSS_QUAD_WEIGHTS[N_QUAD_2D] = {0.6521451548625461, 0.6521451548625461, 0.3478548451374538, 0.3478548451374538};


//////////////////////////////// TYPEDEFS

typedef double (*integration_cb_2d)(double, double, double, double, void*);
typedef double (*vertices_2d)[4][3];
typedef double (*charges_2d)[N_QUAD_2D];

//////////////////////////////// ELLIPTIC FUNCTIONS

// Chebyshev Approximations for the Complete Elliptic Integrals K and E.
// W. J. Cody. 1965.
//
// Augmented with the tricks shown on the Scipy documentation for ellipe and ellipk.


double ellipk_singularity(double k) {
	double eta = 1 - k;
	
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
	
	double L = log(1./eta);
	double sum_ = 0.0;

	for(int i = 0; i < 8; i++)
		sum_ += (A[i] + L*B[i])*pow(eta, i);
	
	return sum_;
}

EXPORT double ellipk(double k) {
	if(k > -1) return ellipk_singularity(k);
	
	return ellipk_singularity(1 - 1./(1-k))/sqrt(1-k);
}

double ellipe_01(double k) {
	double eta = 1 - k;
	
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
	
	double L = log(1./eta);
	double sum_ = 0.0;

	for(int i = 0; i < 8; i++)
		sum_ += (A[i] + L*B[i])*pow(eta, i);
		
	return sum_;
}

EXPORT double ellipe(double k) {
	if (0 <= k && k <= 1) return ellipe_01(k);

	return ellipe_01(k/(k-1.))*sqrt(1-k);
}


//////////////////////////////// UTILITIES 2D



EXPORT inline double
norm_2d(double x, double y) {
	return sqrt(x*x + y*y);
}

EXPORT inline double
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

void
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


EXPORT void inline position_and_jacobian_radial(double alpha, double *v1, double *v2, double *v3, double *v4, double *pos_out, double *jac) {

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


typedef double (*integration_cb_3d)(double, double, double, double, double, double, void*);

EXPORT inline double
norm_3d(double x, double y, double z) {
	return sqrt(x*x + y*y + z*z);
}

EXPORT void
normal_3d(double *p1, double *p2, double *p3, double *normal) {
	double x1 = p1[0], y1 = p1[1], z1 = p1[2];
	double x2 = p2[0], y2 = p2[1], z2 = p2[2];
	double x3 = p3[0], y3 = p3[1], z3 = p3[2];

	double normal_x = (y2-y1)*(z3-z1)-(y3-y1)*(z2-z1);
	double normal_y = (x3-x1)*(z2-z1)-(x2-x1)*(z3-z1);
	double normal_z = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1);
	double length = norm_3d(normal_x, normal_y, normal_z);
	
	normal[0] = normal_x/length;
	normal[1] = normal_y/length;
	normal[2] = normal_z/length;
}

// Triangle quadrature constants
const double QUAD_B1[N_TRIANGLE_QUAD] = {0.124949503233232, 0.437525248383384, 0.437525248383384, 0.797112651860071, 0.797112651860071, 0.165409927389841, 0.165409927389841, 0.037477420750088, 0.037477420750088};
const double QUAD_B2[N_TRIANGLE_QUAD] = {0.437525248383384, 0.124949503233232, 0.437525248383384, 0.165409927389841, 0.037477420750088, 0.797112651860071, 0.037477420750088, 0.797112651860071, 0.165409927389841};
const double QUAD_WEIGHTS[N_TRIANGLE_QUAD] = {0.205950504760887, 0.205950504760887, 0.205950504760887, 0.063691414286223, 0.063691414286223, 0.063691414286223, 0.063691414286223, 0.063691414286223, 0.063691414286223};

EXPORT double
triangle_integral(double target[3], double v1[3], double v2[3], double v3[3], integration_cb_3d function, void *args) {
	double v1x = v1[0], v1y = v1[1], v1z = v1[2];
	double v2x = v2[0], v2y = v2[1], v2z = v2[2];
	double v3x = v3[0], v3y = v3[1], v3z = v3[2];
		
	double area = 0.5*sqrt(pow((v2y-v1y)*(v3z-v1z)-(v2z-v1z)*(v3y-v1y), 2) + pow((v2z-v1z)*(v3x-v1x)-(v2x-v1x)*(v3z-v1z), 2) + pow((v2x-v1x)*(v3y-v1y)-(v2y-v1y)*(v3x-v1x), 2));
	
	double sum_ = 0.0;
	
	for (int k=0; k < N_TRIANGLE_QUAD; k++) {
		double b1_ = QUAD_B1[k];
		double b2_ = QUAD_B2[k];
		double w = QUAD_WEIGHTS[k];
			
        double x = v1x + b1_*(v2x-v1x) + b2_*(v3x-v1x);
        double y = v1y + b1_*(v2y-v1y) + b2_*(v3y-v1y);
        double z = v1z + b1_*(v2z-v1z) + b2_*(v3z-v1z);
			
        sum_ += w*function(target[0], target[1], target[2], x, y, z, args);
	}
	      
    return area*sum_;
}

// This is a bit of a hack.. we supply a triangle_integral function which is exactly
// the same as above, except we inline the 'potential_3d_point' function directly. Weirdly this
// seem to trigger some kind of optimization within GCC that makes building the matrix much faster. I was not
// able to reproduce this behaviour simply by extensive use of the 'inline' keyword.
EXPORT inline double potential_3d_point(double x0, double y0, double z0, double x, double y, double z, void *_) {
	double r = norm_3d(x-x0, y-y0, z-z0);
    return 1/(4*r);
}

double
triangle_integral_potential_3d_point(double target[3], double v1[3], double v2[3], double v3[3]) {
	double v1x = v1[0], v1y = v1[1], v1z = v1[2];
	double v2x = v2[0], v2y = v2[1], v2z = v2[2];
	double v3x = v3[0], v3y = v3[1], v3z = v3[2];
		
	double area = 0.5*sqrt(pow((v2y-v1y)*(v3z-v1z)-(v2z-v1z)*(v3y-v1y), 2) + pow((v2z-v1z)*(v3x-v1x)-(v2x-v1x)*(v3z-v1z), 2) + pow((v2x-v1x)*(v3y-v1y)-(v2y-v1y)*(v3x-v1x), 2));
	
	double sum_ = 0.0;
	
	for (int k=0; k < N_TRIANGLE_QUAD; k++) {
		double b1_ = QUAD_B1[k];
		double b2_ = QUAD_B2[k];
		double w = QUAD_WEIGHTS[k];
			
        double x = v1x + b1_*(v2x-v1x) + b2_*(v3x-v1x);
        double y = v1y + b1_*(v2y-v1y) + b2_*(v3y-v1y);
        double z = v1z + b1_*(v2z-v1z) + b2_*(v3z-v1z);
			
        sum_ += w*potential_3d_point(target[0], target[1], target[2], x, y, z, NULL);
	}
	      
    return area*sum_;
}





//////////////////////////////// PARTICLE TRACING


const double EM = -0.1758820022723908; // e/m units ns and mm

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
		
		double TE = 0.0; // Error 
		
		for(int i = 0; i < 6; i++) {
			double err = 0.0;
			for(int j = 0; j < 6; j++) err += CT[j]*k[j][i];
			if(fabs(err) > TE) TE = fabs(err);
		}
			
		if(TE <= atol) {
			for(int i = 0; i < 6; i++) {
				y[i] += CH[0]*k[0][i] + CH[1]*k[1][i] + CH[2]*k[2][i] + CH[3]*k[3][i] + CH[4]*k[4][i] + CH[5]*k[5][i];
				positions[N][i] = y[i];
				times_array[N] = times_array[N-1] + h;
			}
				
			N += 1;
			if(N==TRACING_BLOCK_SIZE) return N;
		}
		
		h = fmin(0.9 * h * pow(atol / TE, 0.2), hmax);
	}
		
	return N;
}



//////////////////////////////// RADIAL RING POTENTIAL (DERIVATIVES)


EXPORT double dr1_potential_radial_ring(double r_0, double z_0, double r, double z, void *_) {
	
	if (fabs(r_0) < MIN_DISTANCE_AXIS) return 0.0; // Prevent stepping into singularity
	
    double s = norm_2d(z-z_0, r+r_0);
    double s1 = (r_0 + r) / s;
    double t = 4.0 * r * r_0 / pow(s, 2);
    double A = ellipe(t);
    double B = ellipk(t);
    double ellipe_term = -(2.0 * r * r_0 * s1 - r * s) / (2.0 * r_0 * pow(s, 2) - 8.0 * pow(r_0, 2) * r);
    double ellipk_term = -r / (2.0 * r_0 * s);
    return A * ellipe_term + B * ellipk_term;
}


EXPORT double potential_radial_ring(double r_0, double z_0, double r, double z, void *_) {
    double rz2 = pow(r + r_0, 2) + pow(z - z_0, 2);
    double t = 4.0 * r * r_0 / rz2;
    return ellipk(t) * r / sqrt(rz2);
}

EXPORT double dz1_potential_radial_ring(double r_0, double z_0, double r, double z, void *_) {
    double rz2 = pow(r + r_0, 2) + pow(z - z_0, 2);
    double t = 4.0 * r * r_0 / rz2;
    double numerator = r * (z - z_0) * ellipe(t);
    double denominator = ((pow(z - z_0, 2) + pow(r - r_0, 2)) * sqrt(rz2));
    return numerator / denominator;
}


EXPORT void
axial_derivatives_radial_ring(double *derivs_p, vertices_2d lines, charges_2d charges, size_t N_lines, double *z, size_t N_z) {

	double (*derivs)[9] = (double (*)[9]) derivs_p;	
		
	for(int i = 0; i < N_z; i++) 
	for(int j = 0; j < N_lines; j++)
	for(int k = 0; k < N_QUAD_2D; k++) {
		double z0 = z[i];

		double *v1 = &lines[j][0][0];
		double *v2 = &lines[j][2][0];
		double *v3 = &lines[j][3][0];
		double *v4 = &lines[j][1][0];
			
		double pos[2], jac;
		position_and_jacobian_radial(GAUSS_QUAD_POINTS[k], v1, v2, v3, v4, pos, &jac);
		double r = pos[0], z = pos[1];
		
		double weight = GAUSS_QUAD_WEIGHTS[k] * jac;
			
		double R = norm_2d(z0-z, r);
		
		double D[9] = {0.}; // Derivatives of the currently considered line element.
		D[0] = 1/R;
		D[1] = -(z0-z)/pow(R, 3);
			
		for(int n = 1; n+1 < DERIV_2D_MAX; n++)
			D[n+1] = -1./pow(R,2) *( (2*n + 1)*(z0-z)*D[n] + pow(n,2)*D[n-1]);
		
		for(int l = 0; l < 9; l++) derivs[i][l] += weight * M_PI*r/2 * charges[j][k]*D[l];
	}
}

//////////////////////////////// RADIAL SYMMETRY POTENTIAL EVALUATION



// John A. Crow. Quadrature of Integrands with a Logarithmic Singularity. 1993.
// Computed higher order points and weights with own Python code..
#define N_LOG_QUAD_2D 12
const double GAUSS_LOG_QUAD_POINTS[N_LOG_QUAD_2D] =
					{0.000245284264977222214486999757349594712755863,
					0.003593698020213691584953180874184458866517133,
					0.01712292595159614370417434799786314278267083,
					0.05020561232318819384573670241984682206686949,
					0.1115235855573625644218104518656856812018872,
					0.2060030029051239758752495945505766309866304,
					0.3324626975136065400973675503604997045074356,
					0.4826067009659319425485030383099190543511725,
					0.6416794701239928589863342367708391692838668,
					0.7907090871833554068663998987642233419999952,
					0.9098838287655625921196627463680156613773641,
					0.9823479377157619510221418912858930542857947};

const double GAUSS_LOG_QUAD_WEIGHTS[N_LOG_QUAD_2D] =
					{0.0009331998830671428063097434941623766739840873,
					0.006977495915143715985530451759041482867494198,
					0.02169468902199853397715717421909584842021132,
					0.04597069439559112469266362311262695693833166,
					0.07752703869583178458842135310107317470803586,
					0.1112525181837396068263131383589812009329696,
					0.1402731060085833332292272698216837072064324,
					0.1575171300868296275541516485738687199310457,
					0.1574083422236489759715520802610080785790687,
					0.1372838327177858585860313716723887135089706,
					0.09819669547418950067876642091789781881144955,
					0.04496525739359079510387572470817192142199618};



EXPORT double
potential_radial(double point[3], vertices_2d vertices, charges_2d charges, size_t N_vertices) {

	double sum_ = 0.0;
	
	for(int i = 0; i < N_vertices; i++) {
		double *v1 = &vertices[i][0][0];
		double *v2 = &vertices[i][2][0]; // Strange ordering following from GMSH line4 element
		double *v3 = &vertices[i][3][0]; // Strange ordering following from GMSH line4 element
		double *v4 = &vertices[i][1][0];
		
		for(int j = 0; j < N_QUAD_2D; j++) {
			double pos[2], jac;
			position_and_jacobian_radial(GAUSS_QUAD_POINTS[j], v1, v2, v3, v4, pos, &jac);
			sum_ += jac*GAUSS_QUAD_WEIGHTS[j] * charges[i][j] * potential_radial_ring(point[0], point[1], pos[0], pos[1], NULL);
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
		
	return derivs[0] - pow(r,2)*derivs[2] + pow(r,4)/64.*derivs[4] - pow(r,6)/2304.*derivs[6] + pow(r,8)/147456.*derivs[8];
}


//////////////////////////////// RADIAL SYMMETRY FIELD EVALUATION


double
field_dot_normal_radial(double r0, double z0, double r, double z, void* args_p) {

	struct {double *normal; double K;} *args = args_p;
	
	// This factor is hard to derive. It takes into account that the field
	// calculated at the edge of the dielectric is basically the average of the
	// field at either side of the surface of the dielecric (the field makes a jump).
	double K = args->K;
	double factor = (2*K - 2) / (M_PI*(1 + K));
	
	double Er = -dr1_potential_radial_ring(r0, z0, r, z, NULL);
	double Ez = -dz1_potential_radial_ring(r0, z0, r, z, NULL);
	
	return factor*(args->normal[0]*Er + args->normal[1]*Ez);

}

EXPORT double
charge_radial(double *vertices_p, double *charges) {

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
		sum_ += 2*M_PI*pos[0]*GAUSS_QUAD_WEIGHTS[k]*jac*charges[k];
	}

	return sum_;
}

EXPORT void
field_radial(double point[3], double result[3], vertices_2d vertices, charges_2d charges, size_t N_vertices) {
		
	double Ex = 0.0, Ey = 0.0;
	
	for(int i = 0; i < N_vertices; i++) 
	for(int k = 0; k < N_QUAD_2D; k++) {
			
		double *v1 = &vertices[i][0][0];
		double *v2 = &vertices[i][2][0]; // Strange ordering following from GMSH line4 element
		double *v3 = &vertices[i][3][0];
		double *v4 = &vertices[i][1][0];
		
		double pos[2], jac;
		position_and_jacobian_radial(GAUSS_QUAD_POINTS[k], v1, v2, v3, v4, pos, &jac);
		
		Ex -= GAUSS_QUAD_WEIGHTS[k] * jac * charges[i][k] * dr1_potential_radial_ring(point[0], point[1], pos[0], pos[1], NULL);
		Ey -= GAUSS_QUAD_WEIGHTS[k] * jac * charges[i][k] * dz1_potential_radial_ring(point[0], point[1], pos[0], pos[1], NULL);
	}
		
	result[0] = Ex;
	result[1] = Ey;
	result[2] = 0.0;
}

struct field_evaluation_args {
	double *vertices;
	double *charges;
	size_t N_vertices;
};

void
field_radial_traceable(double point[3], double result[3], void *args_p) {

	struct field_evaluation_args *args = (struct field_evaluation_args*)args_p;
	field_radial(point, result, (vertices_2d) args->vertices, (charges_2d) args->charges, args->N_vertices);
}

EXPORT size_t
trace_particle_radial(double *times_array, double *pos_array, double bounds[3][2], double atol,
	double *vertices, double *charges, size_t N_vertices) {

	struct field_evaluation_args args = { vertices, charges, N_vertices };
				
	return trace_particle(times_array, pos_array, field_radial_traceable, bounds, atol, (void*) &args);
}

EXPORT void
field_radial_derivs(double point[3], double field[3], double *z_inter, double *coeff_p, size_t N_z) {
	
	double (*coeff)[DERIV_2D_MAX][6] = (double (*)[DERIV_2D_MAX][6]) coeff_p;
	
	double r = point[0], z = point[1];
	double z0 = z_inter[0], zlast = z_inter[N_z-1];
	
	if(!(z0 < z && z < zlast)) {
		field[0] = 0.0, field[1] = 0.0; field[2] = 0.0;
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
		
	field[0] = r/2*(derivs[2] - pow(r,2)/8*derivs[4] + pow(r,4)/192*derivs[6] - pow(r,6)/9216*derivs[8]);
	field[1] = -derivs[1] + pow(r,2)/4*derivs[3] - pow(r,4)/64*derivs[5] + pow(r,6)/2304*derivs[7];
	field[2] = 0.0;
}

struct field_derivs_args {
	double *z_interpolation;
	double *axial_coefficients;
	size_t N_z;
};

void
field_radial_derivs_traceable(double point[3], double field[3], void *args_p) {
	struct field_derivs_args *args = (struct field_derivs_args*) args_p;
	field_radial_derivs(point, field, args->z_interpolation, args->axial_coefficients, args->N_z);
}

EXPORT size_t
trace_particle_radial_derivs(double *times_array, double *pos_array, double bounds[3][2], double atol,
	double *z_interpolation, double *axial_coefficients, size_t N_z) {

	struct field_derivs_args args = { z_interpolation, axial_coefficients, N_z };
		
	return trace_particle(times_array, pos_array, field_radial_derivs_traceable, bounds, atol, (void*) &args);
}


//////////////////////////////// 3D POINT POTENTIAL (DERIVATIVES)

EXPORT double dx1_potential_3d_point(double x0, double y0, double z0, double x, double y, double z, void *_) {
	double r = norm_3d(x-x0, y-y0, z-z0);
    return (x-x0)/(4*pow(r, 3));
}

EXPORT double dy1_potential_3d_point(double x0, double y0, double z0, double x, double y, double z, void *_) {
	double r = norm_3d(x-x0, y-y0, z-z0);
    return (y-y0)/(4*pow(r, 3));
}

EXPORT double dz1_potential_3d_point(double x0, double y0, double z0, double x, double y, double z, void *_) {
	double r = norm_3d(x-x0, y-y0, z-z0);
    return (z-z0)/(4*pow(r, 3));
}

EXPORT void
axial_coefficients_3d(double *restrict vertices_p, double *restrict charges, size_t N_v,
	double *restrict zs, double *restrict output_coeffs_p, size_t N_z,
	double *restrict thetas, double *restrict theta_coeffs_p, size_t N_t) {
	
	double (*vertices)[3][3] = (double (*)[3][3]) vertices_p;
	double (*theta_coeffs)[NU_MAX][M_MAX][4] = (double (*)[NU_MAX][M_MAX][4]) theta_coeffs_p;
	double (*output_coeffs)[2][NU_MAX][M_MAX] = (double (*)[2][NU_MAX][M_MAX]) output_coeffs_p;

	double theta0 = thetas[0];
	double dtheta = thetas[1] - thetas[0];
	
	for(int h = 0; h < N_v; h++) {

		double v1x = vertices[h][0][0], v1y = vertices[h][0][1], v1z = vertices[h][0][2];
		double v2x = vertices[h][1][0], v2y = vertices[h][1][1], v2z = vertices[h][1][2];
		double v3x = vertices[h][2][0], v3y = vertices[h][2][1], v3z = vertices[h][2][2];
			
		double area = 0.5*sqrt(pow((v2y-v1y)*(v3z-v1z)-(v2z-v1z)*(v3y-v1y), 2) + pow((v2z-v1z)*(v3x-v1x)-(v2x-v1x)*(v3z-v1z), 2) + pow((v2x-v1x)*(v3y-v1y)-(v2y-v1y)*(v3x-v1x), 2));
		
        for (int i=0; i < N_z; i++) 
		UNROLL
		for (int k=0; k < N_TRIANGLE_QUAD; k++) {
			double b1_ = QUAD_B1[k];
			double b2_ = QUAD_B2[k];
			double w = QUAD_WEIGHTS[k];

			double x = v1x + b1_*(v2x-v1x) + b2_*(v3x-v1x);
			double y = v1y + b1_*(v2y-v1y) + b2_*(v3y-v1y);
			double z = v1z + b1_*(v2z-v1z) + b2_*(v3z-v1z);

			double r = norm_3d(x, y, z-zs[i]);
			double theta = atan2((z-zs[i]), norm_2d(x, y));
			double mu = atan2(y, x);

			int index = (int) ((theta-theta0)/dtheta);

			double t = theta-thetas[index];
			double (*C)[M_MAX][4] = &theta_coeffs[index][0];
				
			UNROLL
			for (int nu=0; nu < NU_MAX; nu++)
			UNROLL
			for (int m=0; m < M_MAX; m++) {
				double base = pow(t, 3)*C[nu][m][0] + pow(t, 2)*C[nu][m][1] + t*C[nu][m][2] + C[nu][m][3];
				double r_dependence = pow(r, -2*nu - m - 1);
					
				output_coeffs[i][0][nu][m] += charges[h]*area*w*base*cos(m*mu)*r_dependence;
				output_coeffs[i][1][nu][m] += charges[h]*area*w*base*sin(m*mu)*r_dependence;
			}
		}
	}
}


//////////////////////////////// 3D POINT POTENTIAL EVALUATION

EXPORT double
potential_3d(double point[3], double *vertices_p, double *charges, size_t N_vertices) {

	double (*vertices)[3][3] = (double (*)[3][3]) vertices_p;	

	double sum_ = 0.0;
	
	for(int i = 0; i < N_vertices; i++) {
		sum_ += charges[i] * triangle_integral(point, vertices[i][0], vertices[i][1], vertices[i][2], potential_3d_point, NULL);
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
field_3d(double point[3], double result[3], double *vertices_p, double *charges, size_t N_vertices) {
	
		double (*vertices)[3][3] = (double (*)[3][3]) vertices_p;
		
		double Ex = 0.0, Ey = 0.0, Ez = 0.0;
		
		for(int i = 0; i < N_vertices; i++) {
			
			double *v1, *v2, *v3;
			v1 = &vertices[i][0][0], v2 = &vertices[i][1][0], v3 = &vertices[i][2][0];
			
			Ex -= charges[i]*triangle_integral(point, v1, v2, v3, dx1_potential_3d_point, NULL);
			Ey -= charges[i]*triangle_integral(point, v1, v2, v3, dy1_potential_3d_point, NULL);
			Ez -= charges[i]*triangle_integral(point, v1, v2, v3, dz1_potential_3d_point, NULL);
		} 

		result[0] = Ex;
		result[1] = Ey;
		result[2] = Ez;
}

void
field_3d_traceable(double point[3], double result[3], void *args_p) {

	struct field_evaluation_args *args = (struct field_evaluation_args*)args_p;
	field_3d(point, result, args->vertices, args->charges, args->N_vertices);
}

EXPORT size_t
trace_particle_3d(double *times_array, double *pos_array, double bounds[3][2], double atol,
	double *vertices, double *charges, size_t N_vertices) {

	struct field_evaluation_args args = { vertices, charges, N_vertices };
				
	return trace_particle(times_array, pos_array, field_3d_traceable, bounds, atol, (void*) &args);
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
field_3d_derivs_traceable(double point[3], double field[3], void *args_p) {
	struct field_derivs_args *args = (struct field_derivs_args*) args_p;
	field_3d_derivs(point, field, args->z_interpolation, args->axial_coefficients, args->N_z);
}

EXPORT size_t
trace_particle_3d_derivs(double *times_array, double *pos_array, double bounds[3][2], double atol,
	double *z_interpolation, double *axial_coefficients, size_t N_z) {

	struct field_derivs_args args = { z_interpolation, axial_coefficients, N_z };
	
	return trace_particle(times_array, pos_array, field_3d_derivs_traceable, bounds, atol, (void*) &args);
}


//////////////////////////////// SOLVER

enum ExcitationType{
    VOLTAGE_FIXED = 1,
    VOLTAGE_FUN = 2,
    DIELECTRIC = 3,
    FLOATING_CONDUCTOR = 4};


double legendre(int N, double x) {
	switch(N) {
		case 0:
			return 1;
		case 1:
			return x;
		case 2:
			return (3*pow(x,2)-1)/2.;
		case 3:
			return (5*pow(x,3) -3*x)/2.;
		case 4:
			return (35*pow(x,4)-30*pow(x,2)+3)/8.;
		case 5:
			return (63*pow(x,5)-70*pow(x,3)+15*x)/8.;
		case 6:
			return (231*pow(x,6)-315*pow(x,4)+105*pow(x,2)-5)/16.;
		case 7:
			return (429*pow(x,7)-693*pow(x,5)+315*pow(x,3)-35*x)/16.;
		case 8:
			return (6435*pow(x,8) - 12012*pow(x,6) + 6930*pow(x,4) - 1260*pow(x,2) + 35) / 128;
		/*case 9:
			return (12155*pow(x,9) - 25740*pow(x,7) + 18018*pow(x,5) - 4620*pow(x,3) + 315*x) / 128;
		case 10:
			return (46189*pow(x,10) - 109395*pow(x,8) + 90090*pow(x,6) - 30030*pow(x,4) + 3465*pow(x,2) - 63) / 256;
		case 11:
			return (88179*pow(x,11) - 230945*pow(x,9) + 218790*pow(x,7) - 90090*pow(x,5) + 15015*pow(x,3) - 693*x) / 256;
		case 12:
			return (676039*pow(x,12) - 1939938*pow(x,10) + 2078505*pow(x,8) - 1021020*pow(x,6) + 225225*pow(x,4) - 18018*pow(x,2) + 231) / 1024;
		case 13:
			return (1300075*pow(x,13) - 4056234*pow(x,11) + 4849845*pow(x,9) - 2771340*pow(x,7) + 765765*pow(x,5) - 90090*pow(x,3) + 3003*x) / 1024;
		case 14:
			return (5014575*pow(x,14) - 16900975*pow(x,12) + 22309287*pow(x,10) - 14549535*pow(x,8) + 4849845*pow(x,6) - 765765*pow(x,4) + 45045*pow(x,2) - 429) / 2048;
		case 15:
			return (9694845*pow(x,15) - 35102025*pow(x,13) + 50702925*pow(x,11) - 37182145*pow(x,9) + 14549535*pow(x,7) - 2909907*pow(x,5) + 255255*pow(x,3) - 6435*x) / 2048;
		case 16:
			return (300540195*pow(x,16) - 1163381400*pow(x,14) + 1825305300*pow(x,12) - 1487285800*pow(x,10) + 669278610*pow(x,8) - 162954792*pow(x,6) + 19399380*pow(x,4) - 875160*pow(x,2) + 6435) / 32768;*/
	}
	exit(1);
}


// Modified weight taking into account that the charge contribution
// is a sum of Legendre polynomials.
double legendre_log_weight(int k, int l, double legendre_arg) {

	double sum_ = 0.0;

	for(int i = 0; i < N_QUAD_2D; i++)
		sum_ += GAUSS_QUAD_WEIGHTS[k] * GAUSS_LOG_QUAD_WEIGHTS[l] * (2*i + 1)/2. * legendre(i, GAUSS_QUAD_POINTS[k]) * legendre(i, legendre_arg);
	
	return sum_;
}


double log_integral(
	double *v1,
	double *v2,
	double *v3,
	double *v4,
	int row, int k,
	integration_cb_2d callback, void *args) {
	
	double s = GAUSS_QUAD_POINTS[row];
	
	double spos[2], jac;
	position_and_jacobian_radial(s, v1, v2, v3, v4, spos, &jac);
	
	double spos_left1[2];
	position_and_jacobian_radial( -1 + 2*(s+1)/3., v1, v2, v3, v4, spos_left1, &jac);
	
	double spos_left2[2];
	position_and_jacobian_radial( -1 + (s+1)/3., v1, v2, v3, v4, spos_left2, &jac);
	
	double spos_right1[2];
	position_and_jacobian_radial( s + (1-s)/3, v1, v2, v3, v4, spos_right1, &jac);
	
	double spos_right2[2];
	position_and_jacobian_radial( s + 2*(1-s)/3, v1, v2, v3, v4, spos_right2, &jac);

	double integration_sum = 0.0;
	
	// Logarithmic integration using improved quadrature weights
	// split the integration around the singularity
	for(int l = 0; l < N_LOG_QUAD_2D; l++) {
		
		// To left direction
		double local_alpha = 2*GAUSS_LOG_QUAD_POINTS[l] - 1;
		double global_alpha = s + GAUSS_LOG_QUAD_POINTS[l]*(-s-1);
		
		assert( (-1<local_alpha) && (local_alpha<1) );
		assert( (-1<global_alpha) && (global_alpha<1) );
		
		double pos[2], jac;
		position_and_jacobian_radial(local_alpha, spos, spos_left1, spos_left2, v1, pos, &jac);
		double pot_ring = callback(spos[0], spos[1], pos[0], pos[1], args);
		integration_sum += 2*jac * legendre_log_weight(k, l, global_alpha) * pot_ring;
		// To right direction
		global_alpha = s + GAUSS_LOG_QUAD_POINTS[l]*(1-s);
			
		assert( (-1<local_alpha) && (local_alpha<1) );
		assert( (-1<global_alpha) && (global_alpha<1) );
		
		position_and_jacobian_radial(local_alpha, spos, spos_right1, spos_right2, v4, pos, &jac);
		pot_ring = callback(spos[0], spos[1], pos[0], pos[1], args);
		integration_sum += 2*jac * legendre_log_weight(k, l, global_alpha) * pot_ring;
	}
	
	return integration_sum;
}

void fill_self_voltages(double *matrix, 
                        vertices_2d line_points,
						uint8_t *excitation_types,
						double *excitation_values,
						size_t N_lines,
						size_t N_matrix,
                        int lines_range_start, 
                        int lines_range_end) {
	 
	for(int i = lines_range_start; i <= lines_range_end; i++) {
		
		double *v1 = &line_points[i][0][0];
		double *v2 = &line_points[i][2][0];
		double *v3 = &line_points[i][3][0];
		double *v4 = &line_points[i][1][0];
		
		enum ExcitationType type_ = excitation_types[i];
			
		if (type_ == VOLTAGE_FIXED || type_ == VOLTAGE_FUN || type_ == FLOATING_CONDUCTOR) {
			for(int l = 0; l < N_QUAD_2D; l++) 
			for(int k = 0; k < N_QUAD_2D; k++) {

				matrix[(N_QUAD_2D*i + l)*N_matrix + N_QUAD_2D*i + k] = log_integral(v1, v2, v3, v4, l, k, potential_radial_ring, NULL);
			}
		}
		else if(type_ == DIELECTRIC) {
			for(int l = 0; l < N_QUAD_2D; l++)  {
				for(int k = 0; k < N_QUAD_2D; k++) {

					double normal[2];
					higher_order_normal_radial(GAUSS_QUAD_POINTS[l], v1, v2, v3, v4, normal);
					double K = excitation_values[i];

					struct {double *normal; double K;} args = {normal, K};

					matrix[(N_QUAD_2D*i + l)*N_matrix + N_QUAD_2D*i + k] = log_integral(v1, v2, v3, v4, l, k, field_dot_normal_radial, &args);
				}
				// When working with dielectrics, the constraint is that
				// the electric field normal must sum to the surface charge.
				// The constraint is satisfied by subtracting the integral
				// over the charge from the line element.
				matrix[(N_QUAD_2D*i + l)*N_matrix + N_QUAD_2D*i + l] -= 1;
			}
		}
	}
}

EXPORT void add_floating_conductor_constraints_radial(double *matrix, vertices_2d vertices, size_t N_matrix, int64_t *indices, size_t N_indices, int row) {
	for(int j = 0; j < N_indices; j++) {
		int i = indices[j];
			
		double *v1 = &vertices[i][0][0];
		double *v2 = &vertices[i][2][0]; // Strange ordering following from GMSH line4 element
		double *v3 = &vertices[i][3][0];
		double *v4 = &vertices[i][1][0];
			
		// An extra unknown voltage is added to the matrix for every floating conductor.
		// The column related to this unknown voltage is positioned at the rightmost edge of the matrix.
		// If multiple floating conductors are present the column lives at -len(floating) + i
		for(int k = 0; k < N_QUAD_2D; k++) {
			double pos[2], jac;
			position_and_jacobian_radial(GAUSS_QUAD_POINTS[k], v1, v2, v3, v4, pos, &jac);
			
			matrix[(N_QUAD_2D*i + k)*N_matrix + N_matrix - row - 1] = -1;
			// See charge_radial function
			matrix[(N_matrix - row - 1)*N_matrix + N_QUAD_2D*i + k] = 2*M_PI*pos[0]*GAUSS_QUAD_WEIGHTS[k]*jac;
		}
	}
}



EXPORT void fill_matrix_radial(double *matrix, 
						vertices_2d line_points,
                        uint8_t *excitation_types, 
                        double *excitation_values, 
						size_t N_lines,
						size_t N_matrix,
                        int lines_range_start, 
                        int lines_range_end) {
    
	assert(lines_range_start < N_lines && lines_range_end < N_lines);
	assert(N_matrix >= N_QUAD_2D*N_lines);
		
    for (int i = lines_range_start; i <= lines_range_end; i++) {
		
		double *target_v1 = &line_points[i][0][0];
		double *target_v2 = &line_points[i][2][0];
		double *target_v3 = &line_points[i][3][0];
		double *target_v4 = &line_points[i][1][0];
		
		enum ExcitationType type_ = excitation_types[i];
			
		if (type_ == VOLTAGE_FIXED || type_ == VOLTAGE_FUN || type_ == FLOATING_CONDUCTOR) {
			for (int j = 0; j < N_lines; j++) {
				
				if (i == j) continue;
					
				double *v1 = &line_points[j][0][0];
				double *v2 = &line_points[j][2][0]; // Strange ordering following from GMSH line4 element
				double *v3 = &line_points[j][3][0];
				double *v4 = &line_points[j][1][0];
					
				for(int l = 0; l < N_QUAD_2D; l++) {
					double target[2], jac_t;
					position_and_jacobian_radial(GAUSS_QUAD_POINTS[l], target_v1, target_v2, target_v3, target_v4, target, &jac_t);
						
					for(int k = 0; k < N_QUAD_2D; k++) {
						
						double pos[2], jac;
						position_and_jacobian_radial(GAUSS_QUAD_POINTS[k], v1, v2, v3, v4, pos, &jac);
						
						matrix[(N_QUAD_2D*i + l)*N_matrix + N_QUAD_2D*j + k] = GAUSS_QUAD_WEIGHTS[k]*jac*potential_radial_ring(target[0], target[1], pos[0], pos[1], NULL);
					}
				}
			} 
		}
		else if(type_ == DIELECTRIC) {
			            
            for (int j = 0; j < N_lines; j++) {

				if(i == j) continue;
					
				double *v1 = &line_points[j][0][0];
				double *v2 = &line_points[j][2][0]; // Strange ordering following from GMSH line4 element
				double *v3 = &line_points[j][3][0];
				double *v4 = &line_points[j][1][0];
				
				for(int l = 0; l < N_QUAD_2D; l++) {
					
					double normal[2];
					higher_order_normal_radial(GAUSS_QUAD_POINTS[l], target_v1, target_v2, target_v3, target_v4, normal);
					double K = excitation_values[i];
					
					struct {double *normal; double K;} args = {normal, K};

					double target[2], jac_t;
					position_and_jacobian_radial(GAUSS_QUAD_POINTS[l], target_v1, target_v2, target_v3, target_v4, target, &jac_t);
					
					for(int k = 0; k < N_QUAD_2D; k++) {
						
						double pos[2], jac;
						position_and_jacobian_radial(GAUSS_QUAD_POINTS[k], v1, v2, v3, v4, pos, &jac);
						matrix[(N_QUAD_2D*i + l)*N_matrix + N_QUAD_2D*j + k] = GAUSS_QUAD_WEIGHTS[k]*jac*field_dot_normal_radial(target[0], target[1], pos[0], pos[1], &args);
					}
				}
            }
		}
		else {
		    printf("ExcitationType unknown");
            exit(1);
		}
	}
	
	fill_self_voltages(matrix, line_points, excitation_types, excitation_values, N_lines, N_matrix, lines_range_start, lines_range_end);
}


EXPORT void fill_matrix_3d(double *matrix, 
                    double *triangle_points_p, 
                    uint8_t *excitation_types, 
                    double *excitation_values, 
					size_t N_lines,
					size_t N_matrix,
                    int lines_range_start, 
                    int lines_range_end) {
    
	assert(lines_range_start < N_lines && lines_range_end < N_lines);
	double (*triangle_points)[3][3] = (double (*)[3][3]) triangle_points_p;
		
    for (int i = lines_range_start; i <= lines_range_end; i++) {
		double *p1 = &triangle_points[i][0][0];
		double *p2 = &triangle_points[i][1][0];
		double *p3 = &triangle_points[i][2][0];
		double target[3] = {(p1[0] + p2[0] + p3[0])/3, (p1[1] + p2[1] + p3[1])/3, (p1[2] + p2[2] + p3[2])/3};
        enum ExcitationType type_ = excitation_types[i];
		 
        if (type_ == VOLTAGE_FIXED || type_ == VOLTAGE_FUN || type_ == FLOATING_CONDUCTOR) {
            for (int j = 0; j < N_lines; j++) {
                double *v1 = &triangle_points[j][0][0];
                double *v2 = &triangle_points[j][1][0];
                double *v3 = &triangle_points[j][2][0];
                matrix[i*N_matrix + j] = triangle_integral_potential_3d_point(target, v1, v2, v3);
            }
        } 
        else if (type_ == DIELECTRIC) {
            double normal[3];
            normal_3d(p1, p2, p3, normal);
            double K = excitation_values[i];
            
            for (int j = 0; j < N_lines; j++) {
				double *v1 = &triangle_points[j][0][0];
                double *v2 = &triangle_points[j][1][0];
                double *v3 = &triangle_points[j][2][0];
				// See comments in 'fill_matrix_2d'.
                double factor = (2*K - 2) / (M_PI*(1 + K));
                matrix[i*N_matrix + j] = factor * triangle_integral(target, v1, v2, v3, field_dot_normal_3d, normal);
				 
                if (i == j) matrix[i*N_matrix + j] -= 1.0;
            }
        }
        else {
            printf("ExcitationType unknown");
            exit(1);
        }
    }
}

EXPORT bool
xy_plane_intersection_2d(double *positions_p, size_t N_p, double result[4], double z) {

	double (*positions)[4] = (double (*)[4]) positions_p;

	for(int i = N_p-1; i > 0; i--) {
	
		double z1 = positions[i-1][1];
		double z2 = positions[i][1];
		
		if(fmin(z1, z2) <= z && z <= fmax(z1, z2)) {
			double ratio = fabs( (z-z1)/(z1-z2) );
			
			for(int k = 0; k < 4; k++)
				result[k] = positions[i-1][k] + ratio*(positions[i][k] - positions[i-1][k]);

			return true;
		}
	}

	return false;
}

EXPORT bool
xy_plane_intersection_3d(double *positions_p, size_t N_p, double result[6], double z) {

	double (*positions)[6] = (double (*)[6]) positions_p;

	for(int i = N_p-1; i > 0; i--) {
	
		double z1 = positions[i-1][2];
		double z2 = positions[i][2];
		
		if(fmin(z1, z2) <= z && z <= fmax(z1, z2)) {
			double ratio = fabs( (z-z1)/(z1-z2) );
			
			for(int k = 0; k < 6; k++)
				result[k] = positions[i-1][k] + ratio*(positions[i][k] - positions[i-1][k]);

			return true;
		}
	}

	return false;
}










