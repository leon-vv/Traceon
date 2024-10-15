#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)

#include <Python.h>
PyMODINIT_FUNC PyInit_traceon_backend(void) {
	return NULL;
}

#else
#define EXPORT extern
#endif

#define INLINE EXPORT inline

#if defined(__clang__)
	#define UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__) || defined(__GNUG__)
	#define UNROLL _Pragma("GCC unroll 100")
#else
	#define UNROLL
#endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif


// python -c "from scipy.constants import m_e, e; print(-e/m_e);"
const double EM = -175882001077.2163; // Electron charge over electron mass
// python -c "from scipy.constants import mu_0; print(mu_0);"
const double MU_0 = 1.25663706212e-06;


enum ExcitationType{
    VOLTAGE_FIXED = 1,
    VOLTAGE_FUN = 2,
    DIELECTRIC = 3,
	CURRENT=4,
	MAGNETOSTATIC_POT=5,
	MAGNETIZABLE=6};




