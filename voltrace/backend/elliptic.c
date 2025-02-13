#include <math.h>

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


