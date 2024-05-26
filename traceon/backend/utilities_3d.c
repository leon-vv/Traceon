INLINE double
dot_3d(double v1[3], double v2[3]) {
	return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

INLINE double
norm_3d(double x, double y, double z) {
	return sqrt(x*x + y*y + z*z);
}

INLINE double
distance_3d(double v0[3], double v1[3]) {
	return norm_3d(v0[0]-v1[0], v0[1]-v1[1], v0[2]-v1[2]);
}

INLINE void
normalize_3d(double *v) {
	double length = norm_3d(v[0], v[1], v[2]);
	v[0] /= length;
	v[1] /= length;
	v[2] /= length;
	assert(fabs(norm_3d(v[0], v[1], v[2])-1.) < 1e-8);
}

INLINE void
cross_product_3d(double v1[3], double v2[3], double out[3]) {
	double v1x = v1[0], v1y = v1[1], v1z = v1[2];
	double v2x = v2[0], v2y = v2[1], v2z = v2[2];

	out[0] = v1y*v2z-v1z*v2y;
	out[1] = v1z*v2x-v1x*v2z;
	out[2] = v1x*v2y-v1y*v2x;
}

