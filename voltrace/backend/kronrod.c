#include <stdio.h>
#include <math.h>

#define G7_WEIGHTS_SIZE 4
#define K15_WEIGHTS_SIZE 8

const double G7_WEIGHTS[G7_WEIGHTS_SIZE] = {
    0.417959183673469,
    0.381830050505119,
    0.279705391489277,
    0.129484966168870,
};

const double G7_NODES[G7_WEIGHTS_SIZE] = {
    0.000000000000000,
    0.405845151377397,
    0.741531185599394,
    0.949107912342759,
};

const double K15_WEIGHTS[K15_WEIGHTS_SIZE] = {
    0.209482141084728,
    0.204432940075299,
    0.190350578064785,
    0.169004726639268,
    0.140653259715526,
    0.104790010322250,
    0.063092092629979,
    0.022935322010529,
};

const double K15_NODES[K15_WEIGHTS_SIZE] = {
    0.000000000000000,
    0.207784955007898,
    0.405845151377397,
    0.586087235467691,
    0.741531185599394,
    0.864864423359769,
    0.949107912342759,
    0.991455371120813,
};

EXPORT double kronrod_adaptive(double (*f)(double, void*), double a, double b, void* args, double abs_tol, double rel_tol) {
    double result = 0.0;
    double current_start = a;
    double current_end = b;

    while (current_start < b) {
        double c = (current_start + current_end) / 2.0;
        double h = (current_end - current_start) / 2.0;

        double g7 = h * G7_WEIGHTS[0] * f(c, args);
        double k15 = h * K15_WEIGHTS[0] * f(c, args);

        for (int i = 1; i < G7_WEIGHTS_SIZE; i++) {
            double x = h * G7_NODES[i];
            g7 += h * G7_WEIGHTS[i] * (f(c - x, args) + f(c + x, args));
        }

        for (int i = 1; i < K15_WEIGHTS_SIZE; i++) {
            double x = h * K15_NODES[i];
            k15 += h * K15_WEIGHTS[i] * (f(c - x, args) + f(c + x, args));
        }

        double error = fabs(k15 - g7);

        if (error <= abs_tol || error <= rel_tol * fabs(k15)) {
            result += k15;
            current_start = current_end;
            current_end = b;
        } else {
            current_end = c;
        }
    }

    return result;
}
