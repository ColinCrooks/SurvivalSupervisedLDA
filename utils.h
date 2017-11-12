#ifndef UTILS_H
#define UTILS_H

# if defined(_MSC_VER)
#  define isn(x) (_isnan(x))
#  define isf(x) (_finite(x))
# else
#  define isn(x) (isnan(x))
#  define isf(x) (isfinite(x))
#endif

#include <math.h>

double log_sum(double log_a, double log_b);
double trigamma(double x);
int cholesky2(double **matrix, int n, double toler);
void chsolve2(double **matrix, int n, double *y);

#endif

