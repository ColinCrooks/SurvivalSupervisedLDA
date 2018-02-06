// Latent Dirichlet Allocation supervised by penalised Cox proportional hazards modelling with optional learning of asymmetrical priors.

//This has been modified from the original code (C) Copyright 2009, Chong Wang, David Blei and Li Fei-Fei ([1] Blei DM, McAuliffe JD. Supervised Topic Models. Adv Neural Inf Process Syst 20 2007:121–8.) and modified following the algorithms developed by Ye et al. 2014 ([1] Ye S, Dawson JA, Kendziorski C. Extending information retrieval methods to personalized genomic-based studies of disease. Cancer Inform 2014;13:85–95. doi:10.4137/CIN.S16354.)

// Modifications by  (C) Copyright 2017 Colin Crooks (colin.crooks@nottingham.ac.uk)

// This file is part of sslda.

//sslda is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your
// option) any later version.

//sslda is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more cov_betails.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "utils.h"


/*
 * given log(a) and log(b), return log(a + b)
 *
 */

double log_sum(double log_a, double log_b)
{
    double v;

    if (log_a < log_b)
        v = log_b+log(1 + exp(log_a-log_b));
    else
        v = log_a+log(1 + exp(log_b-log_a));

    return v;
}

/**
* Procedure from slda to calculate the value of the trigamma, the second
* derivative of the loggamma function. Accepts positive matrices.
* From Abromowitz and Stegun.  Uses formulas 6.4.11 and 6.4.12 with
* recurrence formula 6.4.6.  Each requires workspace at least 5
* times the size of X.
*
**/
double trigamma(double x)
{
	double p;
	int i;

	x = x + 6;
	p = 1 / (x*x);
	p = (((((0.075757575757576*p - 0.033333333333333)*p + 0.0238095238095238)
		*p - 0.033333333333333)*p + 0.166666666666667)*p + 1) / x + 0.5*p;
	for (i = 0; i<6; i++)
	{
		x = x - 1;
		p = 1 / (x*x) + p;
	}
	return(p);
}

/*  $Id: chsolve2.c 11376 2009-12-14 22:53:57Z therneau $
**
** Solve the equation Ab = y, where the cholesky decomposition of A and y
**   are the inputs.
**
** Input  **matrix, which contains the chol decomp of an n by n
**   matrix in its lower triangle.
**        y[n] contains the right hand side
**
**  y is overwriten with b
**
**  Terry Therneau
*/


void chsolve2(double **matrix, int n, double *y)
{
	register int i, j;
	register double temp;

	/*
	** solve Fb =y
	*/
	for (i = 0; i<n; i++) {
		temp = y[i];
		for (j = 0; j<i; j++)
			temp -= y[j] * matrix[i][j];
		y[i] = temp;
	}
	/*
	** solve DF'z =b
	*/
	for (i = (n - 1); i >= 0; i--) {
		if (matrix[i][i] == 0)  y[i] = 0;
		else {
			temp = y[i] / matrix[i][i];
			for (j = i + 1; j<n; j++)
				temp -= y[j] * matrix[j][i];
			y[i] = temp;
		}
	}
}


/* $Id: cholesky2.c 11357 2009-09-04 15:22:46Z therneau $
**
** subroutine to do Cholesky decompostion on a matrix: C = FDF'
**   where F is lower triangular with 1's on the diagonal, and D is diagonal
**
** arguments are:
**     n         the size of the matrix to be factored
**     **matrix  a ragged array containing an n by n submatrix to be factored
**     toler     the threshold value for detecting "singularity"
**
**  The factorization is returned in the lower triangle, D occupies the
**    diagonal and the upper triangle is left undisturbed.
**    The lower triangle need not be filled in at the start.
**
**  Return value:  the rank of the matrix (non-negative definite), or -rank
**     it not SPD or NND
**
**  If a column is deemed to be redundant, then that diagonal is set to zero.
**
**   Terry Therneau
*/

int cholesky2(double **matrix, int n, double toler)
{
	double temp;
	int  i, j, k;
	double eps, pivot;
	int rank;
	int nonneg;

	nonneg = 1;
	eps = 0;
	for (i = 0; i<n; i++) {
		if (matrix[i][i] > eps)  eps = matrix[i][i];
		for (j = (i + 1); j<n; j++)  matrix[j][i] = matrix[i][j];
	}
	eps *= toler;

	rank = 0;
	for (i = 0; i<n; i++) {
		pivot = matrix[i][i];
		if (pivot < eps) {
			matrix[i][i] = 0;
			if (pivot < -8 * eps) nonneg = -1;
		}
		else  {
			rank++;
			for (j = (i + 1); j<n; j++) {
				temp = matrix[j][i] / pivot;
				matrix[j][i] = temp;
				matrix[j][j] -= temp*temp*pivot;
				for (k = (j + 1); k<n; k++) matrix[k][j] -= temp*matrix[k][i];
			}
		}
	}
	return(rank * nonneg);
}

