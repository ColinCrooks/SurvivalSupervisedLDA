// (C) Copyright 2009, Chong Wang, David Blei and Li Fei-Fei

// written by Chong Wang, chongw@cs.princeton.edu

// This file is part ofsslda.

//sslda is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

//sslda is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more cov_betails.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
#ifndef OPT_H_INCLUDED
#define OPT_H_INCLUDED

# if defined(_MSC_VER)
#  define isn(x) (_isnan(x))
#  define isf(x) (_finite(x))
# else
#  define isn(x) (isnan(x))
#  define isf(x) (isfinite(x))
#endif

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>

#include "sslda.h"
#include "corpus.h"


double opt_alpha(double * init_a, double *  ss, int D, int K, const settings* setting);

int cox_reg(
	double * beta, 
	double * zbeta, 
	double ** var, 
	int nvar , 
	double lambda, 
	const suffstats * ss,  
	double * f, 
	const settings* setting
	);

double cox_reg_cross_val(
	int group, 
	double * newbeta, 
	double ** var,
	int nvar, 
	double lambda, 
	const suffstats * ss, 
	const settings* setting
	);

int cox_reg_sparse(
	double * beta,
	double * zbeta,
	const corpus *c,
	int nvar,
	double lambda,
	const suffstats * ss,
	double * f,
	const settings* setting);



#endif // OPT_H_INCLUDED

