// Latent Dirichlet Allocation supervised by penalised Cox proportional hazards modelling with optional learning of asymmetrical priors.

//This has been developed from the original code (C) Copyright 2009, Chong Wang, David Blei and Li Fei-Fei ([1] Blei DM, McAuliffe JD. Supervised Topic Models. Adv Neural Inf Process Syst 20 2007:121–8.) and modified following the algorithms developed by Ye et al. 2014 ([1] Ye S, Dawson JA, Kendziorski C. Extending information retrieval methods to personalized genomic-based studies of disease. Cancer Inform 2014;13:85–95. doi:10.4137/CIN.S16354.)

// Modifications by Colin Crooks (colin.crooks@nottingham.ac.uk)

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


int cox_lasso(
	double * beta,
	double * zbeta,
	double ** var, int nvar,
	double lambda,
	suffstats * ss,
	double * f,
	const settings* setting
	);

double cox_lasso_cross_val(
	int group,
	double * newbeta,
	double ** var,
	double * zbeta,
	double * step,
	int nvar,
	double lambda,
	const suffstats * ss,
	const settings* setting
	);

int cox_net(
	double alphanet, 
	double * beta, 
	double * zbeta, 
	double ** var, 
	int nvar, 
	double lambda,
	suffstats * ss,
	double * f,
	const settings* setting
	);

double cox_net_cross_val(
	double alphanet,
	int group,
	double * newbeta,
	double ** var,
	double * zbeta,
	double * step,
	int nvar,
	double lambda,
	const suffstats * ss,
	const settings* setting
	);

int coxfit(double * beta, int nvar,
	double ** var,
	double * zbeta,
	const suffstats * ss, double * f, const settings* setting);



//double beta_f(const gsl_vector * x, void * opt_param);
//
//void beta_df(const gsl_vector * x, void * opt_param, gsl_vector * df);
//
//void  beta_ddf(const gsl_vector * x, void * opt_param, double * f, gsl_vector * df);


#endif // OPT_H_INCLUDED

