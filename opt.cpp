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
#include "opt.h"
#include "sslda.h"
#include "utils.h"




/*
* newtons method to find maximum likelihood value for alpha given the allocations in gamma
*
*/
/*
double opt_alpha(double init_a, double ss, int D, int K, const settings* setting)
{
	double a, log_a;
	double oldf = 0.0, f = 0.0, df, d2f;

	log_a = log(init_a);
	for (int iter = 0; iter <= setting->MSTEP_MAX_ITER; iter++)
	{
		iter++;
		oldf = f;
		a = exp(log_a);
		if (isnan(a)) 
		{
			init_a = init_a * 10;
			printf("warning : alpha is nan; new init = %5.5f\n", init_a);
			a = init_a;
			log_a = log(a);
		}
		*/
		//f = (static_cast<double>(D) * ((boost::math::lgamma(K) * a) - static_cast<double>(K) * boost::math::lgamma(a))) + ((a - 1) * ss); // num_docs * (gamma(num_topics * alpha) / num_topics * gamma(alpha)) * exp((alpha - 1) * observed allocations) 
		//df = (static_cast<double>(D) * ((static_cast<double>(K) * boost::math::digamma(static_cast<double>(K) * a))- static_cast<double>(K) * boost::math::digamma(a))) + ss; // num_docs * (num_topics *(digamma(num_topics * alpha) / num_topics * digamma(alpha))) * exp(observed allocations)
		//d2f = static_cast<double>(D) * ((static_cast<double>(K) * static_cast<double>(K) * trigamma(static_cast<double>(K) * a)) - (static_cast<double>(K) * trigamma(a))); // num_docs * (num_topics^2 *(digamma(num_topics * alpha) / num_topics * digamma(alpha)) * exp(observed allocations) 
		/*log_a = log_a - (df / (d2f * a + df));
		if (log_a < -10) return exp(-10);
		if (log_a > -0.1) return exp(-0.1);
		if (fabs(1 - (f/oldf)) <= setting->MAX_EPS) return(exp(log_a));
	} 
	return(exp(log_a));
}*/
// Works but convergence becomes unstable when some groups become unpopulated.
double opt_alpha(double * a, double * ss, int D, int K, const settings* setting)
{
	double init_a = 1.0/static_cast<double>(K);
	double * log_a = new double[K];
	for (int k = 0; k < K; k++)
	{
		log_a[k] = log(init_a);
		a[k] = init_a;
	}

	double oldf = 0.0, f = 0.0, df = 0.0, d2f = 0.0;

	int iter = 0;

	for (iter = 1; iter <= setting->MSTEP_MAX_ITER ; iter++)
	{
		oldf = f;
		f = 0.0;
		int stop = 0;		
		for (int k = 0; k < K; k++)
		{
			
			if (isnan(a[k]) || a[k]==0)
			{
				init_a /= 10.0;
				a[k] = init_a;
				log_a[k] = log(a[k]);
				stop += 1;
			}
			double a_sum = 0.0;
			double lgsum = 0.0;
			double ss_sum = 0.0;
			for (int k1 = 0; k1 < K; k1++)
			{
				a_sum += a[k1];
				lgsum += boost::math::lgamma(a[k1]);
				ss_sum += (a[k1] - 1) * ss[k1];
			}
			f += static_cast<double>(D)* (boost::math::lgamma(a_sum) - (lgsum)) + ss_sum;
			df = static_cast<double>(D)* ( boost::math::digamma(a_sum) - boost::math::digamma(a[k])) + ss[k];
			d2f = static_cast<double>(D)* (trigamma(a_sum) -  trigamma(a[k]));
			log_a[k] = log_a[k] -  df / (d2f * a[k] + df) ;
			//std::cout << "alpha " << a[k] << "; a_sum " << a_sum <<
			//	"; f " << f << "; df " << df << "; d2f " << d2f << "; log_a[k] " << log_a[k] << " ss[k] " << ss[k] << std::endl;
			if (log_a[k] < -10.0 || isnan(log_a[k]))
			{
				log_a[k] = -10.0;
				a[k] = exp(log_a[k]);
				stop += 1;
			}
			else if (log_a[k] > 2 )
			{
				log_a[k] = 2 ;
				a[k] = exp(log_a[k]);
				stop += 1;
			}
			else
			{
				a[k] = exp(log_a[k]);
			}
		}

		if (fabs(1 - (f / oldf)) <= setting->MAX_EPS || stop == K)
			break;
		//printf("alpha maximization : %5.5f   %5.5f\n", f, df);
	}
	for (int k = 0; k < K; k++)	a[k] = exp(log_a[k]);
	delete[]  log_a;
	log_a = NULL;
	return -f;
}

int cox_reg(double * beta, double * zbeta, double ** var, int nvar, double lambda, const suffstats * ss, double * f, const settings* setting)
{
	//Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043
	int i,k, person, iter;
	int nused = ss->num_docs, lastvar = 0;
	int ntimes = ss->times[nused - 1] - ss->time_start + 1;
	double * denom = new double[ntimes];
	double * a = new double[ntimes];
	double * cdiag = new double[ntimes];
	double  risk=0.0, temp=0.0, temp2=0.0,	loglik = 0.0,  newlk=0.0, d2 = 0.0, efron_wt = 0.0;
	double eta = 0.0, a2 = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
	double * newbeta = new double [nvar];
	double * step = new double [nvar];

	for (i = 0; i < nvar; i++)
	{
		newbeta[i] = beta[i];
		step[i] = 1.0;
	}

	for (person=nused-1; person>=0; person--)
	{

		zbeta[person] = 0.0;
		for (i = 0; i < nvar; i++)
			zbeta[person] += var[person][i] * newbeta[i];
		zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
		zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
	}
	for (iter=1; iter<=setting->MSTEP_MAX_ITER; iter++) 
	{
		newlk = -(log(sqrt(lambda)) * nvar);
		for (i = 0; i < nvar; i++)
		{

			/*
			** The data is sorted from smallest time to largest
			** Start at the largest time, accumulating the risk set 1 by 1
			*/
			for (int r = ntimes - 1; r >= 0; r--)
			{
				denom[r] = 0.0;
				a[r] = 0.0;
				cdiag[r] = 0.0;
			}
			efron_wt = 0.0;
			
			gdiag = 0.0;
			hdiag = 0.0;
			a2 = 0.0;
			cdiag2 = 0.0;

			for (person = nused - 1; person >= 0; person--)
			{
				int time_index_entry = (ss->time0[person]) - (ss->time_start);
				int time_index_exit = (ss->times[person]) - (ss->time_start);

				zbeta[person] -= eta * var[person][lastvar];
				zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
				zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
				/*if (isn(zbeta[person]))
				{
					zbeta[person] = 0;
					std::cout << "zbeta reset as eta = " << eta << std::endl;
					for (i = 0; i<nvar; i++)
						std::cout<< newbeta[i]<<",";
					break;
				}*/
				risk = exp(zbeta[person]);

				//cumumlative sums for all patients
				for (int r = time_index_entry; r <= time_index_exit; r++)
				{
					denom[r] += risk;
					a[r] += risk * var[person][i];
					cdiag[r] += risk * var[person][i] * var[person][i];
				}
				if (ss->labels[person] > 0)
				{
					//cumumlative sums for event patients
					newlk += zbeta[person];
					gdiag -= var[person][i];

					efron_wt += risk; /* sum(denom) for tied deaths*/
					a2 += risk * var[person][i];
					cdiag2 += risk * var[person][i] * var[person][i];
				}
				if (ss->mark[person][setting->CROSSVAL] > 0)
				{  /* once per unique death time */
					for (k = 0; k < ss->mark[person][setting->CROSSVAL]; k++)
					{
						temp = static_cast<double>(k)
							/ static_cast<double>(ss->mark[person][setting->CROSSVAL]);
						d2 = denom[time_index_exit] - (temp * efron_wt); /* sum(denom) adjusted for tied deaths*/
						newlk -= log(d2);
						temp2 = (a[time_index_exit] - (temp * a2)) / d2;
						gdiag += temp2;
						hdiag += ((cdiag[time_index_exit] - (temp * cdiag2)) / d2) -
							(temp2 * temp2);
					}
					efron_wt = 0.0;
					a2 = 0.0;
					cdiag2 = 0.0;
				}
			}   /* end  of accumulation loop  */

			eta = (gdiag + (newbeta[i] / lambda)) / (hdiag + (1.0 / lambda));
			if (fabs(eta) > step[i])	
				eta = (eta > 0.0) ? step[i] : -step[i];

			step[i] = ((2.0 * fabs(eta)) > (step[i] / 2.0)) ? 2.0 * fabs(eta) : (step[i] / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			newbeta[i] -= eta;
			lastvar = i;
		}
		
		for( i = 0 ; i < nvar ; i++)
			newlk -= (newbeta[i] * newbeta[i]) / (2.0 * lambda);
		if (fabs(1.0 - (newlk / loglik)) <= setting->MAX_EPS) break;
		loglik = newlk;
	}   /* return for another iteration */
	

	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] -= eta * var[person][lastvar];
		zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
		zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
	}

	* f = loglik;
	for (i=0; i<nvar; i++)
		beta[i] = newbeta[i];	


	delete [] newbeta;
	delete [] step;
	delete [] denom;
	delete [] a;
	delete [] cdiag;
	return iter;
}



double cox_reg_cross_val(
	int group, 
	double * newbeta, 
	double ** var,
	int nvar, 
	double lambda, 
	const suffstats * ss, 
	const settings* setting)
{
    int i,k, r, person, iter;
    int nused = ss->num_docs, lastvar = 0;
	int ntimes = ss->times[nused - 1] - ss->time_start + 1;
	double * denom = new double[ntimes];
	double * a = new double[ntimes];
	double * cdiag = new double[ntimes];
	double * zbeta = new double[nused];
 	double * step = new double[ntimes];
    double risk=0.0, temp=0.0, temp2=0.0,	loglik = 0.0,  newlk=0.0;
	double d2 = 0.0, efron_wt = 0.0;
	double eta = 0.0, a2 = 0.0,  cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
	for (k = 0; k < nvar; k++)
		step[k] = 1.0;
	for (person = 0; person < nused; person++)
		zbeta[person] = 0.0;
	for (r = 0; r < ntimes; r++)
		step[r] = 1.0;
	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] = 0.0;
		for (i = 0; i<nvar; i++)
			zbeta[person] += newbeta[i] * var[person][i];
		zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
		zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
	}

	for (iter=1; iter<=setting->MSTEP_MAX_ITER; iter++) 
	{
		newlk = -log(sqrt(lambda)); // New likelihood for current iteration
		for (i=0; i<nvar; i++) 
		{		
			/*
			** The data is sorted from smallest time to largest
			** Start at the largest time, accumulating the risk set 1 by 1
			*/

		

			for (r = ntimes - 1; r >= 0; r--)
			{
				denom[r] = 0.0; //sum(exp(XB)) for remaining patients in risk set
				a[r] = 0.0; // sum(x..j * exp(XB))
				cdiag[r] = 0.0; // sum((x.j)^2 * exp(XB))
			}
			

			efron_wt = 0.0; // sum(denom) for tied deaths
			a2 = 0.0; // sum(x..j * exp(XB)) for tied deaths
			cdiag2 = 0.0; //	sum((x.j)^2 * exp(XB)) for tied deaths

			gdiag = 0.0; 
			hdiag = 0.0; // Hessian matrix diagonal for ij weighted for tied deaths

			for (person = nused - 1; person >= 0; person--)
			{
				int time_index_entry = (ss->time0[person]) - (ss->time_start);
				int time_index_exit = (ss->times[person]) - (ss->time_start);
				if (ss->group[person] != group) //(beta_j-1)
				{
					zbeta[person] -= eta * var[person][lastvar];
					zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
					zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
					risk = exp(zbeta[person]);

					//cumumlative sums for all patients
					for (r = time_index_entry; r <= time_index_exit; r++)
					{
						denom[r] += risk;
						a[r] += risk * var[person][i];
						cdiag[r] += risk * var[person][i] * var[person][i];
					}
					if (isn(denom[time_index_exit]))
						return numeric_limits<double>::lowest();

					if (ss->labels[person] > 0)
					{
						//cumumlative sums for all patients
						newlk += zbeta[person];

						efron_wt += risk;
						gdiag -= var[person][i];
						a2 += risk * var[person][i];
						cdiag2 += risk * var[person][i] * var[person][i];
					}
				}
				 /* once per unique death time */
				int numdeaths = (ss->mark[person][setting->CROSSVAL] - ss->mark[person][group]);
				if (numdeaths > 0)
				{
					for (k = 0; k < numdeaths; k++)
					{
						temp = static_cast<double>(k)
							/ static_cast<double>(numdeaths);
						d2 = denom[time_index_exit] - (temp * efron_wt);
						newlk -= log(d2);
						temp2 = (a[time_index_exit] - (temp * a2)) / d2;
						gdiag += temp2;
						hdiag += ((cdiag[time_index_exit] - (temp * cdiag2)) / d2) -
							(temp2 * temp2);
					}
					efron_wt = 0.0;
					a2 = 0.0;
					cdiag2 = 0.0;
				}
				if (isn(hdiag))
					return numeric_limits<double>::lowest();

			} /* end  of accumulation loop  */

			eta = (gdiag + (newbeta[i] / lambda)) / (hdiag + (1.0 / lambda));
			if (fabs(eta) > step[i])
				eta = (eta > 0.0) ? step[i] : -step[i];
			step[i] = ((2.0 * fabs(eta)) > (step[i] / 2.0)) ? 2.0 * fabs(eta) : (step[i] / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			newbeta[i] -= eta;
			lastvar = i;
			if (isn(newbeta[i]))
				return numeric_limits<double>::lowest();

			/* am I done?
			**   update the betas and test for convergence
			*/
		}
		
		for( i = 0 ; i < nvar ; i++)
			newlk -= (newbeta[i] * newbeta[i]) / (2.0 * lambda); //(l_j-1)
		if(isn(newlk))
			return numeric_limits<double>::lowest();
		
		//if (fabs(1.0 - (loglik / newlk)) <= setting->MAX_EPS) break; //For cross validation just to 1e-6 to save time
		if (fabs(1.0 - (loglik / newlk)) <= 1e-3 /*setting->MAX_EPS*/ ) break;
		loglik = newlk;
	}   /* return for another iteration */
	if (iter == setting->MSTEP_MAX_ITER + 1) 
		return numeric_limits<double>::lowest();
	for (r = ntimes - 1; r >= 0; r--)
		denom[r] = 0.0; //sum(exp(XB)) for remaining patients in risk set

	d2 = 0.0;
	efron_wt = 0.0;
	a2 = 0.0;
	loglik = 0.0; //use for l(b_j-1)
	newlk = 0.0;//use for l_j-1(b_j-1)
	for (person = nused - 1; person >= 0; person--)
	{
		int time_index_entry = (ss->time0[person]) - (ss->time_start);
		int time_index_exit = (ss->times[person]) - (ss->time_start);

		zbeta[person] = 0.0;
		for (i = 0; i < nvar; i++)
			zbeta[person] += newbeta[i] * var[person][i]; // recalculate with final iteration updated betas
		risk = exp(zbeta[person]);
		for (r = time_index_entry; r <= time_index_exit; r++)
			denom[r] += risk;
		if (ss->group[person] != group)
			d2 += risk;
		if (ss->labels[person] > 0)
		{
			efron_wt += risk;
			loglik += zbeta[person];
			if (ss->group[person] != group)
			{
				a2 += risk;
				newlk += zbeta[person];
			}
		}
		if (ss->mark[person][setting->CROSSVAL] > 0)
		{
			for (k = 0; k < ss->mark[person][setting->CROSSVAL]; k++)
				loglik -= log(denom[time_index_exit] - ((static_cast<double>(k)
					/ static_cast<double>(ss->mark[person][setting->CROSSVAL]))
					* efron_wt));
			efron_wt = 0.0;	
			int num_deaths = ss->mark[person][setting->CROSSVAL] - ss->mark[person][group];
			for (k = 0; k < num_deaths; k++)
				newlk -= log(d2 - (static_cast<double>(k) 
					/ static_cast<double>(num_deaths)
						* a2));
			a2 = 0.0;
		}
		
	}
	for( i = 0 ; i < nvar ; i++)
	{
		loglik -= log(sqrt(lambda))  + ((newbeta[i] * newbeta[i]) / (2.0 * lambda)); 
		newlk -= log(sqrt(lambda))  + ((newbeta[i] * newbeta[i]) / (2.0 * lambda)); 
	}
	if(isn(loglik))
		return numeric_limits<double>::lowest();
	if(isn(newlk))
		return numeric_limits<double>::lowest();
	return loglik - newlk; 
	delete[] denom;
	denom = nullptr;
	delete[] a;
	a = nullptr;
	delete[] cdiag;
	cdiag = nullptr;
	delete[] zbeta;
	zbeta = nullptr;
	delete[] step;
	step = nullptr;
}

int cox_reg_sparse(double * beta, double * zbeta, const corpus *c, int nvar, double lambda, const suffstats * ss, double * f, const settings* setting)
{
	//Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043
	int i, k, person, iter;
	int nused = ss->num_docs, lastvar = 0;
	int ntimes = ss->times[nused - 1] - ss->time_start + 1;
	double * denom = new double[ntimes];
	double * a = new double[ntimes];
	double * cdiag = new double[ntimes];
	double  risk = 0.0, temp = 0.0, temp2 = 0.0, loglik = 0.0, newlk = 0.0, d2 = 0.0, efron_wt = 0.0;
	double eta = 0.0, a2 = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
	double * newbeta = new double[nvar];
	double * step = new double[nvar];

	for (i = 0; i < nvar; i++)
	{
		newbeta[i] = beta[i];
		step[i] = 1.0;
	}

	for (person = nused - 1; person >= 0; person--)
	{

		zbeta[person] = 0.0;
		for (i = 0; i < c->docs[person]->length; i++)
			zbeta[person] += c->docs[person]->counts[i] * newbeta[c->docs[person]->words[i]];
		zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
		zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
	}
	for (iter = 1; iter <= setting->MSTEP_MAX_ITER; iter++)
	{
		for (i = 0; i < nvar; i++)
		{

			/*
			** The data is sorted from smallest time to largest
			** Start at the largest time, accumulating the risk set 1 by 1
			*/
			for (int r = ntimes - 1; r >= 0; r--)
			{
				denom[r] = 0.0;
				a[r] = 0.0;
				cdiag[r] = 0.0;
			}
			efron_wt = 0.0;
			newlk = -(log(sqrt(lambda)) * nvar);
			gdiag = 0.0;
			hdiag = 0.0;
			a2 = 0.0;
			cdiag2 = 0.0;

			for (person = nused - 1; person >= 0; person--)
			{
				int time_index_entry = (ss->time0[person]) - (ss->time_start);
				int time_index_exit = (ss->times[person]) - (ss->time_start);

				int word = -1;
				int lastword = -1;
				for (int w = 0; w < c->docs[person]->length; w++)
				{
					if (c->docs[person]->words[w] == i) word = w;
					if (c->docs[person]->words[w] == lastvar) lastword = w;
				}
				if (lastword >= 0)
				{
					zbeta[person] -= eta * c->docs[person]->counts[lastword];
					zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
					zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
				}
				risk = exp(zbeta[person]);
				if (word == -1)
				{
					for (int r = time_index_entry; r <= time_index_exit; r++)
						denom[r] += risk;
					if (ss->labels[person] > 0)
					{
						newlk += zbeta[person];
						efron_wt += risk; /* sum(denom) for tied deaths*/
					}
				}
				else
				{
					//cumumlative sums for all patients
					for (int r = time_index_entry; r <= time_index_exit; r++)
					{
						denom[r] += risk;
						a[r] += risk *c->docs[person]->counts[word];
						cdiag[r] += risk * c->docs[person]->counts[word] * c->docs[person]->counts[word];
					}
					if (ss->labels[person] > 0)
					{
						//cumumlative sums for event patients
						newlk += zbeta[person];
						gdiag -= c->docs[person]->counts[word];
						efron_wt += risk; /* sum(denom) for tied deaths*/
						a2 += risk * c->docs[person]->counts[word];
						cdiag2 += risk * c->docs[person]->counts[word] * c->docs[person]->counts[word];
					}
				}

				
				if (ss->mark[person][setting->CROSSVAL] > 0)
				{  /* once per unique death time */
					for (k = 0; k < ss->mark[person][setting->CROSSVAL]; k++)
					{
						temp = static_cast<double>(k)
							/ static_cast<double>(ss->mark[person][setting->CROSSVAL]);
						d2 = denom[time_index_exit] - (temp * efron_wt); /* sum(denom) adjusted for tied deaths*/
						newlk -= log(d2);
						temp2 = (a[time_index_exit] - (temp * a2)) / d2;
						gdiag += temp2;
						hdiag += ((cdiag[time_index_exit] - (temp * cdiag2)) / d2) -
							(temp2 * temp2);
					}
					efron_wt = 0.0;
					a2 = 0.0;
					cdiag2 = 0.0;
				}
			}   /* end  of accumulation loop  */

			eta = (gdiag + (newbeta[i] / lambda)) / (hdiag + (1.0 / lambda));
			if (fabs(eta) > step[i])
				eta = (eta > 0.0) ? step[i] : -step[i];

			step[i] = ((2.0 * fabs(eta)) > (step[i] / 2.0)) ? 2.0 * fabs(eta) : (step[i] / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			newbeta[i] -= eta;
			lastvar = i;
		}

		for (i = 0; i < nvar; i++)
			newlk -= (newbeta[i] * newbeta[i]) / (2.0 * lambda);
		if (fabs(1.0 - (newlk / loglik)) <= setting->MAX_EPS) break;
		loglik = newlk;
		std::cout << "Cox likelihood : " << newlk << std::endl;
	}   /* return for another iteration */
	for (person = nused - 1; person >= 0; person--)
	{
		int lastword = -1;
		for (int w = 0; w < c->docs[person]->length; w++)
			if (c->docs[person]->words[w] == lastvar) lastword = w;
		if (lastword >= 0)
		{
			zbeta[person] -= eta * c->docs[person]->counts[lastword];
			zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
			zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person]; //update zbeta with last beta change before returning.
		}
	}
	
	/*
	loglik = -(log(sqrt(lambda)) * nvar);
	for (int r = ntimes - 1; r >= 0; r--)
		denom[r] = 0.0;


	efron_wt = 0.0;
	for (person = nused - 1; person >= 0; person--)
	{
		int time_index_entry = (ss->time0[person]) - (ss->time_start);
		int time_index_exit = (ss->times[person]) - (ss->time_start);
		zbeta[person] = 0.0;
		for (i = 0; i<nvar; i++)
			zbeta[person] += newbeta[i] * var[person][i]; // recalculate with final iteration updated betas
		risk = exp(zbeta[person]);
		for (int r = time_index_entry; r <= time_index_exit; r++)
			denom[r] += risk;
		if (ss->labels[person] > 0)
		{
			loglik += zbeta[person];
			efron_wt += risk;
		}
		if (ss->mark[person][setting->CROSSVAL] > 0)
		{
			for (k = 0; k < ss->mark[person][setting->CROSSVAL]; k++)
				loglik -= log(denom[time_index_exit] - ((static_cast<double>(k)
				/ static_cast<double>(ss->mark[person][setting->CROSSVAL]))
				* efron_wt));
			efron_wt = 0.0;
		}
	}
	for (i = 0; i < nvar; i++)
		loglik -= (pow(newbeta[i], 2.0) / (2.0 * lambda));
	*/
	*f = loglik;
	for (i = 0; i<nvar; i++)
		beta[i] = newbeta[i];


	delete[] newbeta;
	delete[] step;
	delete[] denom;
	delete[] a;
	delete[] cdiag;
	return iter;
}



/*
int cox_lasso(double * beta, double * zbeta, double ** var, int nvar, double lambda, suffstats * ss, double * f, const settings* setting)
{
	int i, k, person, iter;
	int nused = ss->num_docs, lastvar = 0;
	double  denom = 0.0, risk = 0.0, temp = 0.0, temp2 = 0.0, loglik = 0.0, newlk = 0.0, d2 = 0.0, efron_wt = 0.0;
	double eta = 0.0, a = 0.0, a2 = 0.0, cdiag = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
	double * newbeta = new double[nvar];
	double * step = new double[nvar];

	for (i = 0; i < nvar; i++)
	{
		newbeta[i] = beta[i];
		step[i] = 1.0;
	}

	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] = 0.0;
		for (i = 0; i < nvar; i++)
			zbeta[person] += var[person][i] * newbeta[i];
		//zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
		//zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
	}
	for (iter = 1; iter <= setting->MSTEP_MAX_ITER; iter++)
	{
		for (i = 0; i < nvar; i++)
		{
			//	if (newbeta[i] != 0.0 || i < setting->ADJN )
			{
	
				** The data is sorted from smallest time to largest
				** Start at the largest time, accumulating the risk set 1 by 1
	
				denom = 0.0;
				efron_wt = 0.0;
				newlk = 0.0;
				gdiag = 0.0;
				hdiag = 0.0;
				a = 0.0;
				a2 = 0.0;
				cdiag = 0.0;
				cdiag2 = 0.0;

				for (person = nused - 1; person >= 0; person--)
				{
					zbeta[person] -= eta * var[person][lastvar];
					//zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
					//zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
					risk = exp(zbeta[person]);

					//cumumlative sums for all patients
					denom += risk;
					a += risk * var[person][i];
					cdiag += risk * var[person][i] * var[person][i];

					if (ss->labels[person] == 1)
					{
						//cumumlative sums for event patients
						newlk += zbeta[person];
						gdiag -= var[person][i];

						efron_wt += risk; // sum(denom) for tied deaths//
						a2 += risk * var[person][i];
						cdiag2 += risk * var[person][i] * var[person][i];
					}
					if (ss->mark[person][setting->CROSSVAL] > 0)
					{  // once per unique death time //
						for (k = 0; k < ss->mark[person][setting->CROSSVAL]; k++)
						{
							temp = static_cast<double>(k)
								/ static_cast<double>(ss->mark[person][setting->CROSSVAL]);
							d2 = denom - (temp * efron_wt); // sum(denom) adjusted for tied deaths//
							newlk -= log(d2);
							temp2 = (a - (temp * a2)) / d2;
							gdiag += temp2;
							hdiag += ((cdiag - (temp * cdiag2)) / d2) -
								(temp2 * temp2);
						}
						efron_wt = 0.0;
						a2 = 0.0;
						cdiag2 = 0.0;
					}
				}   // end  of accumulation loop  //
				eta = 0.0;
				if (newbeta[i] > 0)
					eta = (gdiag + sqrt(lambda)) / (hdiag);
				else if (newbeta[i] < 0)
					eta = (gdiag - sqrt(lambda)) / (hdiag);
				if (fabs(eta) > step[i])
					eta = (eta > 0.0) ? step[i] : -step[i];
				if ((newbeta[i] - eta > 0.0 && newbeta[i] > 0.0) || (newbeta[i] - eta < 0.0 && newbeta[i] < 0.0))
					newbeta[i] -= eta;
				else //update must cross the null
				{
					eta = (gdiag + sqrt(lambda)) / (hdiag);
					if ((-eta) > 0)
						newbeta[i] -= (fabs(eta) > step[i]) ? step[i] : eta ;
					else
					{
						eta = (gdiag - sqrt(lambda)) / (hdiag);
						if ((-eta) < 0)
							newbeta[i] -= (fabs(eta) >  step[i]) ? -step[i] : eta;
						else
							newbeta[i] = 0.0;
					}
				}
				step[i] = ((2.0 * fabs(eta)) > (step[i] / 2.0)) ? 2.0 * fabs(eta) : (step[i] / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
				lastvar = i;
			}
		}
		for (i = 0; i < nvar; i++)
			newlk -= log(2.0) - log(sqrt(lambda)) + (sqrt(lambda) * fabs(newbeta[i]));
		if (fabs(1 - (loglik / newlk)) <= setting->MAX_EPS) break;
		loglik = newlk;
	}   // return for another iteration //

	loglik = 0.0;
	denom = 0;
	efron_wt = 0;
	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] = 0.0;
		for (i = 0; i<nvar; i++)
			zbeta[person] += newbeta[i] * var[person][i]; // recalculate with final iteration updated betas
		risk = exp(zbeta[person]);
		denom += risk;
		if (ss->labels[person] == 1)
		{
			loglik += zbeta[person];
			efron_wt += risk;
		}
		if (ss->mark[person][setting->CROSSVAL] > 0)
		{
			for (k = 0; k < ss->mark[person][setting->CROSSVAL]; k++)
				loglik -= log(denom - ((static_cast<double>(k)
				/ static_cast<double>(ss->mark[person][setting->CROSSVAL]))
				* efron_wt));
			efron_wt = 0.0;
		}
	}
	for (i = 0; i < nvar; i++)
		loglik -= log(2.0) - log(sqrt(lambda)) + (sqrt(lambda) * fabs(newbeta[i]));

	*f = loglik;
	for (i = 0; i<nvar; i++)
		beta[i] = newbeta[i];
	delete[] newbeta;
	delete[] step;
	return iter;
}
*/
/*
double cox_lasso_cross_val(
	int group,
	double * newbeta,
	double ** var,
	double * zbeta,
	double * step,
	int nvar,
	double lambda,
	const suffstats * ss,
	const settings* setting)
{
	int i, k, person, iter;
	int nused = ss->num_docs, lastvar = 0;
	double denom = 0.0, risk = 0.0, temp = 0.0, temp2 = 0.0, loglik = 0.0, newlk = 0.0;
	double d2 = 0.0, efron_wt = 0.0;
	double eta = 0.0, a = 0.0, a2 = 0.0, cdiag = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
	for (k = 0; k < nvar; k++)
		step[k] = 1.0;
	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] = 0;
		for (i = 0; i<nvar; i++)
			zbeta[person] += newbeta[i] * var[person][i];
		//	zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
		//	zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
	}

	for (iter = 1; iter <= setting->MSTEP_MAX_ITER; iter++)
	{
		for (i = 0; i<nvar; i++)
		{
			//if (newbeta[i] != 0.0 || i < setting->ADJN)
			{
				
				** The data is sorted from smallest time to largest
				** Start at the largest time, accumulating the risk set 1 by 1
				

				newlk = 0.0; // New likelihood for current iteration

				denom = 0.0;  //sum(exp(XB)) for remaining patients in risk set
				a = 0.0; // sum(x..j * exp(XB))
				cdiag = 0.0; // sum((x.j)^2 * exp(XB))

				efron_wt = 0.0; // sum(denom) for tied deaths
				a2 = 0.0; // sum(x..j * exp(XB)) for tied deaths
				cdiag2 = 0.0; //	sum((x.j)^2 * exp(XB)) for tied deaths

				gdiag = 0.0;
				hdiag = 0.0; // Hessian matrix diagonal for ij weighted for tied deaths

				for (person = nused - 1; person >= 0; person--)
				{
					if (ss->group[person] != group) //(beta_j-1)
					{
						zbeta[person] -= eta * var[person][lastvar];
						//zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
						//zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
						risk = exp(zbeta[person]);

						//cumumlative sums for all patients
						denom += risk;
						if (isn(denom))
							return numeric_limits<double>::lowest();
						a += risk * var[person][i];
						cdiag += risk * var[person][i] * var[person][i];

						if (ss->labels[person] == 1)
						{
							//cumumlative sums for all patients
							newlk += zbeta[person];

							efron_wt += risk;
							gdiag -= var[person][i];
							a2 += risk * var[person][i];
							cdiag2 += risk * var[person][i] * var[person][i];
						}
					}
					int numdeaths = (ss->mark[person][setting->CROSSVAL] - ss->mark[person][group]);
					if (numdeaths > 0)
					{  // once per unique death time //
						for (k = 0; k < numdeaths; k++)
						{
							temp = static_cast<double>(k)
								/ static_cast<double>(numdeaths);
							d2 = denom - (temp * efron_wt);
							newlk -= log(d2);
							temp2 = (a - (temp * a2)) / d2;
							gdiag += temp2;
							hdiag += ((cdiag - (temp * cdiag2)) / d2) -
								(temp2 * temp2);
						}
						efron_wt = 0.0;
						a2 = 0.0;
						cdiag2 = 0.0;
					}
					if (isn(hdiag))
						return numeric_limits<double>::lowest();

				} //end  of accumulation loop  //

				if (newbeta[i] > 0)
					eta = (gdiag + sqrt(lambda)) / (hdiag);
				else if (newbeta[i] < 0)
					eta = (gdiag - sqrt(lambda)) / (hdiag);
				if (fabs(eta) > step[i])
					eta = (eta > 0.0) ? step[i] : -step[i];
				if ((newbeta[i] - eta > 0.0 && newbeta[i] > 0.0) || (newbeta[i] - eta < 0.0 && newbeta[i] < 0.0))
					newbeta[i] -= eta;
				else //update must cross the null
				{
					eta = (gdiag + sqrt(lambda)) / (hdiag);
					if ((-eta) > 0)
						newbeta[i] -= (fabs(eta) > step[i]) ? step[i] : eta;
					else
					{
						eta = (gdiag - sqrt(lambda)) / (hdiag);
						if ((-eta) < 0)
							newbeta[i] -= (fabs(eta) >  step[i]) ? -step[i] : eta;
						else
							newbeta[i] = 0.0;
					}
				}
				step[i] = ((2.0 * fabs(eta)) > (step[i] / 2.0)) ? 2.0 * fabs(eta) : (step[i] / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
				newbeta[i] -= eta;
				lastvar = i;
				if (isn(newbeta[i]))
					return numeric_limits<double>::lowest();

				//am I done?
				**   update the betas and test for convergence
				
			}
		}
		for (i = 0; i < nvar; i++)
			newlk -= log(2.0) - log(sqrt(lambda)) + (sqrt(lambda) * fabs(newbeta[i])); //(l_j-1)
		if (isn(newlk))
			return numeric_limits<double>::lowest();

		if (fabs(1.0 - (loglik / newlk)) <= setting->MAX_EPS) break;

		loglik = newlk;
	}   // return for another iteration //
	denom = 0.0;
	d2 = 0.0;
	efron_wt = 0.0;
	a2 = 0.0;
	loglik = 0.0; //use for l(b_j-1)
	newlk = 0.0;//use for l_j-1(b_j-1)
	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] = 0.0;
		for (i = 0; i < nvar; i++)
			zbeta[person] += newbeta[i] * var[person][i]; // recalculate with final iteration updated betas
		risk = exp(zbeta[person]);
		denom += risk;
		if (ss->group[person] != group)
			d2 += risk;
		if (ss->labels[person] == 1)
		{
			efron_wt += risk;
			loglik += zbeta[person];
			if (ss->group[person] != group)
			{
				a2 += risk;
				newlk += zbeta[person];
			}
		}
		if (static_cast<int>(ss->mark[person][setting->CROSSVAL]) > 0)
		{
			for (k = 0; k < ss->mark[person][setting->CROSSVAL]; k++)
				loglik -= log(denom - ((static_cast<double>(k)
				/ static_cast<double>(ss->mark[person][setting->CROSSVAL]))
				* efron_wt));
			efron_wt = 0.0;
			int num_deaths = ss->mark[person][setting->CROSSVAL] - ss->mark[person][group];
			for (k = 0; k < num_deaths; k++)
				newlk -= log(d2 - (static_cast<double>(k)
				/ static_cast<double>(num_deaths)
				* a2));
			a2 = 0.0;
		}
	}
	for (i = 0; i < nvar; i++)
	{
		loglik -= log(2.0) - log(sqrt(lambda)) + (sqrt(lambda) * fabs(newbeta[i]));
		newlk -= log(2.0) - log(sqrt(lambda)) + (sqrt(lambda) * fabs(newbeta[i]));
	}
	if (isn(loglik))
		return numeric_limits<double>::lowest();
	if (isn(newlk))
		return numeric_limits<double>::lowest();
	return loglik - newlk;
}
*/
/*
int cox_net(double alphanet, double * beta, double * zbeta, double ** var, int nvar, double lambda, suffstats * ss, double * f, const settings* setting)
{
	int i, k, person, iter;
	int nused = ss->num_docs, lastvar = 0;
	double  denom = 0.0, risk = 0.0, temp = 0.0, temp2 = 0.0, loglik = 0.0, newlk = 0.0, d2 = 0.0, efron_wt = 0.0;
	double eta = 0.0, a = 0.0, a2 = 0.0, cdiag = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
	double * newbeta = new double[nvar];
	double * step = new double[nvar];

	for (i = 0; i < nvar; i++)
	{
		newbeta[i] = beta[i];
		step[i] = 1.0;
	}

	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] = 0.0;
		for (i = 0; i < nvar; i++)
			zbeta[person] += var[person][i] * newbeta[i];
		//zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
		//zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
	}
	for (iter = 1; iter <= setting->MSTEP_MAX_ITER; iter++)
	{
		for (i = 0; i < nvar; i++)
		{
			//	if (newbeta[i] != 0.0 || i < setting->ADJN )
			{
				
				** The data is sorted from smallest time to largest
				** Start at the largest time, accumulating the risk set 1 by 1
				
				denom = 0.0;
				efron_wt = 0.0;
				newlk = 0.0;
				gdiag = 0.0;
				hdiag = 0.0;
				a = 0.0;
				a2 = 0.0;
				cdiag = 0.0;
				cdiag2 = 0.0;

				for (person = nused - 1; person >= 0; person--)
				{
					zbeta[person] -= eta * var[person][lastvar];
					//zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
					//zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
					risk = exp(zbeta[person]);

					//cumumlative sums for all patients
					denom += risk;
					a += risk * var[person][i];
					cdiag += risk * var[person][i] * var[person][i];

					if (ss->labels[person] == 1)
					{
						//cumumlative sums for event patients
						newlk += zbeta[person];
						gdiag -= var[person][i];

						efron_wt += risk; // sum(denom) for tied deaths//
						a2 += risk * var[person][i];
						cdiag2 += risk * var[person][i] * var[person][i];
					}
					if (ss->mark[person][setting->CROSSVAL] > 0)
					{  // once per unique death time //
						for (k = 0; k < ss->mark[person][setting->CROSSVAL]; k++)
						{
							temp = static_cast<double>(k)
								/ static_cast<double>(ss->mark[person][setting->CROSSVAL]);
							d2 = denom - (temp * efron_wt); // sum(denom) adjusted for tied deaths//
							newlk -= log(d2);
							temp2 = (a - (temp * a2)) / d2;
							gdiag += temp2;
							hdiag += ((cdiag - (temp * cdiag2)) / d2) -
								(temp2 * temp2);
						}
						efron_wt = 0.0;
						a2 = 0.0;
						cdiag2 = 0.0;
					}
				}   // end  of accumulation loop  //

				eta = 0.0;
				if (newbeta[i] > 0)
					eta = (
					gdiag
					+ (alphanet * sqrt(2.0 / lambda))
					+ ((1 - alphanet) * (newbeta[i] / lambda))
					) / (
					hdiag
					+ ((1 - alphanet) * (1.0 / lambda)));
				else if (newbeta[i] < 0)
					eta = (
					gdiag
					- (alphanet * sqrt(2.0 / lambda))
					+ ((1 - alphanet) * (newbeta[i] / lambda))
					) / (
					hdiag
					+ ((1 - alphanet) * (1.0 / lambda)));
				if (fabs(eta) > step[i])
					eta = (eta > 0.0) ? step[i] : -step[i];
				if ((newbeta[i] - eta > 0.0 && newbeta[i] > 0.0) || (newbeta[i] - eta < 0.0 && newbeta[i] < 0.0))
					newbeta[i] -= eta;
				else //update must cross the null
				{
					eta = (
						gdiag
						+ (alphanet * sqrt(2.0 / lambda))
						+ ((1 - alphanet) * (newbeta[i] / lambda))
						) / (
						hdiag
						+ ((1 - alphanet) * (1.0 / lambda)));
					if ((-eta) > 0)
						newbeta[i] -= (fabs(eta) > step[i]) ? -step[i] : eta;
					else
					{
						eta = (
							gdiag
							- (alphanet * sqrt(2.0 / lambda))
							+ ((1 - alphanet) * (newbeta[i] / lambda))
							) / (
							hdiag
							+ ((1 - alphanet) * (1.0 / lambda)));
						if ((-eta) < 0)
							newbeta[i] -= (fabs(eta) >  step[i]) ? step[i] : eta;
						else
							newbeta[i] = 0.0;
					}
				}
				step[i] = ((2.0 * fabs(eta)) > (step[i] / 2.0)) ? 2.0 * fabs(eta) : (step[i] / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
				newbeta[i] -= eta;
				
				lastvar = i;
			}
		}
		for (i = 0; i < nvar; i++)
			newlk -= (alphanet * (log(2.0) - log(sqrt(2.0 / lambda)) + (sqrt(2.0 / lambda) * fabs(newbeta[i]))))
			+ ((1 - alphanet)  * (newbeta[i] * newbeta[i]) / (2.0 * lambda));
		if (fabs(1 - (loglik / newlk)) <= setting->MAX_EPS) break;
		loglik = newlk;
	}   // return for another iteration //

	loglik = 0.0;
	denom = 0;
	efron_wt = 0;
	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] = 0.0;
		for (i = 0; i<nvar; i++)
			zbeta[person] += newbeta[i] * var[person][i]; // recalculate with final iteration updated betas
		risk = exp(zbeta[person]);
		denom += risk;
		if (ss->labels[person] == 1)
		{
			loglik += zbeta[person];
			efron_wt += risk;
		}
		if (ss->mark[person][setting->CROSSVAL] > 0)
		{
			for (k = 0; k < ss->mark[person][setting->CROSSVAL]; k++)
				loglik -= log(denom - ((static_cast<double>(k)
				/ static_cast<double>(ss->mark[person][setting->CROSSVAL]))
				* efron_wt));
			efron_wt = 0.0;
		}
	}
	for (i = 0; i < nvar; i++)
		loglik -= (alphanet * (log(2.0) - log(sqrt(2.0 / lambda)) + (sqrt(2.0 / lambda) * fabs(newbeta[i]))))
		+ ((1 - alphanet)  * (newbeta[i] * newbeta[i]) / (2.0 * lambda));

	*f = loglik;
	for (i = 0; i<nvar; i++)
		beta[i] = newbeta[i];
	delete[] newbeta;
	delete[] step;
	return iter;
}


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
	const settings* setting)
{
	int i, k, person, iter;
	int nused = ss->num_docs, lastvar = 0;
	double denom = 0.0, risk = 0.0, temp = 0.0, temp2 = 0.0, loglik = 0.0, newlk = 0.0;
	double d2 = 0.0, efron_wt = 0.0;
	double eta = 0.0, a = 0.0, a2 = 0.0, cdiag = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
	for (k = 0; k < nvar; k++)
		step[k] = 1.0;
	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] = 0;
		for (i = 0; i<nvar; i++)
			zbeta[person] += newbeta[i] * var[person][i];
		//	zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
		//	zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
	}

	for (iter = 1; iter <= setting->MSTEP_MAX_ITER; iter++)
	{
		for (i = 0; i<nvar; i++)
		{
			//if (newbeta[i] != 0.0 || i < setting->ADJN)
			{
				
				** The data is sorted from smallest time to largest
				**Start at the largest time, accumulating the risk set 1 by 1
				

				newlk = 0.0; // New likelihood for current iteration

				denom = 0.0;  //sum(exp(XB)) for remaining patients in risk set
				a = 0.0; // sum(x..j * exp(XB))
				cdiag = 0.0; // sum((x.j)^2 * exp(XB))

				efron_wt = 0.0; // sum(denom) for tied deaths
				a2 = 0.0; // sum(x..j * exp(XB)) for tied deaths
				cdiag2 = 0.0; //	sum((x.j)^2 * exp(XB)) for tied deaths

				gdiag = 0.0;
				hdiag = 0.0; // Hessian matrix diagonal for ij weighted for tied deaths

				for (person = nused - 1; person >= 0; person--)
				{
					if (ss->group[person] != group) //(beta_j-1)
					{
						zbeta[person] -= eta * var[person][lastvar];
						//zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
						//zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
						risk = exp(zbeta[person]);

						//cumumlative sums for all patients
						denom += risk;
						if (isn(denom))
							return numeric_limits<double>::lowest();
						a += risk * var[person][i];
						cdiag += risk * var[person][i] * var[person][i];

						if (ss->labels[person] == 1)
						{
							//cumumlative sums for all patients
							newlk += zbeta[person];

							efron_wt += risk;
							gdiag -= var[person][i];
							a2 += risk * var[person][i];
							cdiag2 += risk * var[person][i] * var[person][i];
						}
					}
					int numdeaths = (ss->mark[person][setting->CROSSVAL] - ss->mark[person][group]);
					if (numdeaths > 0)
					{  // once per unique death time //
						for (k = 0; k < numdeaths; k++)
						{
							temp = static_cast<double>(k)
								/ static_cast<double>(numdeaths);
							d2 = denom - (temp * efron_wt);
							newlk -= log(d2);
							temp2 = (a - (temp * a2)) / d2;
							gdiag += temp2;
							hdiag += ((cdiag - (temp * cdiag2)) / d2) -
								(temp2 * temp2);
						}
						efron_wt = 0.0;
						a2 = 0.0;
						cdiag2 = 0.0;
					}
					if (isn(hdiag))
						return numeric_limits<double>::lowest();

				} // end  of accumulation loop //
				
				eta = 0.0;
				if (newbeta[i] > 0)
					eta = (
					gdiag 
					+ (alphanet * sqrt(2.0 / lambda)) 
					+ ((1 - alphanet) * (newbeta[i] / lambda))
					) / (
					hdiag 
					+ ((1 - alphanet) * (1.0 / lambda)));
				else if (newbeta[i] < 0)
					eta = (
					gdiag
					- (alphanet * sqrt(2.0 / lambda))
					+ ((1 - alphanet) * (newbeta[i] / lambda))
					) / (
					hdiag
					+ ((1 - alphanet) * (1.0 / lambda)));
				if (fabs(eta) > step[i])
					eta = (eta > 0.0) ? step[i] : -step[i];
				if ((newbeta[i] - eta > 0.0 && newbeta[i] > 0.0) || (newbeta[i] - eta < 0.0 && newbeta[i] < 0.0) || alphanet == 0.0)
					newbeta[i] -= eta;
				else //update must cross the null
				{
					eta = (
						gdiag
						+ (alphanet * sqrt(2.0 / lambda))
						+ ((1 - alphanet) * (newbeta[i] / lambda))
						) / (
						hdiag
						+ ((1 - alphanet) * (1.0 / lambda)));
					if ((-eta) > 0)
						newbeta[i] -= (fabs(eta) > step[i]) ? -step[i] : eta;
					else
					{
						eta = (
							gdiag
							- (alphanet * sqrt(2.0 / lambda))
							+ ((1 - alphanet) * (newbeta[i] / lambda))
							) / (
							hdiag
							+ ((1 - alphanet) * (1.0 / lambda)));
						if ((-eta) < 0)
							newbeta[i] -= (fabs(eta) >  step[i]) ? step[i] : eta;
						else
							newbeta[i] = 0.0;
					}
				}
				step[i] = ((2.0 * fabs(eta)) > (step[i] / 2.0)) ? 2.0 * fabs(eta) : (step[i] / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
				lastvar = i;
				if (isn(newbeta[i]))
					return numeric_limits<double>::lowest();

				// am I done?
				**   update the betas and test for convergence
				
			}
		}
		for (i = 0; i < nvar; i++)
			newlk -= (alphanet * (log(2.0) - log(sqrt(2.0 / lambda)) + (sqrt(2.0 / lambda) * fabs(newbeta[i]))))
			+ ((1 - alphanet)  * (newbeta[i] * newbeta[i]) / (2.0 * lambda));
		if (isn(newlk))
			return numeric_limits<double>::lowest();

		if (fabs(1.0 - (loglik / newlk)) <= setting->MAX_EPS) break;

		loglik = newlk;
	}   // return for another iteration //
	denom = 0.0;
	d2 = 0.0;
	efron_wt = 0.0;
	a2 = 0.0;
	loglik = 0.0; //use for l(b_j-1)
	newlk = 0.0;//use for l_j-1(b_j-1)
	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] = 0.0;
		for (i = 0; i < nvar; i++)
			zbeta[person] += newbeta[i] * var[person][i]; // recalculate with final iteration updated betas
		risk = exp(zbeta[person]);
		denom += risk;
		if (ss->group[person] != group)
			d2 += risk;
		if (ss->labels[person] == 1)
		{
			efron_wt += risk;
			loglik += zbeta[person];
			if (ss->group[person] != group)
			{
				a2 += risk;
				newlk += zbeta[person];
			}
		}
		if (static_cast<int>(ss->mark[person][setting->CROSSVAL]) > 0)
		{
			for (k = 0; k < ss->mark[person][setting->CROSSVAL]; k++)
				loglik -= log(denom - ((static_cast<double>(k)
				/ static_cast<double>(ss->mark[person][setting->CROSSVAL]))
				* efron_wt));
			efron_wt = 0.0;
			int num_deaths = ss->mark[person][setting->CROSSVAL] - ss->mark[person][group];
			for (k = 0; k < num_deaths; k++)
				newlk -= log(d2 - (static_cast<double>(k)
				/ static_cast<double>(num_deaths)
				* a2));
			a2 = 0.0;
		}
	}
	for (i = 0; i < nvar; i++)
	{
		loglik -= (alphanet * (log(2.0) - log(sqrt(2.0 / lambda)) + (sqrt(2.0 / lambda) * fabs(newbeta[i]))))
			+ ((1 - alphanet)  * (newbeta[i] * newbeta[i]) / (2.0 * lambda));
		newlk -= (alphanet * (log(2.0) - log(sqrt(2.0 / lambda)) + (sqrt(2.0 / lambda) * fabs(newbeta[i])))) 
			+ ((1 - alphanet)  * (newbeta[i] * newbeta[i]) / (2.0 * lambda));
	}
	if (isn(loglik))
		return numeric_limits<double>::lowest();
	if (isn(newlk))
		return numeric_limits<double>::lowest();
	return loglik - newlk;
}
*/

int coxfit(double * beta, int nvar,
	double ** var,
	double * zbeta, 
	const suffstats * ss, double * f, const settings* setting)
{
	int i, j, k, person;
	int     iter;
	int     nused;
	nused = ss->num_docs;


	double  denom = 0.0,  risk = 0.0;
	double  temp = 0, temp2 = 0;
	double  ndead = 0;
	double  newlk = 0;
	double  d2, efron_wt;
	int     halving = 0;    /*are we doing step halving at the moment? */
	double  method = 1;
	//double	sctest = 1;


	double * mark = new double[nused];
	double * wtave = new double[nused];
	double * means = new double[nvar];
	double	loglik[2] = { 0 };
	double * u = new double[nvar];
	double * a = new double[nvar];
	double * a2 = new double[nvar];
	double * newbeta = new double[nvar];
	double ** cmat = new double *[nvar];
	double ** cmat2 = new double *[nvar];
	double ** imat = new double *[nvar];
	double ** covar = new double *[nused];
	for (i = 0; i < nvar; i++)
	{
		cmat[i] = new double[nvar];
		cmat2[i] = new double[nvar];
		imat[i] = new double[nvar];
		for (j = 0; j < nvar; j++)
		{
			cmat[i][j] = 0;
			cmat2[i][j] = 0;
			imat[i][j] = 0;
		}
	}

	for (person = 0; person<nused; person++)
		covar[person] = new double[nvar];
	temp = 0;
	j = 0;
	for (i = nused - 1; i>0; i--)
	{
		if (((ss->times[i]) == (ss->times[i - 1])) && (i != 1))
		{
			j += ss->labels[i];
			temp += ss->labels[i];
			mark[i] = 0;
		}
		else
		{
			mark[i] = j + ss->labels[i];
			if (mark[i] >0) wtave[i] = (temp + ss->labels[i]) / mark[i];
			temp = 0; j = 0;
		}
	}
	mark[0] = j + ss->labels[0];
	if (mark[0]>0)
		wtave[0] = (temp + ss->labels[0]) / mark[0];

	/*
	** Subtract the mean from each covar, as this makes the regression
	**  much more stable
	*/

	for (i = 0; i<nvar; i++)
	{
		temp = 0;
		for (person = 0; person<nused; person++)
			temp += var[person][i];
		temp /= nused;
		means[i] = temp;

		for (person = 0; person<nused; person++)
			covar[person][i] = var[person][i] - temp;
	};

	/*
	** do the initial iteration step
	*/
	loglik[1] = 0;
	for (i = 0; i<nvar; i++)
	{
		u[i] = 0;
		for (j = 0; j<nvar; j++)
			imat[i][j] = 0;
	}

	efron_wt = 0;
	for (person = nused - 1; person >= 0; person--)
	{
		if (person == nused - 1)
		{
			denom = 0;
			for (i = 0; i<nvar; i++)
			{
				a[i] = 0;
				a2[i] = 0;
				for (j = 0; j<nvar; j++)
				{
					cmat[i][j] = 0;
					cmat2[i][j] = 0;
				}
			}
		}

		zbeta[person] = 0;    /* form the term beta*z   (vector mult) */
		for (i = 0; i<nvar; i++)
		{
			zbeta[person] += beta[i] * covar[person][i];
		}
		//zbeta = coxsafe(zbeta);
		zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
		zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
		risk = exp(zbeta[person]);

		denom += risk;
		efron_wt += ss->labels[person] * risk;  /*sum(denom) for tied deaths*/

		for (i = 0; i<nvar; i++)
		{
			a[i] += risk*covar[person][i];
			for (j = 0; j <= i; j++)
			{
				cmat[i][j] += risk*covar[person][i] * covar[person][j];
			}
		}

		if (ss->labels[person] == 1)
		{
			loglik[1] += zbeta[person];
			for (i = 0; i<nvar; i++)
			{
				u[i] += covar[person][i];
				a2[i] += risk*covar[person][i];
				for (j = 0; j <= i; j++)
				{
					cmat2[i][j] += risk*covar[person][i] * covar[person][j];
				}
			}
		}
		if (mark[person] >0)
		{  /* once per unique death time */
			/*
			** Trick: when 'method==0' then temp=0, giving Breslow's method
			*/
			ndead = mark[person];
			for (k = 0; k<ndead; k++)
			{
				temp = (double)k * method / ndead;
				d2 = denom - temp*efron_wt;
				loglik[1] -= wtave[person] * log(d2);
				for (i = 0; i<nvar; i++)
				{
					temp2 = (a[i] - temp*a2[i]) / d2;
					u[i] -= wtave[person] * temp2;
					for (j = 0; j <= i; j++)
					{
						imat[j][i] += wtave[person] *
							((cmat[i][j] - temp*cmat2[i][j]) / d2 -
							temp2 * (a[j] - temp * a2[j]) / d2);
					}
				}
			}
			efron_wt = 0;
			for (i = 0; i<nvar; i++)
			{
				a2[i] = 0;
				for (j = 0; j<nvar; j++)
				{
					cmat2[i][j] = 0;
				}
			}
		}
	}   /* end  of accumulation loop */

	loglik[0] = loglik[1];   /* save the loglik for iteration zero  */

	/* am I done?
	**   update the betas and test for convergence
	*/
	for (i = 0; i<nvar; i++) /*use 'a' as a temp to save u0, for the score test*/
	{
		a[i] = u[i];
	}

	cholesky2(imat, nvar, 10e-6);
	chsolve2(imat, nvar, a);        /* a replaced by  a *inverse(i) */

	//sctest=0;
	//for (i=0; i<nvar; i++)
	//{
	//	sctest +=  u[i]*a[i];
	//}

	/*
	**  Never, never complain about convergence on the first step.  That way,
	**  if someone HAS to they can force one iter at a time.
	*/
	for (i = 0; i<nvar; i++)
	{
		newbeta[i] = beta[i] + a[i];
	}
	if (setting->MSTEP_MAX_ITER == 0)
	{
		//chinv2(imat,nvar);
		//for (i=1; i<nvar; i++)
		//{
		//	for (j=0; j<i; j++)  
		//	{
		//		imat[i][j] = imat[j][i];
		//	}
		//}
		* f = loglik[1];

		delete[] mark;
		delete[] wtave;
		delete[] means;
		delete[] u;
		delete[] a;
		delete[] a2;
		delete[] newbeta;
		for (i = 0; i<nused; i++)
			covar[i] = NULL;
		delete[] covar;
		for (i = 0; i<nvar; i++)
		{
			cmat[i] = NULL;
			cmat2[i] = NULL;
			imat[i] = NULL;
		}
		delete[] cmat;
		delete[] cmat2;
		delete[] imat;

		return -1;   /* and we leave the old beta in peace */
	}

	/*
	** here is the main loop
	*/
	//halving =0 ;             /* =1 when in the midst of "step halving" */
	for (iter = 1; iter <= setting->MSTEP_MAX_ITER; iter++)
	{
		newlk = 0;
		for (i = 0; i<nvar; i++)
		{
			u[i] = 0;
			for (j = 0; j<nvar; j++)
				imat[i][j] = 0;
		}



		/*
		** The data is sorted from smallest time to largest
		** Start at the largest time, accumulating the risk set 1 by 1
		*/
		for (person = nused - 1; person >= 0; person--)
		{
			if (person == nused - 1)
			{ /* rezero temps for each strata */
				efron_wt = 0;
				denom = 0;
				for (i = 0; i<nvar; i++)
				{
					a[i] = 0;
					a2[i] = 0;
					for (j = 0; j<nvar; j++)
					{
						cmat[i][j] = 0;
						cmat2[i][j] = 0;
					}
				}
			}

			zbeta[person] = 0;
			for (i = 0; i<nvar; i++)
			{
				zbeta[person] += newbeta[i] * covar[person][i];
			}

			zbeta[person] = zbeta[person] > 22 ? 22 : zbeta[person];
			zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
			risk = exp(zbeta[person]);
			denom += risk;
			efron_wt += ss->labels[person] * risk;  /* sum(denom) for tied deaths*/

			for (i = 0; i<nvar; i++)
			{
				a[i] += risk*covar[person][i];
				for (j = 0; j <= i; j++)
				{
					cmat[i][j] += risk*covar[person][i] * covar[person][j];
				}
			}

			if (ss->labels[person] == 1)
			{
				newlk += zbeta[person];
				for (i = 0; i<nvar; i++)
				{
					u[i] += covar[person][i];
					a2[i] += risk*covar[person][i];
					for (j = 0; j <= i; j++)
					{
						cmat2[i][j] += risk*covar[person][i] * covar[person][j];
					}
				}
			}

			if (mark[person] >0)
			{  /* once per unique death time */
				for (k = 0; k<mark[person]; k++)
				{
					temp = (double)k* method / mark[person];
					d2 = denom - temp*efron_wt;
					newlk -= wtave[person] * log(d2);
					for (i = 0; i<nvar; i++)
					{
						temp2 = (a[i] - temp*a2[i]) / d2;
						u[i] -= wtave[person] * temp2;
						for (j = 0; j <= i; j++)
						{
							imat[j][i] += wtave[person] * (
								(cmat[i][j] - temp * cmat2[i][j]) / d2 -
								temp2 * (a[j] - temp * a2[j]) / d2);
						}
					}
				}
				efron_wt = 0;
				for (i = 0; i<nvar; i++)
				{
					a2[i] = 0;
					for (j = 0; j<nvar; j++)
					{
						cmat2[i][j] = 0;
					}
				}
			}
		}   /* end  of accumulation loop  */

		/* am I done?
		**   update the betas and test for convergence
		*/
		cholesky2(imat, nvar, 10e-6);

		if (fabs(1 - (loglik[1] / newlk)) <= setting->MAX_EPS && halving == 0)
		{ /* all done */
			loglik[1] = newlk;
			for (i = 0; i<nvar; i++)
			{
				beta[i] = newbeta[i];
			}
			//iter=MSTEP_MAX_ITER;
			*f = loglik[1];

			delete[] mark;
			delete[] wtave;
			delete[] means;
			delete[] u;
			delete[] a;
			delete[] a2;
			delete[] newbeta;
			for (i = 0; i<nused; i++)
				covar[i] = NULL;
			delete[] covar;
			for (i = 0; i<nvar; i++)
			{
				cmat[i] = NULL;
				cmat2[i] = NULL;
				imat[i] = NULL;
			}
			delete[] cmat;
			delete[] cmat2;
			delete[] imat;
			return iter;
		}

		if (iter == setting->MSTEP_MAX_ITER)
			break;  /*skip the step halving calc*/

		if (newlk < loglik[1])
		{    /*it is not converging ! */
			halving = 1;
			for (i = 0; i<nvar; i++)
				newbeta[i] = (newbeta[i] + beta[i]) / 2; /*half of old increment */
		}
		else
		{
			halving = 0;
			loglik[1] = newlk;
			chsolve2(imat, nvar, u);

			j = 0;
			for (i = 0; i<nvar; i++)
			{
				beta[i] = newbeta[i];
				newbeta[i] = newbeta[i] + u[i];
			}
		}
		if (iter %  setting->LAG == 0)
			printf("Maximisation iteration %d ... likelihood %f\n", iter, loglik[1]);
	}   /* return for another iteration */

	loglik[1] = newlk;
	for (i = 0; i<nvar; i++)
		beta[i] = newbeta[i];
	
	*f = loglik[1];

	delete[] mark;
	delete[] wtave;
	delete[] means;
	delete[] u;
	delete[] a;
	delete[] a2;
	delete[] newbeta;
	for (i = 0; i < nused; i++)
	{
		delete[] covar[i];
	}
	delete[] covar;
	covar = NULL;
	for (i = 0; i<nvar; i++)
	{
		delete[] cmat[i];
		cmat[i] = NULL;
		delete[] cmat2[i];
		cmat2[i] = NULL;
		delete imat[i];
		imat[i] = NULL;
	}
	delete[] cmat;
	cmat = NULL;
	delete[] cmat2;
	cmat2 = NULL;
	delete[] imat;
	imat = NULL;
	return iter;
}

//
//double beta_f(const gsl_vector * x, void * opt_param)
//{
//	int d, j, k;
//	double f{ 0.0 };
//	for (d = 0; d < opt_param.ss->num_docs; d++)
//	{
//		int time_index_entry = (opt_param.ss->time0[d]) - (opt_param.ss->time_start);
//		int time_index_exit = (opt_param.ss->times[d]) - (opt_param.ss->time_start);
//		if (ss->labels > 0)
//			f += log(basehaz[time_index_exit]);
//		double temp = 0;
//		for (k = 0; k < opt_param.ss->num_topics; k++)
//		{
//			f += x[k] * ss->z_bar[d][k];
//			temp *= (exp(x[k] / c->docs[d]->total)) * opt_param.ss->z_count[d][k];
//		}
//		f -= cumbasehaz[time_index_exit] * temp;
//	}
//	return f;
//}
//
//void beta_df(double * beta, double * cumbasehaz,  int nvar, corpus * c, const suffstats * ss)
//{
//	int d, j, k;
//	double df{ 0.0 };
//	for (d = 0; d < ss->num_docs; d++)
//	{
//		int time_index_exit = (ss->times[d]) - (ss->time_start);
//		double temp = 0;
//		for (k = 0; k < nvar; k++)
//		{
//			df += beta[k] * ss->z_bar[d][k];
//			temp *= (exp(beta[k] / c->docs[d]->total)) * ss->z_count[d][k];
//		}
//		df -= cumbasehaz[time_index_exit] * temp;
//		df /= c->docs[d]->total;
//
//	}
//
//}
//void  beta_ddf(double * beta, double * cumbasehaz, int nvar, corpus * c, const suffstats * ss)
//{
//	int d, j, k;
//	double ddf{ 0.0 };
//	for (d = 0; d < ss->num_docs; d++)
//	{
//		int time_index_exit = (ss->times[d]) - (ss->time_start);
//		double temp = 0;
//		for (k = 0; k < nvar; k++)
//		{
//			ddf += beta[k] * ss->z_bar[d][k];
//			temp *= (exp(beta[k] / c->docs[d]->total)) * ss->z_count[d][k];
//		}
//		ddf -= cumbasehaz[time_index_exit] * temp;
//		ddf /= (c->docs[d]->total) * (c->docs[d]->total);
//
//	}
//	return ddf;
//}