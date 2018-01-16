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
#include "opt.h"
#include "sslda.h"
#include "utils.h"




/*
* newtons method to find maximum likelihood value for alpha given the allocations in gamma
*
*/

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
	double dif = 0.0, a2 = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
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

				zbeta[person] -= dif * var[person][lastvar];
				zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
				zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person];
				/*if (isn(zbeta[person]))
				{
					zbeta[person] = 0;
					std::cout << "zbeta reset as dif = " << dif << std::endl;
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

			dif = (gdiag + (newbeta[i] / lambda)) / (hdiag + (1.0 / lambda));
			if (fabs(dif) > step[i])	
				dif = (dif > 0.0) ? step[i] : -step[i];

			step[i] = ((2.0 * fabs(dif)) > (step[i] / 2.0)) ? 2.0 * fabs(dif) : (step[i] / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			newbeta[i] -= dif;
			lastvar = i;
		}
		
		for( i = 0 ; i < nvar ; i++)
			newlk -= (newbeta[i] * newbeta[i]) / (2.0 * lambda);
		if (fabs(1.0 - (newlk / loglik)) <= setting->MAX_EPS) break;
		loglik = newlk;
	}   /* return for another iteration */
	

	for (person = nused - 1; person >= 0; person--)
	{
		zbeta[person] -= dif * var[person][lastvar];
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
	double dif = 0.0, a2 = 0.0,  cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
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
					zbeta[person] -= dif * var[person][lastvar];
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

			dif = (gdiag + (newbeta[i] / lambda)) / (hdiag + (1.0 / lambda));
			if (fabs(dif) > step[i])
				dif = (dif > 0.0) ? step[i] : -step[i];
			step[i] = ((2.0 * fabs(dif)) > (step[i] / 2.0)) ? 2.0 * fabs(dif) : (step[i] / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			newbeta[i] -= dif;
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
	double dif = 0.0, a2 = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
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
					zbeta[person] -= dif * c->docs[person]->counts[lastword];
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

			dif = (gdiag + (newbeta[i] / lambda)) / (hdiag + (1.0 / lambda));
			if (fabs(dif) > step[i])
				dif = (dif > 0.0) ? step[i] : -step[i];

			step[i] = ((2.0 * fabs(dif)) > (step[i] / 2.0)) ? 2.0 * fabs(dif) : (step[i] / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			newbeta[i] -= dif;
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
			zbeta[person] -= dif * c->docs[person]->counts[lastword];
			zbeta[person] = zbeta[person] >22 ? 22 : zbeta[person];
			zbeta[person] = zbeta[person] < -200 ? -200 : zbeta[person]; //update zbeta with last beta change before returning.
		}
	}
	
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

