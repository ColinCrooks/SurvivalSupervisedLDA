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



#include <time.h>
#include <math.h>
#include <omp.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <iostream>




#include "sslda.h"
#include "utils.h"
#include "opt.h"

sslda::sslda()
{
    //creator
	alpha = nullptr;
	eta = nullptr;
    num_topics = 0;
    size_vocab = 0;
	time_start = 0;
	events = nullptr;
	event_times = 0;
	basehaz = nullptr; 
	cumbasehaz = nullptr;
	ldelta = 0.0;
	ddelta = nullptr;
	topic_beta = nullptr;
	covariatesN = 0;
	cov_beta = nullptr;
	lambda = 0;

}

sslda::~sslda()
{
    free_model();
}

/*
 * init the model
 */

void sslda::init(double alpha_, int num_topics_,
	int covariatesN_, const corpus * c)
{
	size_vocab = c->size_vocab;
	num_topics = num_topics_;
	alpha = new double[num_topics];
	for (int k = 0; k < num_topics ; k++) 
		alpha[k] = alpha_;
	eta = new double[size_vocab];
	for (int w = 0; w < c->size_vocab; w++) 
		eta[w] = 1.0 / static_cast<double>(c->size_vocab);
	lambda = 0.0;
	covariatesN = covariatesN_;
    event_times = c->event_times;
	events = new int[event_times];
	for (int r = 0; r < event_times; r++)
	{
		events[r] = 0;
	}
	time_start = c->time_start;
	ldelta = 0.0;
	ddelta = new double* [num_topics];
    for (int k = 0; k < num_topics; k++)
    {
		ddelta[k] = new double[size_vocab];
		for (int w = 0; w < size_vocab; w++)
		{
			ddelta[k][w] = 0.0;
		}
    }
	
	basehaz = new double  [event_times];
	for (int r = 0; r < event_times; r++)
		basehaz[r] = 0.0;
	cumbasehaz = new double[event_times];
	for (int r = 0; r < event_times; r++)
		cumbasehaz[r] = 0.0;

	topic_beta = new double  [num_topics];
	for (int k = 0; k < num_topics; k++)
		topic_beta[k] = 0.0;


	if (covariatesN > 0)
	{
		cov_beta = new double[covariatesN];
		for (int k = 0; k < covariatesN; k++)
			cov_beta[k] = 0.0;
	}
}

/*
 * free the model
 */

void sslda::free_model()
{
	size_vocab = 0;
	num_topics = 0;
	lambda = 0.0;
	covariatesN = 0;
	event_times = 0;
	time_start = 0;
	lambda = 0;
	ldelta = 0.0;
	if (events != nullptr)
	{
		delete[] events;
		events = nullptr;
	}

	if (basehaz != nullptr)
	{
		delete[] basehaz;
		basehaz = nullptr;
	}
	if (cumbasehaz != nullptr)
	{
		delete[] cumbasehaz;
		cumbasehaz = nullptr;
	}
	if (eta != nullptr)
	{
		delete[] eta;
		eta = nullptr;
	}
	if (alpha != nullptr)
	{
		delete[] alpha;
		alpha = nullptr;
	}

	if (ddelta != nullptr)
	{
		for (int k = 0; k < num_topics; k++)
			delete[] ddelta[k];

		delete[] ddelta;
		ddelta = nullptr;
	}
    if (topic_beta != nullptr)
    {
        delete []topic_beta;
       topic_beta = nullptr;
    }
	if (cov_beta != nullptr)
	{
		delete[] cov_beta;
		cov_beta = nullptr;
	}
}

/*
 * save the model in the binary format
 */

int sslda::save_model(const char * filename)
{
	ofstream file(filename, ios::binary);
	if (file.is_open())
	{
		file.write(reinterpret_cast<char*>(&num_topics), sizeof (int));
		file.write(reinterpret_cast<char*>(&covariatesN), sizeof (int));
		file.write(reinterpret_cast<char*>(&size_vocab), sizeof (int));
		file.write(reinterpret_cast<char*>(&event_times), sizeof (int));
		file.write(reinterpret_cast<char*>(&lambda), sizeof (double));
		file.write(reinterpret_cast<char*>(&ldelta), sizeof (double));
		for (int k = 0; k < num_topics; k++)
			file.write(reinterpret_cast<char*>(&ddelta[k][0]), sizeof(double)* size_vocab);
		file.write(reinterpret_cast<char*>(&topic_beta[0]), sizeof(double)* num_topics);
		if (cov_beta != nullptr)
			file.write(reinterpret_cast<char*>(&cov_beta[0]), sizeof(double)* covariatesN);
		file.write(reinterpret_cast<char*>(&alpha[0]), sizeof(double) * num_topics);
		file.write(reinterpret_cast<char*>(&eta[0]), sizeof(double) * size_vocab);
		file.close();
	}
	else
	{
		std::cerr << "Failed to save model file in " << filename << std::endl;
		return -1;
	}
	return 1;
}


int sslda::load_model(const char * filename)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		file.read(reinterpret_cast<char*>(&num_topics), sizeof (int));
		file.read(reinterpret_cast<char*>(&covariatesN), sizeof (int));
		file.read(reinterpret_cast<char*>(&size_vocab), sizeof (int));
		file.read(reinterpret_cast<char*>(&event_times), sizeof (int));
		file.read(reinterpret_cast<char*>(&lambda), sizeof (double));
		file.read(reinterpret_cast<char*>(&ldelta), sizeof (double));
		ddelta = new double *[num_topics];
		for (int k = 0; k < num_topics; k++)
		{
			ddelta[k] = new double[size_vocab];
			file.read(reinterpret_cast<char*>(&ddelta[k][0]), sizeof(double)* size_vocab);
		}

		topic_beta = new double[num_topics];
		file.read(reinterpret_cast<char*>(&topic_beta[0]), sizeof(double)* num_topics);

		if (covariatesN > 0)
		{
			cov_beta = new double[covariatesN];
			file.read(reinterpret_cast<char*>(&cov_beta[0]), sizeof(double)* covariatesN);
		}
		alpha = new double[num_topics];
		file.read(reinterpret_cast<char*>(&alpha[0]), sizeof (double) * num_topics);
		eta = new double[size_vocab];
		file.read(reinterpret_cast<char*>(&eta[0]), sizeof(double) * size_vocab);
		file.close();
	}
	else
	{
		std::cerr << "Failed to load model file from " << filename << std::endl;
		return -1;
	}

	return 1;
}

/*
 * save the model in the text format
 */

int sslda::save_model_text(const char * filename)
{
	ofstream file(filename);
	if (file.is_open())
	{
		file << "number of topics: " << num_topics << std::endl << "size of vocab: " << size_vocab << std::endl;
		file << "number of event times: " << event_times << std::endl << "lambda: " << lambda << std::endl << "number of covariates: " << covariatesN << std::endl;
		file << "log probability of each word conditional on topic: " << std::endl; // in log space
		for (int j = 0; j < size_vocab; j++)
		{
			for (int k = 0; k < num_topics; k++)
				file << ddelta[k][j] << " ";
			file << std::endl;
		}
		if (cov_beta != nullptr)
		{
			file << "Coefficients for covariates : " << std::endl;
			for (int j = 0; j < covariatesN; j++)
				file << cov_beta[j] << " ";
			file << std::endl;
		}

		file << "Coefficients for topics: " <<std::endl;
		for (int j = 0; j < num_topics; j++)
			 file << topic_beta[j] << " ";
		file << std::endl;

		file << "Baseline hazard: " << std::endl;
		for (int r = 0; r < event_times; r++)
			file << basehaz[r] <<std::endl;
		file << "alpha: " << std::endl;
		for (int k = 0; k < num_topics; k++) 
			file << alpha[k] << " ";
		file << std::endl;
		file << "eta: ";
		for (int k = 0; k < size_vocab; k++)
			file << eta[k] << " ";
		file << std::endl;
		file.close();
	}
	else
	{
		std::cerr << "Failed to save model file in " << filename << std::endl;
		return -1;
	}
	return 1;
}

/*
 * create the data structure for sufficient statistic 
 */

suffstats * sslda::new_suffstats(int num_docs)
{
    suffstats * ss = new suffstats;
    ss->num_docs = num_docs;
	ss->time_start = 0;
    ss->word_total_ss = new double [num_topics];
	for (int k = 0; k < num_topics; k++)
		ss->word_total_ss[k] = 0.0;
    ss->word_ss = new double * [num_topics];
	ss->group = new int [num_docs];
	
    for (int k = 0; k < num_topics; k ++)
    {
        ss->word_ss[k] = new double [size_vocab];
		for (int w = 0; w < size_vocab; w++)
			ss->word_ss[k][w] = 0.0;
    }

    ss->z_bar =  new double * [num_docs];
	if (covariatesN > 0)
		ss->covariates = new double *[num_docs];
	else
		ss->covariates = nullptr;
    for (int d = 0; d < num_docs; d ++)
    {
		if (covariatesN > 0)
		{
			ss->covariates[d] = new double[covariatesN];
			for (int k = 0; k < covariatesN; k++)
				ss->covariates[d][k] = 0;
		}
        ss->z_bar[d] = new double [num_topics ]; 
		for (int k = 0; k < num_topics; k++)
			ss->z_bar[d][k] = 0.0;
    }
	ss->times = new int [num_docs];
	for (int d = 0; d < num_docs; d++)
		ss->times[d] = 0;
	ss->time0 = new int[num_docs];
	for (int d = 0; d < num_docs; d++)
		ss->time0[d] = 0;
    ss->labels = new int [num_docs];
	for (int d = 0; d < num_docs; d++)
		ss->labels[d] = 0;
	ss->mark = nullptr;
	ss->alpha_ss = new double [num_topics];
	for (int k = 0; k < num_topics; k++) ss->alpha_ss[k] = 0.0;
    return(ss);
}


/*
 * initialize the sufficient statistics with zeros
 */

void sslda::zero_initialize_ss(suffstats * ss)
{
	for (int k = 0; k < num_topics; k++)
	{
		ss->word_total_ss[k] = 0.0;
	}
	for (int k = 0; k < num_topics; k++)
		for (int n = 0; n < size_vocab; n++)
			ss->word_ss[k][n] = 0.0;

	for (int d = 0; d < ss->num_docs; d++)
		for (int k = 0; k < num_topics; k++)
			ss->z_bar[d][k] = 0.0;
	for (int k = 0; k < num_topics; k++) ss->alpha_ss[k] = 0.0;
}


/*
 * initialize the sufficient statistics with random numbers 
 */

void sslda::random_initialize_ss(suffstats * ss, const corpus* c, const settings* setting)
{
	boost::random::mt19937 rng;  
    boost::random::uniform_01<double> rng_uniform;
    time_t seed;
    time(&seed);
	rng.seed(static_cast<int>(seed));
    int k = 0, w = 0, d = 0, g = 0;
    for (k = 0; k < num_topics; k++)
    {
        for (w = 0; w < size_vocab; w++)
        {
			ss->word_ss[k][w] = 1.0 / static_cast<double>(size_vocab)+(0.1 * rng_uniform(rng));
            ss->word_total_ss[k] += ss->word_ss[k][w];
        }
    }

	ss->num_docs = c->num_docs;
	ss->time_start = c->time_start;
	for (d = 0; d < c->num_docs; d++)
	{
		document * doc = c->docs[d];
		ss->labels[d] = doc->label;
		ss->times[d] = c->docs[d]->t_exit;
		ss->time0[d] = c->docs[d]->t_enter;
		ss->group[d] = static_cast<int>(floor(rng_uniform(rng) * static_cast<double>(setting->CROSSVAL)));
		double total = 0.0;
		if (covariatesN > 0)
			for (w = 0; w < covariatesN; w++)
				ss->covariates[d][w] = c->docs[d]->covariates[w];
		for (k = 0; k < num_topics; k++)
		{
			ss->z_bar[d][k] = rng_uniform(rng);
			total += ss->z_bar[d][k];
		}
	}
		

    int j  = 0;
	int * gj = new int[setting->CROSSVAL];
	for (k = 0; k < setting->CROSSVAL; k++)
		gj[k] = 0;
	ss->mark = new int *[ss->num_docs];
	for (d = c->num_docs - 1; d>=0; d--)
	{
		ss->mark[d] = new int[setting->CROSSVAL + 1];
		for (k = 0; k < setting->CROSSVAL + 1; k++)
			ss->mark[d][k] = 0;

		if (ss->times[d] == ss->times[d - 1])
		{
			j += ss->labels[d];

			for (g = 0; g < setting->CROSSVAL; g++)
				if (ss->group[d] == g)
					gj[g] += ss->labels[d];
		}
		else
		{
			for (g = 0; g < setting->CROSSVAL; g++)
			{
				if (ss->group[d] == g)
					ss->mark[d][g] = gj[g] + ss->labels[d];
				else
					ss->mark[d][g] = gj[g];
				gj[g] = 0;
			}
			ss->mark[d][setting->CROSSVAL] = j + ss->labels[d]; //Last patient at this time point - store number of deaths
			j = 0;
		}
	}

	ss->mark[0] = new int[setting->CROSSVAL + 1];
	for (k = 0; k < setting->CROSSVAL + 1; k++)
		ss->mark[0][k] = 0;
	for (g = 0; g < setting->CROSSVAL; g++)
	{
		if (ss->group[0] == g)
			ss->mark[0][g] = gj[g] + ss->labels[0];
		else
			ss->mark[0][g] = gj[g];
	}
	ss->mark[0][setting->CROSSVAL] = j + ss->labels[0];
	delete[] gj;
	gj = nullptr;
	for (k = 0; k < num_topics; k++) ss->alpha_ss[k] = 0.0;
}



void sslda::corpus_initialize_ss(suffstats* ss, const corpus* c, const settings* setting)
{
	int num_docs = ss->num_docs;
	boost::random::mt19937 rng;
	time_t seed;
	time(&seed);
	rng.seed(static_cast<int>(seed));
	int k = 0, n = 0, d = 0, g = 0, i = 0, w = 0;
	boost::random::uniform_01<double> rng_uniform;
	for (k = 0; k < num_topics; k++)
	{
		for (i = 0; i < setting->NUM_INIT; i++)
		{
			d = static_cast<int>(floor(rng_uniform(rng) * num_docs));
			document * doc = c->docs[d];
			for (n = 0; n < doc->length; n++)
				ss->word_ss[k][doc->words[n]] += doc->counts[n];

		}
		for (w = 0; w < size_vocab; w++)
		{
			ss->word_ss[k][w] = 2 * ss->word_ss[k][w] + 5 + rng_uniform(rng);
			ss->word_total_ss[k] = ss->word_total_ss[k] + ss->word_ss[k][w];
		}
	}

	ss->num_docs = num_docs;
	ss->time_start = c->time_start;
	for (d = 0; d < num_docs; d++)
	{
		document * doc = c->docs[d];
		ss->labels[d] = doc->label;
		ss->times[d] = c->docs[d]->t_exit;
		ss->time0[d] = c->docs[d]->t_enter;
		ss->group[d] = static_cast<int>(floor(rng_uniform(rng) * static_cast<double>(setting->CROSSVAL)));
		double total = 0.0;
		if (covariatesN > 0)
			for (w = 0; w < covariatesN; w++)
				ss->covariates[d][w] = c->docs[d]->covariates[w];
		for (k = 0; k < num_topics; k++)
		{
			ss->z_bar[d][k] = rng_uniform(rng);
			total += ss->z_bar[d][k];
		}

	}

	int j = 0;
	int * gj = new int[setting->CROSSVAL];
	for (k = 0; k < setting->CROSSVAL; k++)
		gj[k] = 0;
	ss->mark = new int *[ss->num_docs];
	for (d = c->num_docs - 1; d >= 0; d--)
	{
		ss->mark[d] = new int[setting->CROSSVAL + 1];
		for (k = 0; k < setting->CROSSVAL + 1; k++)
			ss->mark[d][k] = 0;

		if (ss->times[d] == ss->times[d - 1])
		{
			j += ss->labels[d];

			for (g = 0; g < setting->CROSSVAL; g++)
				if (ss->group[d] == g)
					gj[g] += ss->labels[d];
		}
		else
		{
			for (g = 0; g < setting->CROSSVAL; g++)
			{
				if (ss->group[d] == g)
					ss->mark[d][g] = gj[g] + ss->labels[d];
				else
					ss->mark[d][g] = gj[g];
				gj[g] = 0;
			}
			ss->mark[d][setting->CROSSVAL] = j + ss->labels[d]; //Last patient at this time point - store number of deaths
			j = 0;
		}
	}

	ss->mark[0] = new int[setting->CROSSVAL + 1];
	for (k = 0; k < setting->CROSSVAL + 1; k++)
		ss->mark[0][k] = 0;
	for (g = 0; g < setting->CROSSVAL; g++)
	{
		if (ss->group[0] == g)
			ss->mark[0][g] = gj[g] + ss->labels[0];
		else
			ss->mark[0][g] = gj[g];
	}
	ss->mark[0][setting->CROSSVAL] = j + ss->labels[0];
	delete[] gj;
	gj = nullptr;
	for (k = 0; k < num_topics; k++) ss->alpha_ss[k] = 0.0;
}



//void sslda::load_model_initialize_ss(suffstats* ss, corpus * c)
//{
//    int num_docs = ss->num_docs;
//
//    for (int d = 0; d < num_docs; d ++)       
//    {                                                                                                    
//       document * doc = c->docs[d];
//       ss->labels[d] = doc->label;
//	   ss->times[d] = c->docs[d]->t;
//	   if (covariatesN > 0)
//			for (int w = 0; w < covariatesN; w++)
//				ss->covariates[d][w] = c->docs[d]->covariates[w];
//    }  
//	ss->num_docs = num_docs;  
//	double temp=0;
//    int j=0;
//	int i=0;
//    for (i=num_docs-1; i>0; i--) 
//	{
//		if (((ss->times[i]) == (ss->times[i-1])) && (i != 1)) 
//		{
//			j += ss->labels[i];
//			ss->mark[i]=0;
//	    }
//		else  
//		{
//			ss->mark[i] = j + ss->labels[i];
//			j=0;
//	    }
//	}
//    ss->mark[0]  = j + ss->labels[i];
//	ss->alpha_ss = 0.0;
//}

void sslda::free_suffstats(suffstats * ss)
{
	delete[] ss->word_total_ss;

	for (int k = 0; k < num_topics; k++)
		delete[] ss->word_ss[k];

	delete[] ss->word_ss;


	for (int d = 0; d < ss->num_docs; d++)
	{
		delete[] ss->z_bar[d];
		delete[] ss->mark[d];
		if (ss->covariates!=nullptr)
			delete[] ss->covariates[d];
	}
    
    delete [] ss->z_bar;
	delete [] ss->times;
	delete [] ss->time0;
    delete [] ss->labels;
	delete [] ss->group;
	delete [] ss->mark;


	if (ss->covariates != nullptr)
		delete [] ss->covariates;
	ss->z_bar = nullptr;
	ss->times = nullptr;
	ss->time0 = nullptr;
	ss->labels = nullptr;
	ss->group = nullptr;
	ss->mark = nullptr;
	ss->covariates = nullptr;
	ss->word_ss = nullptr;


    delete ss;
}



int sslda::v_em(const corpus * c, const settings * setting,
	const char * start, const char * directory /*, const corpus * c_val*/ )
{
	const int max_length = c->max_length;
	double **var_gamma;
	double * xb = new double[event_times];
	double likelihood = 0.0, likelihood_old = 0.0, converged = 1.0, f = numeric_limits<double>::lowest();
	int d, i, k;
	alpha[num_topics - 1] *= setting->WT;
	std::cout << std::endl << " ******** Variational EM for regularised survival supervised LDA ******** " << std::endl;
	std::cout << "Alpha = ";
	for (k = 0; k < num_topics; k++) std::cout << alpha[k] << ", ";
	std::cout<< std::endl;
	std::cout <<"Number of topics = " << num_topics << std::endl;

	var_gamma = new double *[c->num_docs];
	for (d = 0; d < c->num_docs; d++)
	{
		var_gamma[d] = new double[num_topics];
		for (k = 0; k < num_topics; k++)
			var_gamma[d][k] = 0.0;
	}
	

	for (int r = 0; r < event_times; r++)
	{
		xb[r] = 0.0;
	}
	std::cout << "Initializing latent variables using ";
	suffstats * ss = new_suffstats(c->num_docs);
	if (strcmp(start, "seeded") == 0)
	{
		std::cout <<"a sample of documents from the corpus." << std::endl;
		corpus_initialize_ss(ss, c, setting);
		mle(ss, 0, setting);

		for (d = (c->num_docs) - 1; d >= 0; d--)
		{
			int time_index_exit = (c->docs[d]->t_exit) - (c->time_start);
			int time_index_entry = (c->docs[d]->t_enter) - (c->time_start);
			if (time_index_exit > 0)
			{
				events[time_index_exit] += (c->docs[d]->label);
				for (int r = time_index_entry; r <= time_index_exit; r++)
					xb[r] += 1.0;
				
			}
			
			if (d == 0 || c->docs[d]->t_exit!= c->docs[d - 1]->t_exit)
			{
				basehaz[time_index_exit] = static_cast<double>(events[time_index_exit]) / xb[time_index_exit];
			}
			if (basehaz[time_index_exit] < 1e-100 || isn(basehaz[time_index_exit]))
				basehaz[time_index_exit] = 1e-100;
		}

	}
	else if (strcmp(start, "random") == 0)
	{
		std::cout << "random numbers." << std::endl;
		random_initialize_ss(ss, c, setting);
		mle(ss, 0, setting);
		for (d = (c->num_docs) - 1; d >= 0; d--)
		{
			int time_index_exit = (c->docs[d]->t_exit) - (c->time_start);
			int time_index_entry = (c->docs[d]->t_enter) - (c->time_start);
			if (time_index_exit > 0)
			{
				events[time_index_exit] += (c->docs[d]->label);
				for (int r = time_index_entry; r <= time_index_exit; r++)
					xb[r] += 1.0;

			}

			if (d == 0 || c->docs[d]->t_exit!= c->docs[d - 1]->t_exit)
			{
				basehaz[time_index_exit] = static_cast<double>(events[time_index_exit]) / xb[time_index_exit];
			}
			if (basehaz[time_index_exit] < 1e-100 || isn(basehaz[time_index_exit]))
				basehaz[time_index_exit] = 1e-100;
		}
	
	}
	else
	{
		std::cerr << std::endl << "Need to specify whether to initialise latent with seeding from the corpus or randomly" << std::endl;
		return -1;
	}
	delete[] xb;
	cumbasehaz[0] = basehaz[0];
	for (int r = 1; r < event_times; r++) // exp(-cumbasehaz) is the survival function to time t - 1
		cumbasehaz[r] = cumbasehaz[r-1] + basehaz[r];


	stringstream filename;
	filename << directory << "\\likelihood.dat";
	ofstream likelihood_file(filename.str().c_str());
	if (!likelihood_file.is_open())
	{
		std::cerr << "Unable to create likelihood file\n" << std::endl;
		return -1;
	}
	std::cout << "Likelihood file opened to save output." << std::endl;

	int BETA_UPDATE = 0;

	f = 0.0;
	i = 0;

	int var_max_iter = setting->VAR_MAX_ITER;
	double var_converged = 0.1; 
	double em_converged = 0.1;
	while (( /*(converged < 0) ||*/ (fabs(converged) > setting->EM_CONVERGED) || (i <= setting->LDA_INIT_MAX + 2)) && (i <= setting->EM_MAX_ITER))
	{
		std::cout << std::endl << "**** EM iteration " << ++i << " ****" << std::endl;
		likelihood = 0.0;
		zero_initialize_ss(ss);
		if (i > setting->LDA_INIT_MAX )	BETA_UPDATE = 1;

		///////  e-step    /////// 
		std::cout << std::endl << "**** E - Step ****" << std::endl;



#pragma omp parallel reduction(+:likelihood) default(none) shared(c, var_gamma, ss, BETA_UPDATE, var_converged, var_max_iter)
		{ 
			int size = omp_get_num_threads(); // get total number of processes
			int rank = omp_get_thread_num(); // get rank of current

			///variables for working in estimation step, allocated here so done once per model
			double * oldphi = new double[num_topics];
			double * dig = new double[num_topics];
			double * cbhz_params = new double[num_topics];

			for (int kk = 0; kk < num_topics; kk++)
			{
				oldphi[kk] = 0.0;
				dig[kk] = 0.0;
				cbhz_params[kk] = 0.0;
			}

			double ** phi = new double *[max_length];
			for (int nl = 0; nl < max_length; nl++)
			{
				phi[nl] = new double[num_topics];
				for (int kk = 0; kk < num_topics; kk++)
					phi[nl][kk] = 0.0;
			}
			
			for (int docN = (rank * c->num_docs / size); docN < (rank + 1) * c->num_docs / size; docN++)
			{
				likelihood += doc_e_step(c->docs[docN], docN,  var_gamma[docN], phi, oldphi, dig, cbhz_params, ss, BETA_UPDATE, var_converged, var_max_iter);
			}

			for (int nl = 0; nl < max_length; nl++)
				delete[] phi[nl];
			delete[] phi;
			phi = nullptr;
			delete[] dig;
			dig = nullptr;
			delete[] cbhz_params;
			cbhz_params = nullptr;
			delete[] oldphi;
			oldphi = nullptr;

		}
		likelihood+=ldelta;

		std::cout << std::endl << "Likelihood: " << likelihood << std::endl << std::endl;

		///////  m-step    /////// 
		std::cout << "**** M - Step ****" <<std::endl;

		f = mle(ss, BETA_UPDATE, setting);
//		std::cout << std::endl << "M step Likelihood: " << f << std::endl << std::endl;
//  	if (setting->includeETA==1)		likelihood += ldelta;
		///////  check for convergence    /////// 
		converged = (likelihood_old - likelihood) / likelihood_old;
		if (converged < 0) 
			var_max_iter *=  2;

//		std::cout << std::endl << "Overall Likelihood: " << likelihood << std::endl << std::endl;
		likelihood_old = likelihood;

		///////  output model and likelihood    /////// 
		likelihood_file << "EM Likelihood " << likelihood <<  "\t" << converged << "\t" <<" MLE Likelihood " << f << std::endl;
		likelihood_file.flush();
		
		if (converged <= em_converged &&  converged > setting->EM_CONVERGED)
		{
			if (var_converged> setting->VAR_CONVERGED)
				var_converged /= 10;
			em_converged /= 10;
		}
		std::cout << "Var converged = " << var_converged << " EM converged = " << em_converged << std::endl;

	}
	//////// Held out data
	//if (c_val != nullptr)
	//{
	//	double p_val = 0.0;
	//	double lik_val = 0.0;
	//	infer_only(c_val, setting, &p_val, &lik_val, directory, 0);
	//}

	///////  output the final model    /////// 
	
	likelihood_file << "Final likelihood " << likelihood << "\t" << "Convergence " << converged << std::endl
		<< "Perplexity: " << exp(-likelihood / c->num_total_words) << std::endl;
	likelihood_file.close();

	stringstream filename_m;
	filename_m << directory << "//final.model";
	if(save_model(filename_m.str().c_str()) != 1)
		std::cerr <<"Failed to save final model binary file in "<< filename_m.str() <<std::endl;

	stringstream filename_t;
	filename_t << directory << "//final.model.text";
	if (save_model_text(filename_t.str().c_str()) != 1)
		std::cerr << "Failed to save final model text file in " << filename_t.str() << std::endl;

	std::cout << "Final likelihood " << likelihood << "\t" << converged << std::endl;
	std::cout << "Perplexity: " << exp(-likelihood/c->num_total_words) << std::endl;

	//stringstream filename_g;
	//filename_g << directory << "//train-gamma.dat";
	//save_gamma(filename_g.str().c_str(), var_gamma, c->num_docs);
	//std::cout << "Topic allocation gamma saved in " << filename_g.str() << std::endl;

//	stringstream filename_z;
//	filename_z << directory << "//train-zbar.dat";
//	save_zbar(filename_z.str().c_str(), ss->z_bar, c->num_docs);
//	std::cout << "Topic probability z_bar saved in " << filename_z.str() << std::endl;


	free_suffstats(ss);
	for (d = 0; d < c->num_docs; d++)
		delete[] var_gamma[d];
	delete[] var_gamma;
	var_gamma = nullptr;
	return 1;
}


double sslda::mle(suffstats * ss, int BETA_UPDATE, const settings * setting)
{
	int  i, j, k, w, d;
	double * xb = new double[event_times]; //denomimator for hazard of people at risk at each time point
	double  xb2 = 0.0, exb = 0.0, exb2 = 0.0;
	
	if (setting->includeETA == 1)
	{
		ldelta = 0.0;
		double eta_sum{ 0 };
		double * eta_ss = nullptr;
		eta_ss = new double[size_vocab];
		for (w = 0; w < size_vocab; w++)
		{
			eta_sum += eta[w];
			eta_ss[w] = 0;
		}

		for (k = 0; k < num_topics; k++)
		{
			double delta_sum{0};
			double dig_sumdelta = boost::math::digamma(eta_sum + ss->word_total_ss[k]);
			for (w = 0; w < size_vocab; w++)
			{
				double delta = eta[w]+ss->word_ss[k][w];
				ddelta[k][w] = (boost::math::digamma(delta) - dig_sumdelta); //dig_sumdelta not necessary for inference but correct for likelihood
				ldelta+= boost::math::lgamma(delta) -  ((delta-1)*ddelta[k][w]);
				ldelta+=-boost::math::lgamma(eta[w])+ ((eta[w]-1)*ddelta[k][w]) ;
				delta_sum+=delta;
				if (ddelta[k][w] < -800)
					ddelta[k][w] = -800;
/* 				if (w > 0)
					ddelta_norm = log_sum(ddelta_norm, ddelta[k][w]);
				else
					ddelta_norm = ddelta[k][w]; // note, phi is in log space
 */
				eta_ss[w] += ddelta[k][w];  // sufficient statistic for eta update
			}
			ldelta+=boost::math::lgamma(eta_sum);
			ldelta-=boost::math::lgamma(delta_sum);
/* 			for (w = 0; w < size_vocab; w++)
				ddelta[k][w] = ddelta[k][w] - ddelta_norm; //shouldn't be necessary?
 */		}
		if (setting->ETA == 1 && BETA_UPDATE == 1 )
		{
			opt_alpha(eta, eta_ss,
				num_topics,
				size_vocab,
				setting); // means some words are a priori more likely than others for each topic across whole document
			delete[] eta_ss;
		}
	}
	else
	{
		for (k = 0; k < num_topics; k++)
		{
			for (w = 0; w < size_vocab; w++)
			{
				if (ss->word_ss[k][w] > 0)
					ddelta[k][w] = log(ss->word_ss[k][w]) - log(ss->word_total_ss[k]);
				else
					ddelta[k][w] = -800.0;
			}
		}
	}

	
	if (BETA_UPDATE == 0) return 0;
	if (setting->ALPHA == 1)
	{
		opt_alpha(alpha, ss->alpha_ss,
			ss->num_docs,
			num_topics,
			setting);  // means some topics will be more likely across whole corpus
		std::cout << "New alpha: ";
		for (k = 0; k < num_topics; k++)
			std::cout << alpha[k] << ", ";
		std::cout << std::endl;
	}
	

	double max_ss(0.0);
	int base_index(0);
	for (int i = 0; i < num_topics; i++) 
	{
		if (max_ss < ss->word_total_ss[i]) 
		{
			base_index = i;
			max_ss=ss->word_total_ss[i];
		}
	}
	base_index = base_index + covariatesN;
	
	const int nt = num_topics + covariatesN ; 
	lambda = pow(10, setting->LAMBDASTART);
	const int lambdarange = 1 + setting->LAMBDAEND - setting->LAMBDASTART;
	double ** var = new double *[ss->num_docs];
	for (d = 0; d < ss->num_docs; d++)
	{
		var[d] = new double[nt];
		for (w = 0; w < covariatesN; w++)
			var[d][w] = static_cast<double>(ss->covariates[d][w]);
		for (w = covariatesN; w < nt; w++)
			var[d][w] = ss->z_bar[d][w - covariatesN]; 
	}
	for (w = 0; w < nt; w++)
	{
		double temp = 0.0;
		for (d = 0; d < ss->num_docs; d++)
			temp += var[d][w];
		temp /= ss->num_docs;

		for (d = 0; d < ss->num_docs; d++)
			var[d][w] -= temp;
	}

	
	double f = 0.0;
	double cv = numeric_limits<double>::lowest();
	double * newbeta = new double[nt];
	for (k = 0; k < nt; k++)
		newbeta[k] = 0.0;
	
	double * fsum = new double[lambdarange * setting->CROSSVAL];
	for (i = 0; i < lambdarange * setting->CROSSVAL; i++)
		fsum[i] = 0.0;

	double ** cvbeta = new double *[lambdarange * setting->CROSSVAL];
	for (i = 0; i < lambdarange * setting->CROSSVAL; i++)
	{
		cvbeta[i] = new double[nt];
		for (k = 0; k < covariatesN; k++)
			cvbeta[i][k] = 0.0;

		for (k = covariatesN; k < nt; k++)
			cvbeta[i][k] = 0.0;

	}

	
	
#pragma omp parallel  default(none) shared(cvbeta, ss, setting, fsum, var, base_index)
	{
		int size = omp_get_num_threads(); // get total number of processes
		int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )
		for (int ii = (rank * lambdarange * setting->CROSSVAL / size); ii < ((rank + 1) * lambdarange * setting->CROSSVAL / size); ii++)
		{
			fsum[ii] = cox_reg_cross_val(
				ii % setting->CROSSVAL,
				cvbeta[ii], var, nt,
				pow(10.0, (static_cast<double>(setting->LAMBDAEND)
				- floor(static_cast<double>(ii) / static_cast<double>(setting->CROSSVAL))))
				, ss, setting, base_index
				);
		}
	}




	for (i = 0; i < lambdarange; i++)
	{
		for (j = 1; j < setting->CROSSVAL; j++)
		{
			fsum[i * setting->CROSSVAL] += fsum[(i * setting->CROSSVAL) + j];
			for (k = 0; k < nt; k++)
				cvbeta[i * setting->CROSSVAL][k] += cvbeta[(i * setting->CROSSVAL) + j][k];
		}
		//	cout.precision(15);
		//	std::cout << "fsum " << i * setting->CROSSVAL << ": " << std::fixed << fsum[i * setting->CROSSVAL] <<std::endl;
		if (fsum[i * setting->CROSSVAL] >(cv + setting->MAX_EPS))
		{
			cv = fsum[i * setting->CROSSVAL];
			lambda = pow(10.0, (static_cast<double>(setting->LAMBDAEND) - static_cast<double>(i)));
			for (k = 0; k < nt; k++)
				newbeta[k] = cvbeta[i * setting->CROSSVAL][k] / static_cast<double>(setting->CROSSVAL);
		}
		//cout.precision(6);
	}
	std::cout << " Selected Lambda = " << lambda << ", ";
	delete[] fsum;
	fsum = nullptr;
	for (i = 0; i < lambdarange * setting->CROSSVAL; i++)
		delete[] cvbeta[i];
	delete[] cvbeta;
	cvbeta = nullptr;

	

	/////////////////
	double * zbeta = new double[ss->num_docs];
	for (d = 0; d < ss->num_docs; d++)
		zbeta[d] = 0.0;

	int miter = 0;


	miter = cox_reg(newbeta, zbeta, var, nt, lambda, ss, &f, setting, base_index);

	std::cout << " M iter = " << miter << ", f = " << f << std::endl;
	while (isn(f))
	{
		lambda /= 100.0;
		for (k = 0; k < num_topics; k++)
			newbeta[k] = 0.0;
		std::cout << std::endl << "Proportional hazards model failed to converge. Setting lambda to " << lambda << std::endl;
		miter = cox_reg(newbeta, zbeta, var, nt, lambda, ss, &f, setting, base_index);
		std::cout << "Coefficients : ";
		for (k = 0; k < nt; k++)
		std::cout << newbeta[k] << ", ";
		std::cout << std::endl;
		
	}



	for (k = 0; k < covariatesN; k++)
		cov_beta[k] = newbeta[k];
	for (k = covariatesN; k < nt; k++)
		topic_beta[k - covariatesN] = newbeta[k];

	if (covariatesN > 0)
	{
		std::cout << "Coefficients for covariates : ";
		for (k = 0; k < covariatesN; k++)
			std::cout << cov_beta[k] << ", ";
		std::cout << std::endl;
	}
	std::cout << "Coefficients for topics : ";
	for (k = 0; k < num_topics; k++)
		std::cout << topic_beta[k] << ", ";
	std::cout << std::endl;


	for (int r = 0; r < event_times; r++)
	{
		basehaz[r] = 0.0;
		xb[r] = 0.0;
	}
	
				
	for (d = (ss->num_docs) - 1; d >= 0; d--)
	{
		int time_index_entry = (ss->time0[d]) - (ss->time_start);
		int time_index_exit = (ss->times[d]) - (ss->time_start);
		xb2 = 0.0;
		exb2 = 0.0;
		xb2 += zbeta[d];
		if (ss->labels[d] > 0)
			exb2 += zbeta[d];
		
		//std::cout << "exb " << exb << " exb2 " << exb2;
		for (int r = time_index_entry; r <= time_index_exit; r++)
			xb[r] = log_sum(xb[r], xb2);
		if (ss->labels[d] > 0)
			exb = log_sum(exb, exb2);
		
		//std::cout << " = " << exb << std::endl << " xb " << xb << " xb2 " << xb2;

		//std::cout << " = " << xb << std::endl;
		if ((d == 0 || ss->times[d] != ss->times[d - 1]) )
		{
			for (i = 0; i < events[time_index_exit]; i++)
				basehaz[time_index_exit] +=
					1.0 
					/ 
					(exp(xb[time_index_exit]) -
							(	
								(
									static_cast<double>(i)
										/ 
										static_cast<double>(events[time_index_exit])
								)
								* exp(exb)
							)
					 ); //efron's method as in survfit4.c in R survival function
			if (isn(basehaz[time_index_exit]) || basehaz[time_index_exit] < 1e-100)
			{
	//			std::cout << "Base haz set to 1e-100 because xb == " << xb[ss->times[d] - 1] << " and exb == " << exb << " so basehaz[time_index_exit] ==" << basehaz[time_index_exit] << std::endl;
				basehaz[time_index_exit] = 1e-100; //log(basehaz) required so a minimum measureable hazard is required to avoid NaN errors.
			}
			exb = 0.0;
		}
		delete[] var[d];
		var[d] = nullptr;
	}
	cumbasehaz[0] = basehaz[0];
	for (int r = 1; r < event_times; r++)
		cumbasehaz[r] = cumbasehaz[r - 1] + basehaz[r];
	delete[] var;
	var = nullptr;

	delete[] newbeta;
	newbeta = nullptr;

	double num = 0.0, den = 0.0;

	int * mark = new int[ss->num_docs];
	for (d = 0; d < ss->num_docs; d++)
		mark[d] = 0;
	int count = 1;
	mark[ss->num_docs - 1] = 1;
	for (d = ss->num_docs - 2; d >= 0; d--)
	{
		if (ss->times[d] == ss->times[d + 1]) //assume sorted by time, so calculate the number needed to jump to next increment in time
			count++;
		else
			count = 1;
		mark[d] = count;
	}

#pragma omp parallel reduction(+:num,den) default(none) shared(zbeta, ss, mark)
	{
		int dl, ddl;
		int size = omp_get_num_threads(); // get total number of processes
		int rank = omp_get_thread_num(); // get rank of current

		for (dl = (rank*ss->num_docs / size); dl<(rank + 1)*ss->num_docs / size; dl++)
		{
			if (ss->labels[dl]>0)
			{
				for (ddl = dl + mark[dl]; ddl < ss->num_docs; ddl++)
				{
					if (ss->times[dl] >= ss->time0[ddl])
					{
						den += 1.0;
						if (zbeta[dl] > zbeta[ddl])
							num += 1.0;
						else if (zbeta[dl] == zbeta[ddl])
							num += 0.5;
					}


				}
			}
		}
	}
	delete[] mark;
	mark = nullptr;
	delete[] zbeta;
	zbeta = nullptr;
	delete[] xb;
	xb = nullptr;

	std::cout << "C Statistic = " << num/den << std::endl;
	return f;
}

double sslda::coxonly(const corpus * c ,  const settings * setting)
{ //needs checking for left censoriing
	int  i,  k, w, d, g, v;
	double * xb = new double[event_times]; //denomimator for hazard of people at risk at each time point
	double xb2 = 0.0, exb = 0.0, exb2 = 0.0;

	int j = 0;
	double f = numeric_limits<double>::lowest();
	
	std::cout << std::endl << " ********Cox regression only with variables in covariate file ******** " << std::endl;

	boost::random::mt19937 rng;
	boost::random::uniform_01<double> rng_uniform;
	time_t seed;
	time(&seed);
	rng.seed(static_cast<int>(seed));
	events = new int[event_times];
	for (int r = 0; r < event_times; r++)
	{
		events[r] = 0;
		xb[r] = 0.0;
	}
	suffstats * ss = new_suffstats(c->num_docs);
	ss->num_docs = c->num_docs;
	for (d = 0; d < c->num_docs; d++)
	{
		document * doc = c->docs[d];
		ss->labels[d] = doc->label;
		ss->times[d] = c->docs[d]->t_exit;
		ss->group[d] = static_cast<int>(floor(rng_uniform(rng) * static_cast<double>(setting->CROSSVAL)));
	}


	int * gj = new int[setting->CROSSVAL];
	for (k = 0; k < setting->CROSSVAL; k++)
		gj[k] = 0;
	ss->mark = new int *[ss->num_docs];
	for (d = c->num_docs - 1; d>0; d--)
	{
		ss->mark[d] = new int[setting->CROSSVAL + 1];
		for (k = 0; k < setting->CROSSVAL + 1; k++)
			ss->mark[d][k] = 0;

		if (ss->times[d] == ss->times[d - 1])
		{
			j += ss->labels[d];

			for (g = 0; g < setting->CROSSVAL; g++)
				if (ss->group[d] == g)
					gj[g] += ss->labels[d];
		}
		else
		{
			for (g = 0; g < setting->CROSSVAL; g++)
			{
				if (ss->group[d] == g)
					ss->mark[d][g] = gj[g] + ss->labels[d];
				else
					ss->mark[d][g] = gj[g];
				gj[g] = 0;
			}
			ss->mark[d][setting->CROSSVAL] = j + ss->labels[d]; //Last patient at this time point - store number of deaths
			j = 0;
		}
	}

	ss->mark[0] = new int[setting->CROSSVAL + 1];
	for (k = 0; k < setting->CROSSVAL + 1; k++)
		ss->mark[0][k] = 0;
	for (g = 0; g < setting->CROSSVAL; g++)
	{
		if (ss->group[0] == g)
			ss->mark[0][g] = gj[g] + ss->labels[0];
		else
			ss->mark[0][g] = gj[g];
	}
	ss->mark[0][setting->CROSSVAL] = j + ss->labels[0];
	delete[] gj;
	gj = nullptr;
	for ( k = 0; k < num_topics; k++) ss->alpha_ss[k] = 0.0;

	for (d = (c->num_docs) - 1; d >= 0; d--)
	{
		int time_index_exit = (ss->times[d]) - (ss->time_start);
		int time_index_entry = (ss->time0[d]) - (ss->time_start);
		if (time_index_exit > 0)
		{
			events[time_index_exit] += (c->docs[d]->label);
			for (int r = time_index_entry; r <= time_index_exit; r++)
				xb[r] += 1.0;

		}
		if (d == 0 || c->docs[d]->t_exit!= c->docs[d - 1]->t_exit)
			for (i = 0; i < events[time_index_exit]; i++)
				basehaz[time_index_exit] += 1.0 / (xb[time_index_exit] - static_cast<double>(i));
		if (basehaz[time_index_exit] < 1e-100 || isn(basehaz[time_index_exit]))
			basehaz[time_index_exit] = 1e-100;
	}

	lambda = pow(10, setting->LAMBDASTART);
	const int lambdarange = 1 + setting->LAMBDAEND - setting->LAMBDASTART;
	
	double * fsum = new double[lambdarange * setting->CROSSVAL];
	for (i = 0; i < lambdarange * setting->CROSSVAL; i++)
		fsum[i] = 0.0;
	double * newbeta = new double[size_vocab];
	for (k = 0; k < size_vocab; k++)
		newbeta[k] = 0.0;
	
	double * zbeta = new double[ss->num_docs];
	for (d = 0; d < ss->num_docs; d++)
		zbeta[d] = 0.0;

	int miter = 0;
	miter = cox_reg_sparse(newbeta, zbeta, c, size_vocab, lambda, ss, &f, setting);

	while (isn(f))
	{
		lambda /= 10.0;
		for (k = 0; k < covariatesN+size_vocab; k++)
			newbeta[k] = 0.0;
		std::cout << std::endl << "Proportional hazards model failed to converge. Setting lambda to " << lambda << std::endl;
		miter = cox_reg_sparse(newbeta, zbeta, c, size_vocab, lambda, ss, &f, setting);

	}


	std::cout << " M iter = " << miter << ", f = " << f << std::endl;

	std::cout << "Coefficients for terms : ";
	for (v = 0; v < size_vocab; v++)
		std::cout << newbeta[v ] << ", ";
	std::cout << std::endl;


	for (int r = 0; r < event_times; r++)
		basehaz[r] = 0.0;

	xb2 = 0.0,  exb2 = 0.0;
	for (d = (ss->num_docs) - 1; d >= 0; d--)
	{
		int time_index_exit = (ss->times[d]) - (ss->time_start);
		int time_index_entry = (ss->time0[d]) - (ss->time_start);
		xb2 = 0.0;
		exb2 = 0.0;
		for (w = 0; w < c->docs[d]->length; w++)
		{
			xb2 += newbeta[c->docs[d]->words[w]] * c->docs[d]->counts[w];
			if (ss->labels[d] > 0)
				exb2 += newbeta[c->docs[d]->words[w]] * c->docs[d]->counts[w];
		}
		if (ss->labels[d] > 0)
			exb = log_sum(exb, exb2);
		for (int r = time_index_entry; r <= time_index_exit; r++)
			xb[r] = log_sum(xb[r], xb2); //  log(exp(xb)+exp(xb2))
		if (d == 0 || ss->times[d] != ss->times[d - 1])
		{
			for (i = 0; i < events[time_index_exit]; i++)
				basehaz[time_index_exit] +=
				1.0 / (exp(xb[time_index_exit])
				- ((static_cast<double>(i) / static_cast<double>(events[time_index_exit]))
				* exp(exb)));
			exb = 0.0;

			if (isn(basehaz[time_index_exit]) || basehaz[time_index_exit] < 1e-100)
				basehaz[time_index_exit] = 1e-100; //log(basehaz) required so a minimum measureable hazard is required to avoid NaN errors.
		}

	}

	delete[] newbeta;
	newbeta = nullptr;


	double num = 0.0, den = 0.0;

	int * mark = new int[ss->num_docs];
	for (d = 0; d < ss->num_docs; d++)
		mark[d] = 0;
	int count = 1;
	mark[ss->num_docs - 1] = 1;
	for (d = ss->num_docs - 2; d >= 0; d--)
	{
		if (ss->times[d] == ss->times[d + 1]) // sorted by time, so calculate the number needed to jump to next increment in time
			count++;
		else
			count = 1;
		mark[d] = count;
	}

#pragma omp parallel reduction(+:num,den) default(none) shared(zbeta, ss, mark)
	{
		int dl, ddl;
		int size = omp_get_num_threads(); // get total number of processes
		int rank = omp_get_thread_num(); // get rank of current

		for (dl = (rank*ss->num_docs / size); dl<(rank + 1)*ss->num_docs / size; dl++)
		{
			if (ss->labels[dl]>0)
			{
				for (ddl = dl + mark[dl]; ddl < ss->num_docs; ddl++)
				{
					if (zbeta[dl] > zbeta[ddl] && ss->times[dl] >= ss->time0[ddl])
						num += 1.0;
					else if (zbeta[dl] == zbeta[ddl] && ss->times[dl] >= ss->time0[ddl])
						num += 0.5;
					den += 1.0;

				}
			}
		}
	}
	delete[] mark;
	mark = nullptr;
	delete[] zbeta;
	zbeta = nullptr;


	std::cout << "C Statistic = " << num / den << std::endl;
	return f;
}


double sslda::doc_e_step(document* doc, int docN,  double* gamma, double** phi, double* oldphi, double* dig, double* cbhz_params, suffstats * ss, int BETA_UPDATE,  const double var_converged, const int var_max_iter)
{
	int n, k;

	double likelihood = 0.0;
	if (BETA_UPDATE == 1)
	{
		likelihood = sslda_inference(doc, gamma, phi, oldphi, dig, cbhz_params, var_converged, var_max_iter);
	}
	else
	{
		likelihood = lda_inference(doc, gamma, phi, oldphi, dig, var_converged, var_max_iter);
	}
	if (isn(likelihood))
		return likelihood;

	   // update sufficient statistics (zero initialised in v_em)

	
    for (n = 0; n < doc->length; n++)
    {
        for (k = 0; k < num_topics; k++)
        {
			
#pragma omp atomic 
				ss->word_ss[k][doc->words[n]] += doc->counts[n] * phi[n][k];
#pragma omp atomic 
				ss->word_total_ss[k] += doc->counts[n] * phi[n][k];
				//std::cout << "Document = " << docN << " Word = " << doc->words[n] << " topic k = " << k << " phi[n][k] = " << phi[n][k] << " doc->counts[n] " << doc->counts[n] << " ss->word_ss[k][doc->words[n]] = " << ss->word_ss[k][doc->words[n]] << " Likelihood "<< likelihood << std::endl;

				ss->z_bar[docN][k] += doc->counts[n] * phi[n][k]; //document specific so doesn't need to be atomic
	//			std::cout << "Document = " << docN << " Word = " << doc->words[n] << " topic k = " << k << " phi[n][k] = " << phi[n][k] << " doc->counts[n] " << doc->counts[n] << " ss->z_bar[docN][k] = " << ss->z_bar[docN][k] << " Likelihood " << likelihood << std::endl;

		}
    }
	double gamma_sum = 0.0;
	for (k = 0; k < num_topics; k++)
	{
		gamma_sum += gamma[k];
#pragma omp atomic
		ss->alpha_ss[k] += boost::math::digamma(gamma[k]);
		ss->z_bar[docN][k] /= static_cast<double>(doc->total); //document specific so doesn't need to be atomic
	}
	for (k = 0; k < num_topics; k++)
	{
#pragma omp atomic
		ss->alpha_ss[k] -= boost::math::digamma(gamma_sum);
	}
    return (likelihood);
}

double sslda::lda_inference(document* doc, double* var_gamma, double** phi, double* oldphi, double* dig, const double var_converged, const int var_max_iter)
{
    int k, n, var_iter;
    double converged = 1.0, phisum = 0.0, likelihood = 0.0, likelihood_old = 0.0;

    // compute posterior dirichlet
    for (k = 0; k < num_topics ; k++)
    {
		var_gamma[k] = alpha[k] + (static_cast<double>(doc->total) / (static_cast<double>(num_topics)));
        dig[k] = boost::math::digamma(var_gamma[k]);
        for (n = 0; n < doc->length; n++)
			phi[n][k] = 1.0 / (static_cast<double>(num_topics) );
    }
	
    var_iter = 0;

    while (converged > var_converged && (var_iter < var_max_iter || var_max_iter == -1))
    {
        var_iter++;
        for (n = 0; n < doc->length; n++)
        {
            phisum = 0;
            for (k = 0; k < num_topics; k++)
            {
                oldphi[k] = phi[n][k];
                phi[n][k] = dig[k] + ddelta[k][doc->words[n]]; //expectation of word given theta|gamma distribution and Z|phi distribution
                if (k > 0)
                    phisum = log_sum(phisum, phi[n][k]);
                else
                    phisum = phi[n][k]; // note, phi is in log space
            }

            for (k = 0; k < num_topics; k++)
            {
                phi[n][k] = exp(phi[n][k] - phisum); //normalise and exponentiate
				var_gamma[k] = var_gamma[k] + doc->counts[n] * (phi[n][k] - oldphi[k]);  //update var_gamma
                dig[k] = boost::math::digamma(var_gamma[k]);
            }
        }

        likelihood = lda_compute_likelihood(doc, phi, var_gamma, dig);
        converged = fabs((likelihood_old - likelihood) / likelihood_old);
        likelihood_old = likelihood;
    }
    return likelihood;
}

double sslda::lda_compute_likelihood(document* doc, double** phi, double* var_gamma, double* dig)
{
	double likelihood = 0, digsum = 0, var_gamma_sum = 0;

	int k, n;
	double alpha_sum{0};

	for (k = 0; k < num_topics; k++)
	{
		alpha_sum += alpha[k];
		dig[k] = boost::math::digamma(var_gamma[k]);
		var_gamma_sum += var_gamma[k];
	}

    digsum = boost::math::digamma(var_gamma_sum);

    likelihood = boost::math::lgamma(alpha_sum) - boost::math::lgamma(var_gamma_sum); // A5 and A8
	
    for (k = 0; k < num_topics; k++)
    {
        likelihood += - boost::math::lgamma(alpha[k]) + (alpha[k] - 1.0)*(dig[k] - digsum) +
                      boost::math::lgamma(var_gamma[k]) - (var_gamma[k] - 1.0)*(dig[k] - digsum); // A5 and A8
	
	
        for (n = 0; n < doc->length; n++)
        {
			if (phi[n][k] > 0)
            {
				likelihood += doc->counts[n] *(phi[n][k] * ((dig[k] - digsum) -
					log(phi[n][k]) + ddelta[k][doc->words[n]])); ///combines components from A6, A7 and A8 all multiplied by phi[n][k]

			}
        }

    }
    return likelihood;
}
/*
double sslda::sslda_inference(document* doc, double* var_gamma, double** phi, double* oldphi, double* dig, double* cbhz_params, const double var_converged, const int var_max_iter)
{
	int k, n, var_iter;
	double converged = 1.0, phisum = 0.0, likelihood = 0.0, likelihood_old = 0.0;
	double cbhz_prod = 1.0, temp = 0.0;
	double cbz = cumbasehaz[doc->t_exit - time_start];
	double cbeta = 0.0;

	// compute posterior dirichlet
	for (k = 0; k < num_topics; k++)
	{
		var_gamma[k] = alpha[k] + (static_cast<double>(doc->total) / static_cast<double>(num_topics));
		dig[k] = boost::math::digamma(var_gamma[k]);
		for (n = 0; n < doc->length; n++)
			phi[n][k] = 1.0 / (static_cast<double>(num_topics));
	}

	//update phi and gamma
	var_iter = 0;


	if (covariatesN > 0)
	{
		for (n = 0; n < covariatesN; n++)
			cbeta += cov_beta[n] * static_cast<double>(doc->covariates[n]);
		cbz *= exp(cbeta);

	}
	cbhz_prod = 1.0;
	for (n = 0; n < doc->length; n++)
	{
		temp = 0.0;
		for (k = 0; k < num_topics; k++)
			temp += phi[n][k] * exp(topic_beta[k] * doc->counts[n] / static_cast<double>(doc->total));
		cbhz_prod *= temp;
	}

	while (converged > var_converged && (var_iter < var_max_iter || var_max_iter == -1))
	{
		var_iter++;
		for (n = 0; n < doc->length; n++)
		{
			phisum = 0.0;
			temp = 0.0;
			for (k = 0; k < num_topics; k++)
			{
				cbhz_params[k] = exp(topic_beta[k] * doc->counts[n] / static_cast<double>(doc->total)); // Only exponentiate once
				temp += phi[n][k] * cbhz_params[k];
				oldphi[k] = phi[n][k];
			}
			cbhz_prod /= temp; //remove the contribution of word n

			for (k = 0; k < num_topics; k++)
			{
				phi[n][k] =
					dig[k] + ddelta[k][doc->words[n]] //LDA update. digamma(sum(var_gamma)) is a constant across the topic distribution for each word so is dropped from the update
					+ (static_cast<double>(doc->label) * topic_beta[k] * doc->counts[n] / static_cast<double>(doc->total)) //contribution to Cox model of an event
					- (cbz * cbhz_params[k] * cbhz_prod); // update phi given gamma note, phi is in log space
				//In slda - sum_no.n(phi*exp(beta*counts))*n.exp(beta*counts)/sum_no(phi*exp(beta*counts))
				if (k > 0)
					phisum = log_sum(phisum, phi[n][k]);
				else
					phisum = phi[n][k]; // note, phi is in log space
				//std::cout << " phisum = " << phisum << std::endl;
			}

			for (k = 0; k < num_topics; k++)
			{
				phi[n][k] = exp(phi[n][k] - phisum); //normalise phi into exp space 
				var_gamma[k] = var_gamma[k] + (doc->counts[n] * (phi[n][k] - oldphi[k]));  //Update the gamma given phi (document topic distribution updated by new word allocations)
				dig[k] = boost::math::digamma(var_gamma[k]);
				//	std::cout << "topic k results = " << k << std::endl;
				//	std::cout << "phi = " << phi[n][k] << " var_gamma[k] " << var_gamma[k] << " dig[k] " << dig[k] << std::endl;
			}

			temp = 0.0;
			for (k = 0; k < num_topics; k++)
				temp += phi[n][k] * cbhz_params[k];

			cbhz_prod *= temp; //multiply back in the updated contribution for word n

		}

		likelihood = sslda_compute_likelihood(doc, phi, var_gamma, dig);
		converged = fabs((likelihood_old - likelihood) / likelihood_old);
		likelihood_old = likelihood;
		//std::cout << "likelihood = "<< likelihood << std::endl;
	}

	return likelihood;
}*/

double sslda::sslda_inference(document* doc, double* var_gamma, double** phi, double* oldphi, double* dig, double* cbhz_params, const double var_converged, const int var_max_iter)
{
	int k, n, var_iter;
	double converged = 1.0, phisum = 0.0, likelihood = 0.0, likelihood_old = 0.0;
	double cbhz_prod = 1.0, temp = 0.0;

    // compute posterior dirichlet
    for (k = 0; k < num_topics ; k++)
	{
		var_gamma[k] = alpha[k] + (static_cast<double>(doc->total) / static_cast<double>(num_topics));
		dig[k] = boost::math::digamma(var_gamma[k]);
        for (n = 0; n < doc->length; n++)
			phi[n][k] = 1.0 / (static_cast<double>(num_topics) );
    }

	//update phi and gamma
    var_iter = 0;


	cbhz_prod = 1.0;
	for (n = 0; n < doc->length; n ++)
	{
		temp = 0.0;
		for (k = 0; k < num_topics; k++)
			temp += phi[n][k] * exp(topic_beta[k] * doc->counts[n] / static_cast<double>(doc->total));
		cbhz_prod *= temp;
	}  

    while (converged > var_converged && (var_iter < var_max_iter || var_max_iter == -1))
    {
        var_iter++;
		for (n = 0; n < doc->length; n++)
		{		
			phisum = 0.0;
			temp = 0.0;
			for (k = 0; k < num_topics; k++)
			{
				cbhz_params[k] = exp(topic_beta[k] * doc->counts[n] / static_cast<double>(doc->total)); // Only exponentiate once
				temp += phi[n][k] * cbhz_params[k];
				oldphi[k] = phi[n][k];
			}
			cbhz_prod /= temp; //remove the contribution of word n

			for (k = 0; k < num_topics; k++)
			{
				phi[n][k] =
					dig[k] + ddelta[k][doc->words[n]] //LDA update. digamma(sum(var_gamma)) is a constant across the topic distribution for each word so is dropped from the update
					+ (static_cast<double>(doc->label) * topic_beta[k] * doc->counts[n] / static_cast<double>(doc->total)) //contribution to Cox model of an event
					- (cbhz_params[k] * cbhz_prod); // update phi given gamma note, phi is in log space
		//		std::cout << "phi = " << phi[n][k] << std::endl;

				//In slda - sum_no.n(phi*exp(beta*counts))*n.exp(beta*counts)/sum_no(phi*exp(beta*counts))
				if (k > 0)
					phisum = log_sum(phisum, phi[n][k]);
				else
					phisum = phi[n][k]; // note, phi is in log space
			}
		//	std::cout << " phisum = " << phisum << std::endl;
			for (k = 0; k < num_topics;k++)
			{
				phi[n][k] = exp(phi[n][k] - phisum); //normalise phi into exp space 
				var_gamma[k] = var_gamma[k] + (doc->counts[n] * (phi[n][k] - oldphi[k]));  //Update the gamma given phi (document topic distribution updated by new word allocations)
				dig[k] = boost::math::digamma(var_gamma[k]);
		//		std::cout << "topic k results = " << k << std::endl;
		//		std::cout << "phi = " << phi[n][k] << " var_gamma[k] " << var_gamma[k] << " dig[k] " << dig[k] << std::endl;
			}
			
			temp = 0.0;
			for (k = 0; k < num_topics; k++)
				temp += phi[n][k] * cbhz_params[k];

			cbhz_prod *= temp; //multiply back in the updated contribution for word n

		}
		
        likelihood = sslda_compute_likelihood(doc, phi, var_gamma, dig);
		converged =  fabs((likelihood_old - likelihood) / likelihood_old);
        likelihood_old = likelihood;
		//std::cout << "likelihood = "<< likelihood << std::endl;
	}

    return likelihood;
}

double sslda::sslda_compute_likelihood(document* doc, double** phi, double* var_gamma, double* dig)
{
    double likelihood = 0.0, digsum = 0.0, var_gamma_sum = 0.0, temp = 0.0;
	double cbeta = 0.0, alpha_sum = 0.0;
    int k, n;
	
    for (k = 0; k < num_topics; k++)
    {
        dig[k] = boost::math::digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
		alpha_sum += alpha[k];
    }
    digsum = boost::math::digamma(var_gamma_sum);

	likelihood =  boost::math::lgamma(alpha_sum) - boost::math::lgamma(var_gamma_sum); // A5 and A8

	temp = 0.0;
	for (k = 0; k < num_topics; k++)
	{
		likelihood += -boost::math::lgamma(alpha[k]) + ((alpha[k] - 1.0) * (dig[k] - digsum)) + boost::math::lgamma(var_gamma[k]) - ((var_gamma[k] - 1.0) * (dig[k] - digsum));

		for (n = 0; n < doc->length; n++)
		{
			if (phi[n][k] > 0.0)
			{
				likelihood += doc->counts[n] * (phi[n][k] * ((dig[k] - digsum) - log(phi[n][k]) + ddelta[k][doc->words[n]]));
				if (doc->label > 0)
					temp += topic_beta[k] * doc->counts[n] * phi[n][k];

			}
		}
	}
	likelihood += temp / static_cast<double>(doc->total); 	//eta_k*\bar{\phi}

	//E[logp( T, d | Z, beta, cumbasehaz)]

	cbeta = 0.0;
	for (n = 0; n < covariatesN; n++)
		cbeta += cov_beta[n] * static_cast<double>(doc->covariates[n]);

	double cbz = cumbasehaz[doc->t_exit- time_start];
	cbz *= exp(cbeta);
	if (doc->label > 0)
		likelihood += log(basehaz[doc->t_exit- time_start]) + cbeta;

	for (n = 0; n < doc->length; n++)
	{
		temp = 0.0;
		for (k = 0; k < num_topics; k++)
			temp += phi[n][k]  * exp(topic_beta[k] * doc->counts[n] / static_cast<double>(doc->total));
		cbz *= temp;
	}
	likelihood -= cbz;

    return likelihood;
}



double sslda::infer_only(const corpus * c, const settings * setting, double *perplexity, double *loglik ,  const char * directory, int save)
{

	int d, k;
	double lik = 0.0; 
	double ** var_gamma = new double *[c->num_docs];
	const int maxlength = c->max_length;
	for (d = 0; d < c->num_docs; d++)
	{
		var_gamma[d] = new double[num_topics];
		for (k = 0; k < num_topics; k++)
			var_gamma[d][k] = 0.0;
	}


	double ** z_bar = new double *[c->num_docs];
	for (d = 0; d < c->num_docs; d++)
	{
		z_bar[d] = new double[num_topics];
		for (k = 0; k < num_topics; k++)
			z_bar[d][k] = 0.0;
	}

	double * docscore = new double [c->num_docs];
	for (d = 0; d < c->num_docs; d++)
		docscore[d] = 0.0;
	std::cout << std::endl << "Coefficients for covariates :" << std::endl;
	for (k = 0; k < covariatesN; k++)
		std::cout << cov_beta[k] << ", ";
	std::cout << std::endl;
	std::cout << "Coefficients for topics:" << std::endl;
	for (k = 0; k < num_topics; k++)
		std::cout <<topic_beta[k] << ", ";
	std::cout<< std::endl;
	std::cout << std::endl << " ******** Inference ******** " << std::endl;
	int var_max_iter = setting->VAR_MAX_ITER;
	double var_converged = setting->VAR_CONVERGED;

#pragma omp parallel reduction(+:lik) default(none) shared(c, z_bar, var_gamma, docscore, var_converged, var_max_iter)
	{ 
		int kl, nl, docN; //iterators local to thread
		int size=omp_get_num_threads(); // get total number of processes
		int rank=omp_get_thread_num(); // get rank of current
			///variables for working in estimation step, allocated here so done once per thread
		double *oldphi = new double [num_topics];
		for (kl = 0; kl < num_topics; kl++)
			oldphi[kl] = 0.0;

		double *dig = new double [num_topics];
		for (kl = 0; kl < num_topics; kl++)
			dig[kl] = 0.0;
			
		double ** phi = new double *[maxlength];
		for (nl = 0; nl < maxlength; nl++)
		{
			phi[nl] = new double [num_topics];
			for (kl = 0; kl < num_topics; kl++)
				phi[nl][kl] = 0.0;
		}

		for(docN = (rank*c->num_docs/size); docN<(rank+1)*c->num_docs/size; docN++)
		{
			lik += lda_inference(c->docs[docN], var_gamma[docN], phi, oldphi, dig, var_converged, var_max_iter);

			double score = 0.0;
			
			
			for (kl = 0; kl < num_topics; kl++)
			{
				for (nl = 0; nl < c->docs[docN]->length && nl < maxlength; nl++)
					z_bar[docN][kl] += static_cast<double>(c->docs[docN]->counts[nl]) * phi[nl][kl];
				z_bar[docN][kl] /= static_cast<double>(c->docs[docN]->total);
				score += z_bar[docN][kl] * topic_beta[kl];
			}
			for (kl = 0; kl < covariatesN; kl++)
				score += cov_beta[kl] * c->docs[docN]->covariates[kl];
			docscore[docN] = exp(score);	
		}
		for (nl = 0; nl < maxlength; nl++)
			delete [] phi[nl] ;
		delete [] phi;
		phi = nullptr;
		delete [] dig;
		dig = nullptr;
		delete [] oldphi;
		oldphi = nullptr;
	}
	*loglik = lik+ldelta;
	double num = 0.0, den = 0.0;

	for (d = 0; d < c->num_docs; d++)
	{
		double score = 0.0;
		for (k = 0; k < num_topics; k++)
			score += z_bar[d][k] * topic_beta[k];
		for (k = 0; k < covariatesN; k++)
			score += cov_beta[k] * c->docs[d]->covariates[k];
		docscore[d] = exp(score);
	}


	int * mark = new int[c->num_docs];
	for (d = 0; d < c->num_docs; d++)
		mark[d] = 0;
	int count = 1;
	mark[c->num_docs - 1] = 1;
	for (d = c->num_docs - 2; d >= 0; d--)
	{
		if (c->docs[d]->t_exit== c->docs[d + 1]->t_exit) //assume sorted by time, so calculate the number needed to jump to next increment in time
			count++;
		else
			count = 1;
		mark[d] = count;
	}

#pragma omp parallel reduction(+:num,den) /*default(none)*/ shared(docscore, c, mark)
	{
		int dl, ddl;
		int size=omp_get_num_threads(); // get total number of processes
		int rank=omp_get_thread_num(); // get rank of current
		
		for(dl = (rank*c->num_docs/size); dl<(rank+1)*c->num_docs/size; dl++)
		{
			if (c->docs[dl]->label>0)
			{
				for (ddl = dl + mark[dl] ; ddl < c->num_docs ; ddl++)
				{
					if (c->docs[dl]->t_exit >= c->docs[ddl]->t_enter)
					{
						den += 1.0;
						if (docscore[dl] > docscore[ddl])
							num += 1.0;
						else if (docscore[dl] == docscore[ddl])
							num += 0.5;
					}

				}
			}
		}
	}
	delete[] mark;
	mark = nullptr;

	delete[] docscore;
	docscore = nullptr;
	*perplexity = exp(-(*loglik) / c->num_total_words);
	
	for (k = 0; k < num_topics; k++) std::cout << alpha[k] << ", ";
	std::cout << std::endl;
	std::cout << "Num of topics = " << num_topics
		<< ", Lambda = " << lambda << std::endl
		<< "Overall inference likelihood : " << *loglik << std::endl
		<< "Perplexity: " << *perplexity << std::endl
		<< "C statistic : " << static_cast<double>(num) / static_cast<double>(den) << std::endl;

	if (save == 1)
	{
		stringstream filename;
		filename << directory << "//inf-likelihood.dat";
		std::cout << "inf_likelihood will be saved in " << directory << std::endl;
		ofstream likelihood_file(filename.str().c_str());
		if (likelihood_file.is_open())
		{
			likelihood_file << "Alpha = ";
			for (k = 0; k < num_topics; k++) likelihood_file << alpha[k] << ", ";
			likelihood_file << "Num of topics = " << num_topics
				<< ", Lambda = " << lambda << std::endl
				<< "Overall inference likelihood : " << *loglik << std::endl
				<< "C statistic : " << static_cast<double>(num) / static_cast<double>(den) << std::endl
				<< "Perplexity: " << *perplexity << std::endl;
			likelihood_file.close();
		}
		else
			std::cerr << "unable to open " << filename.str() << std::endl;


//		stringstream filename_g;
//		filename_g << directory << "//inf-gamma.dat";
//		save_gamma(filename_g.str().c_str(), var_gamma, c->num_docs);
//		std::cout << "Topic allocation gamma saved in " << filename_g.str() << std::endl;

		stringstream filename_z;
		filename_z << directory << "//inf-zbar.dat";
		save_zbar(filename_z.str().c_str(), z_bar, c->num_docs);
		std::cout << "topic probability z_bar saved in " << filename_z.str() << std::endl;
	}
	for (d = 0; d < c->num_docs; d++)
		delete[] var_gamma[d];
	delete [] var_gamma;
	var_gamma = nullptr;
	for (d = 0; d < c->num_docs; d++)
		delete[] z_bar[d];
	delete [] z_bar;
	z_bar = nullptr;
	return static_cast<double>(num) / static_cast<double>(den);
}

void sslda::save_gamma(const char* filename, double** gamma, int num_docs)
{
    int d, k;
	ofstream fileptr(filename);
	if (fileptr.is_open())
	{
		for (d = 0; d < num_docs; d++)
		{
			fileptr << gamma[d][0];
			for (k = 1; k < num_topics; k++)
				fileptr << ", "<< gamma[d][k];
			fileptr << std::endl;
		}
		fileptr.close();
	}
	else
		std::cerr << "unable to open " << filename << std::endl;
}

void sslda::save_zbar(const char* filename, double ** z_bar, int num_docs)
{
	int d, k;
	ofstream fileptr(filename);
	if (fileptr.is_open())
	{
		for (d = 0; d < num_docs; d++)
		{
			fileptr << z_bar[d][0];
			for (k = 1; k < num_topics; k++)
				fileptr << ", "<< z_bar[d][k];
			fileptr << std::endl;
		}
		fileptr.close();
	}
	else
		std::cerr << "unable to open " << filename << std::endl;
}



