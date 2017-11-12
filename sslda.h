// Latent Dirichlet Allocation supervised by penalised Cox proportional hazards modelling with optional learning of asymmetrical priors.

//This has been developed from the original code (C) Copyright 2009, Chong Wang, David Blei and Li Fei-Fei ([1] Blei DM, McAuliffe JD. Supervised Topic Models. Adv Neural Inf Process Syst 20 2007:121–8.) and modified following the algorithms developed by Ye et al. 2014 ([1] Ye S, Dawson JA, Kendziorski C. Extending information retrieval methods to personalized genomic-based studies of disease. Cancer Inform 2014;13:85–95. doi:10.4137/CIN.S16354.)

// Modifications by Colin Crooks (colin.crooks@nottingham.ac.uk)

// This file is part of sslda.

// sslda is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your
// option) any later version.

// sslda is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more cov_betails.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#ifndef SSLDA_H
#define SSLDA_H

# if defined(_MSC_VER)
#  define isn(x) (_isnan(x))
#  define isf(x) (_finite(x))
# else
#  define isn(x) (isnan(x))
#  define isf(x) (isfinite(x))
#endif



#include "settings.h"
#include "corpus.h"

typedef struct {
    int num_docs ;
	int time_start;

	int * group;
	int * labels;
	int * times;
	int * time0;
	int ** mark;
	double ** covariates;

	double* alpha_ss;

	double * word_total_ss;
	double ** word_ss;	
	double ** z_bar;
} suffstats;

class sslda
{
public:
    sslda();
    ~sslda();
    void free_model();
    void init(double alpha_, int num_topics_, int covariatesN_, const corpus * c);
    int save_model(const char * filename);
    int save_model_text(const char * filename);
    int load_model(const char * model_filename);
	void save_gamma(const char* filename, double** gamma, int num_docs);
	void save_zbar(const char* filename, double** z_bar, int num_docs);
	void save_expected_combined_prob(double ** Expectedcombprob);
    
	suffstats * new_suffstats(int num_docs);   
    void free_suffstats(suffstats* ss);
    void zero_initialize_ss(suffstats* ss);
	void random_initialize_ss(suffstats* ss, const corpus * c, const settings* setting);
    void corpus_initialize_ss(suffstats* ss, const corpus * c, const settings* setting);

	int v_em(const corpus* c, const settings * setting, const char* start, const char* directory , const corpus * c_val );
	double mle(suffstats* ss, int beta_update, const settings * setting);
	double coxonly(const corpus* c, const settings * setting);
	double doc_e_step(document* doc, int docN,  double* gamma, double** phi, double* oldphi, double* dig, double* cbhz_params,  suffstats * ss, int beta_update, const double var_converged, const int var_max_iter);
	double lda_inference(document* doc, double* var_gamma, double** phi, double* oldphi, double* dig, const double var_converged, const int var_max_iter);
    double lda_compute_likelihood(document* doc, double** phi, double* var_gamma, double* dig);
	double sslda_inference(document* doc, double* var_gamma, double** phi, double* oldphi, double* dig, double* cbhz_params, const double var_converged, const int var_max_iter);
    double sslda_compute_likelihood(document* doc, double** phi, double* var_gamma, double* dig);
	double infer_only(const corpus * c, const settings * setting, double *AIC, double *loglik, const char * directory, int save);


public:
    double*  alpha; // the parameter for the per document topic dirichlet
	double* eta; //  the paramenter for the per topic word dirichlet
	double lambda;  // L2 penalty for Cox regression
    int num_topics; // number of topics selected by user or from range command
	int covariatesN; // number of covariates that the model is adjusted for
	int offset; // offset from zero word codes start from
	int event_times; // number of time intervals between start and end of study
	int time_start; // earliest time 
    int size_vocab; // number of unique word codes
	int * events = nullptr; //event counts at each time point
	double * topic_logprob = nullptr;  // 
	double * basehaz = nullptr; // baseline hazard 
	double * cumbasehaz = nullptr; // cummulative baseline hazard
	double *topic_beta = nullptr; //beta for the latent topic hazard ratios
	double * cov_beta = nullptr; // beta for the additional covariates adjusted for
	double ** log_prob_w = nullptr; //Per Term topic probability the log of the topic distribution	
	double ** delta = nullptr; // posterior for the word topic dirichlet 
	double ** ddelta = nullptr; // expectation of the word topic dirichlet
	double  ldelta; // log likelihood component from the word topic dirichlet 
};

#endif // SSLDA_H

