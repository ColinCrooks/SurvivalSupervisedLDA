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


#ifndef CORPUS_H
#define CORPUS_H

# if defined(_MSC_VER)
	#  define isn(x) (_isnan(x))
	#  define isf(x) (_finite(x))
# else
	#  define isn(x) (isnan(x))
	#  define isf(x) (isfinite(x))
#endif


#include <vector>
#include <cstddef>

#include "settings.h"


using namespace std;

class document
{
public:
    int * words;
    double * counts;
	double * covariates;
    int length;
    double total;
    int label;
	int t_exit;
	int t_enter;
public:
    document()
    {
        words = nullptr;
        counts = nullptr;
		covariates = nullptr;
        length = 0;
        total = 0;
        label = 0;
		t_exit = 0;
		t_enter = 0;
	}
    document(int len)
    {
        length = len;
        words = new int [length];
		memset(words, 0, sizeof(int)*length);
		counts = new double [length];
		for (int n = 0; n < length; n++) counts[n] = 0.0;
        total = 0;
        label = 0;
		t_exit = 0;
		t_enter = 0;
    }
	~document()
	{
		free();
	}
	void free()
	{
		if (words != nullptr)
		{
			delete[] words;
			words = nullptr;
		}
		if (counts != nullptr)
		{
			delete[] counts;
			counts = nullptr;
		}
		if (covariates != nullptr)
		{
			delete[] covariates;
			covariates = nullptr;
		}

        length = 0;
        total = 0;
        label = 0;
		t_exit = 0;
		t_enter = 0;
    }
};

class corpus
{
public:
    corpus();
    ~corpus();
	void free();
    int read_data(const char * data_filename, const settings * setting);
	int read_adj(const char * data_filename, const settings * setting);
	int sample(corpus* s, corpus* s_val, const settings * setting);

public:
    int num_docs;
    int size_vocab;
    int num_classes;
    double num_total_words;
    vector<document*> docs;
	int event_times;
	int time_start;
	int time_end;
	int max_length;
};

#endif // CORPUS_H
