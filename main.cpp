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

//MSVS debugging
# if defined(_MSC_VER)
#  define isn(x) (_isnan(x))
#  define isf(x) (_finite(x))
#  define mkd(x) (_mkdir(x))
#  include <direct.h> //windows header for _mkdir
#  include <vld.h> // for debugging
# else
#  define isn(x) (isnan(x))
#  define isf(x) (isfinite(x))
#  define mkd(x) (mkdir(x))
# include <sys/stat.h> // linux header for mkdir
#endif

#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>

#include <cstddef> //NULL
#include <iostream>
#include <omp.h> //for multi threading

#include "corpus.h"
#include "utils.h"
#include "sslda.h"

void help( void ) {
    std::cout <<"Usage: survlda [est] [settings] [data] [adj] [alpha] [k]  [random/seeded/model_path] [directory]" << std::endl;
	std::cout << "survlda [bootstrap] [settings] [data] [adj] [alpha] [k] [iterations]  [random/seeded/model_path] [directory]" << std::endl;
	std::cout << " survlda [cox] [settings] [data] [adj] [directory]" << std::endl;
	std::cout << "       survlda [inf] [settings] [data] [adj] [model] [directory]" << std::endl;
	std::cout << "	   survlda [range] [settings] [data] [adj] [val data] [val adj] " << std::endl;
	std::cout << "				 [k min] [k max] [k step]  [random/seeded/model_path] [directory]" << std::endl;
}

int main(int argc, char* argv[])
{

    if (argc < 2 )
    {
        help();
        return 0;
    }
    if (strcmp(argv[1], "est") == 0)
    {
        settings setting;
        string setting_filename = argv[2];
		if (setting.read_settings(setting_filename.c_str()) != 1)
			return -1;
		
		corpus c;
        string data_filename = argv[3];
		if (c.read_data(data_filename.c_str(), &setting) != 1)
			return -1;
		int a = 0;
		if (setting.ADJN > 0)
		{
			string adj_filename = argv[4];
			if (c.read_adj(adj_filename.c_str(), &setting) !=1 )
				return -1;
			a = 1;
		}

        double  alpha = atof(argv[4 + a]);
        int num_topics = atoi(argv[5 + a]);
        std::cout << "number of topics:  " << num_topics << std::endl;
        string init_method = argv[6 + a];
        string directory = argv[7 + a];
		std::cout << "models will be saved in " << directory << std::endl;
		int check = mkd(directory.c_str()) ;
		if (check != 0 && errno != EEXIST)
		{
			std::cerr << "Unable to make " << directory.c_str() << " " << check << std::endl;
			return -1;
		}

		sslda model;
		model.init(alpha, num_topics, setting.ADJN, &c);

		int test = model.v_em(&c, &setting, init_method.c_str(), directory.c_str() );
		if(test !=1)
			std::cerr << "Model returned " << test << std::endl;
		model.free();
		c.free();
    }
	if (strcmp(argv[1], "bootstrap") == 0)
	{
		settings setting;
		string setting_filename = argv[2];
		if (setting.read_settings(setting_filename.c_str()) != 1)
			return -1;

		corpus c;
		string data_filename = argv[3];
		if (c.read_data(data_filename.c_str(), &setting) != 1)
			return -1;
		int a = 0;
		if (setting.ADJN > 0)
		{
			string adj_filename = argv[4];
			if (c.read_adj(adj_filename.c_str(), &setting) != 1)
				return -1;
			a = 1;
		}

		double  alpha = atof(argv[4 + a]);
		int num_topics = atoi(argv[5 + a]);
		int iterations = atoi(argv[6 + a]);
		std::cout << "number of topics:  " << num_topics << std::endl;
		string init_method = argv[7 + a];
		string directory = argv[8 + a];
		std::cout << "models will be saved in " << directory << std::endl;
		int check = mkd(directory.c_str());
		if (check != 0 && errno != EEXIST)
		{
			std::cerr << "Unable to make " << directory.c_str() << " " << check << std::endl;
			return -1;
		}
		
		double * cstat = new double[iterations];
		double * perplexity = new double[iterations];
		double * loglik = new double[iterations];
		for (int i = 0; i < iterations; i++)
		{
			cstat[i] = 0.0;
			perplexity[i] = 0.0;
			loglik[i] = 0.0;
		}
		for (int i = 0; i < iterations; i++)
		{
			string iterdirectory = directory + "\\Iteration" + std::to_string(i);
			int check2 = mkd(iterdirectory.c_str());
			if (check2 != 0 && errno != EEXIST)
			{
				std::cerr << "Unable to make " << iterdirectory.c_str() << " " << check2 << std::endl;
				return -1;
			}
			
			corpus s;
			corpus s_val;
			c.sample(&s, &s_val, &setting);
			std::cout << s.num_docs << " documents in sample for training in iteration " << i << " out of " << iterations << " iterations" << std::endl;
			std::cout << s_val.num_docs << " documents in sample for testing in iteration " << i  << " out of " << iterations << " iterations" << std::endl;
			sslda model;
			model.init(alpha, num_topics, setting.ADJN, &s);

			int test = model.v_em(&s, &setting, init_method.c_str(), iterdirectory.c_str());
			if (test != 1)
				std::cerr << "Model returned " << test << std::endl;
			model.free_model();
			s.free();
			stringstream val_model_filename;
			val_model_filename << iterdirectory.c_str() << "\\final.model";
			sslda model_val;
			model_val.load_model(val_model_filename.str().c_str());
			std::cout << val_model_filename.str() << " reloaded for validation " << std::endl;
			cstat[i] = model_val.infer_only(&s_val, &setting, &perplexity[i], &loglik[i], iterdirectory.c_str(), 0);
			model_val.free_model();
			s_val.free();
		}


		std::cout << " ...results in file " ;
		stringstream filename_r;
		filename_r << directory << "\\results.csv";
		ofstream rfile(filename_r.str().c_str());
		if (!rfile.is_open())
		{
			std::cout << "Unable to create results file, printing to screen only\n" << std::endl;
			std::cout << std::endl << std::endl << "C statistics : " << std::endl;
			for (int i = 0; i < iterations; i++)
				std::cout << cstat[i] << ", ";
			std::cout << std::endl;

			std::cout << std::endl << std::endl << "Perplexity : " << std::endl;
			for (int i = 0; i < iterations; i++)
				std::cout << perplexity[i] << ", ";
			std::cout << std::endl;


			std::cout << std::endl << std::endl << "Log likelihood : " << std::endl;
			for (int i = 0; i < iterations; i++)
				std::cout << loglik[i] << ", ";
			std::cout << std::endl;

			delete[] loglik;
			delete[] perplexity;
			delete[] cstat;
			loglik = nullptr;
			perplexity = nullptr;
			cstat = nullptr;
			return 0;

		}
		rfile << "C statistic , Perplexity, Log likelihood " << std::endl;;
		for (int i = 0; i < iterations; i++)
		{
			rfile << cstat[i] << ", " << perplexity[i] << "," << loglik[i] << std::endl;
		}

		rfile.close();
		delete[] cstat;
		delete[] perplexity;
		delete[] loglik;
		cstat = nullptr;
		perplexity = nullptr;
		loglik = nullptr;
	}
	if (strcmp(argv[1], "cox") == 0)
	{
		settings setting;
		string setting_filename = argv[2];
		if (setting.read_settings(setting_filename.c_str()) != 1)
			return -1;

		corpus c;
		string data_filename = argv[3];
		if (c.read_data(data_filename.c_str(), &setting) != 1)
			return -1;
		int a = 0;
		if (setting.ADJN > 0)
		{
			string adj_filename = argv[4];
			if (c.read_adj(adj_filename.c_str(), &setting) != 1)
				return -1;
			a = 1;
		}
		string directory = argv[4 + a];
		std::cout << "models will be saved in " << directory << std::endl;
		int check = mkd(directory.c_str());
		if (check != 0 && errno != EEXIST)
		{
			std::cerr << "Unable to make " << directory.c_str() << " " << check << std::endl;
			return -1;
		}

		sslda model;
		model.init(0, c.size_vocab, setting.ADJN, &c);
		double test = model.coxonly(&c, &setting);
		if (test != 1)
			std::cerr << "Model returned " << test << std::endl;
		model.free();
		c.free();
	}
    else if (strcmp(argv[1], "inf") == 0)
    {
		settings setting;
        string setting_filename = argv[2];
		if (setting.read_settings(setting_filename.c_str()) != 1)
			return -1;
		corpus c;
        string data_filename = argv[3];
        if (c.read_data(data_filename.c_str(), &setting) != 1)
			return -1;
		int a = 0;
		if (setting.ADJN > 0)
		{
			string adj_filename = argv[4];
			if (c.read_adj(adj_filename.c_str(), &setting) != 1)
				return -1;
			a = 1;
		}
        string model_filename = argv[4 + a];
        string directory = argv[5 + a];
		
		std::cout << std::endl << "Results will be saved in " << directory << std::endl;
		int check = mkd(directory.c_str()) ;
		if (check != 0 && errno != EEXIST)
		{
			std::cerr << "Unable to make " << directory.c_str() << std::endl;
			return -1;
		}
		sslda model;
		double perplexity, loglik;
        model.load_model(model_filename.c_str());
		std::cout << std::endl << "Model file found in " << model_filename << std::endl;
		std::cout << "Number of topics is " << model.num_topics << std::endl;
		std::cout << "Lambda = " << model.lambda << std::endl;
		double Cstat = model.infer_only(&c, &setting, &perplexity, &loglik , directory.c_str() ,1 );
		if (isn(Cstat))
			std::cerr << "Error in calculating C statistic" << std::endl;
		model.free();
		c.free();
	 }
	else if (strcmp(argv[1], "range") == 0)
	{
		settings setting;
		string setting_filename = argv[2];
		if (setting.read_settings(setting_filename.c_str()) != 1)
			return -1;

		corpus c;
		string data_filename = argv[3];
		if (c.read_data(data_filename.c_str(), &setting) != 1)
			return -1;
		int a = 0;
		if (setting.ADJN > 0)
		{
			string adj_filename = argv[4];
			if (c.read_adj(adj_filename.c_str(), &setting) != 1)
				return -1;
			a = 1;
		}

		corpus c_val;
		string data_val_filename = argv[4 + a];
		if (c_val.read_data(data_val_filename.c_str(), &setting) != 1)
		{
			return -1;
		}
		if (setting.ADJN > 0)
		{
			string adj_val_filename = argv[5 + a];
			if (c_val.read_adj(adj_val_filename.c_str(), &setting) != 1)
				return -1;
			a = 2;
		}
		int min_topics = atoi(argv[5 + a]);
		int max_topics = atoi(argv[6 + a]);
		int step_topic = atoi(argv[7 + a]);
		int ntopic = static_cast<int>(ceil((max_topics - min_topics) / step_topic));
		std::cout << "Range of number of topics = " << min_topics << " - " << max_topics << " in " << ntopic << " steps of " << step_topic << std::endl;
		string init_method = argv[8 + a];
		string directory = argv[9 + a];
		std::cout << "Model files will be saved in " << directory << std::endl;
		int check = mkd(directory.c_str());
		if (check != 0 && errno != EEXIST)
		{
			std::cerr << "Unable to make " << directory.c_str() << std::endl;
			return -1;
		}
		double * cstat = new double [ntopic];
		double * perplexity = new double [ntopic];
		double * loglik = new double [ntopic];
		for (int i = 0; i < ntopic; i++)
		{

			int test = 0;

			stringstream subdirectory;
			subdirectory << directory << "\\topic_" << min_topics + (step_topic *i);
			check = mkd(subdirectory.str().c_str());
			if (check != 0 && errno != EEXIST)
			{
				std::cerr << "Unable to make " << subdirectory.str().c_str() << std::endl;
				return -1;
			}
			{
				sslda model;
				model.init(1.0 / (static_cast<double>(min_topics)+static_cast<double>(step_topic * i)), min_topics + (step_topic * i), setting.ADJN, &c);
				test = model.v_em(&c, &setting, init_method.c_str(), subdirectory.str().c_str() /*,&c_val*/ );
				model.free_model();
			}
			if (test < 0)
			{
				std::cout << "Model failed to converge error = " << test << std::endl;
				cstat[i] = test;
				perplexity[i] = test;
				loglik[i] = test;
			}
			else
			{
				stringstream val_model_filename;
				val_model_filename << subdirectory.str() << "\\final.model";
				sslda model_val;
				model_val.load_model(val_model_filename.str().c_str());
				std::cout << val_model_filename.str() << " reloaded for validation on " << data_val_filename << std::endl;
				cstat[i] = model_val.infer_only(&c_val, &setting, &perplexity[i], &loglik[i], subdirectory.str().c_str() , 1);
				model_val.free_model();
			}
			
		}
		c.free();

		std::cout << " ...results file file" << std::endl;
		stringstream filename_r;
		filename_r << directory << "\\results.csv";
		ofstream rfile(filename_r.str().c_str());
		if (!rfile.is_open())
		{
			std::cout << "Unable to create results file, printing to screen only\n" << std::endl;
			std::cout << std::endl << std::endl << "C statistics : " << std::endl;
			for (int i = 0; i < ntopic; i++)
				std::cout << cstat[i] << ", ";
			std::cout << std::endl;

			std::cout << std::endl << std::endl << "Perplexity : " << std::endl;
			for (int i = 0; i < ntopic; i++)
				std::cout << perplexity[i] << ", ";
			std::cout << std::endl;


			std::cout << std::endl << std::endl << "Log likelihood : " << std::endl;
			for (int i = 0; i < ntopic; i++)
				std::cout << loglik[i] << ", ";
			std::cout << std::endl;

			delete[] loglik;
			delete[] perplexity;
			delete[] cstat;
			return 0;

		}
		rfile << "C statistic : ";
		for (int i = 0; i < ntopic; i++)
			rfile << min_topics + (step_topic * i) << ", "  << cstat[i]  << std::endl;
		
		rfile << std::endl << std::endl << "Perplexity : " << std::endl;
		for (int i = 0; i < ntopic; i++)
			rfile << min_topics + (step_topic * i) << ", " << perplexity[i] << ", " << std::endl;
		
		rfile << std::endl << std::endl << "Log likelihood :"  << ", " << std::endl;
		for (int i = 0; i < ntopic; i++)
			rfile << min_topics + (step_topic * i) << ", " << loglik[i] << ", " << std::endl;
		
		rfile.close();
		delete[] cstat;
		delete[] perplexity;
		delete[] loglik;
		cstat = nullptr;
		perplexity = nullptr;
		loglik = nullptr;
	}

}
