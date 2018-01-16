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

//survLDA est code_wide settings 0.3 100 seeded results >& log.txt

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



#include <cstddef> //NULL
#include <iostream>
#include <omp.h> //for multi threading

#include "corpus.h"
#include "utils.h"
#include "sslda.h"

void help( void ) {
    std::cout <<"Usage: survlda [est] [settings] [data] [adj] [alpha] [k]  [random/seeded/model_path] [directory]" << std::endl;
	std::cout << "Usage: survlda [cox] [settings] [data] [adj] [directory]" << std::endl;
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
	//	corpus c_val;
	//	string valdata_filename = argv[8 + a];
	//	if (c_val.read_data(valdata_filename.c_str(), &setting) != 1)
	//		return -1;
	//	if (setting.ADJN > 0)
	//	{
	//		string adjval_filename = argv[9+a];
	//		if (c_val.read_adj(adjval_filename.c_str(), &setting) != 1)
	//			return -1;
	//		a ++;
	//	}

		int test = model.v_em(&c, &setting, init_method.c_str(), directory.c_str() );
		if(test !=1)
			std::cerr << "Model returned " << test << std::endl;
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
		cstat = NULL;
		perplexity = NULL;
		loglik = NULL;
	}

}
