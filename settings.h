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
#ifndef SETTINGS_H
#define SETTINGS_H

# if defined(_MSC_VER)
#  define isn(x) (_isnan(x))
#  define isf(x) (_finite(x))
# else
#  define isn(x) (isnan(x))
#  define isf(x) (isfinite(x))
#endif




#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/lexical_cast.hpp>

struct settings
{
	int	NUM_INIT;
	int LAG;
	int LDA_INIT_MAX;
    int   VAR_MAX_ITER;
	double VAR_CONVERGED;
	int MSTEP_MAX_ITER;
	double MAX_EPS;   
	int   EM_MAX_ITER;
    double EM_CONVERGED;
	int OFFSET;
	int ADJN;
	int LAMBDASTART;
	int LAMBDAEND;
	int CROSSVAL;
	int ALPHA;
	int ETA;
	double WT;

    int read_settings(const char* filename)
    {
			std::string filenameS(filename);
			std::string line;
			std::ifstream dataS(filenameS);
			if(!dataS.is_open())
			{
				std::cerr<< "Unable to open settings file" <<std::endl;
				return -1;
			}
			std::cout << std::endl << "reading settings from " << filename << std::endl;
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->NUM_INIT = boost::lexical_cast<int>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->LAG = boost::lexical_cast<int>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->LDA_INIT_MAX = boost::lexical_cast<int>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->VAR_MAX_ITER = boost::lexical_cast<int>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->VAR_CONVERGED = boost::lexical_cast<double>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->MSTEP_MAX_ITER = boost::lexical_cast<int>(line);	
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->MAX_EPS = boost::lexical_cast<double>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->EM_MAX_ITER = boost::lexical_cast<int>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->EM_CONVERGED = boost::lexical_cast<double>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->OFFSET = boost::lexical_cast<int>(line);
			dataS.ignore(256, '=');
			std::getline(dataS, line);
			this->ADJN = boost::lexical_cast<int>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->LAMBDASTART = boost::lexical_cast<int>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->LAMBDAEND = boost::lexical_cast<int>(line);
			dataS.ignore(256,'=');
			std::getline(dataS,line);
			this->CROSSVAL = boost::lexical_cast<int>(line);
			dataS.ignore(256, '=');
			std::getline(dataS, line);
			this->ALPHA = boost::lexical_cast<int>(line);
			dataS.ignore(256, '=');
			std::getline(dataS, line);
			this->ETA = boost::lexical_cast<int>(line);
			dataS.ignore(256, '=');
			std::getline(dataS, line);
			this->WT = boost::lexical_cast<double>(line);

		dataS.close();
		std::cout << "LDA maximum initialisation " << this->LDA_INIT_MAX << std::endl;
		std::cout << "Estimation step maximum iterations : " << this->VAR_MAX_ITER << std::endl;
		std::cout << "Estimation step convergence : " << this->VAR_CONVERGED << std::endl;
		std::cout << "Maximisation step maximum iterations : " << this->MSTEP_MAX_ITER << std::endl;
		std::cout << "Maximisation step convergence : " << this->MAX_EPS << std::endl;
		std::cout << "EM maximum iterations : " << this->EM_MAX_ITER << std::endl;
		std::cout << "EM convergence : " << this->EM_CONVERGED << std::endl;
		std::cout << "Path for finding lambda : " << this->LAMBDASTART << " - " << this->LAMBDAEND << std::endl;
		std::cout << this->CROSSVAL << " fold cross validation will be used " << std::endl;
		if (this->ALPHA == 1) std::cout << "Alpha will be estimated." << std::endl;
		else std::cout << "Alpha will be fixed." << std::endl;
		if (this->ETA == 1) std::cout << "Eta will be estimated." << std::endl;
		else std::cout << "Eta will be fixed." << std::endl;
		std::cout << "Initial proportionate increase in weight on the baseline group : " << this->WT << std::endl;
		if (this->ADJN > 0) this->ADJN = this->ADJN + 1; //need intercept
		return 1;
    }
};

#endif // SETTINGS_H

