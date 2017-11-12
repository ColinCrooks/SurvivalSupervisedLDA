// Latent Dirichlet Allocation supervised by penalised Cox proportional hazards modelling with optional learning of asymmetrical priors.

//This has been developed from the original code (C) Copyright 2009, Chong Wang, David Blei and Li Fei-Fei ([1] Blei DM, McAuliffe JD. Supervised Topic Models. Adv Neural Inf Process Syst 20 2007:121–8.) and modified following the algorithms developed by Ye et al. 2014 ([1] Ye S, Dawson JA, Kendziorski C. Extending information retrieval methods to personalized genomic-based studies of disease. Cancer Inform 2014;13:85–95. doi:10.4137/CIN.S16354.)

// Modifications by Colin Crooks (colin.crooks@nottingham.ac.uk)

// This file is part ofsslda.

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


#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/lexical_cast.hpp>

#include "corpus.h"

corpus::corpus()
{
    num_docs = 0;
    size_vocab = 0;
    num_classes = 0;
    num_total_words = 0;
	event_times = 0 ;
	max_length = 0;
}

corpus::~corpus()
{
    for (int i = 0; i < num_docs; i ++)
    {
        document * doc = docs[i];
		delete doc;
    }

    num_docs = 0;
    size_vocab = 0;
    num_classes = 0;
    num_total_words = 0;
	event_times = 0;
	max_length = 0;
	time_start = 0;
	time_end = 0;
}

int corpus::read_adj(const char * data_filename, const settings * setting)
{
	std::string filename;
	filename = string(data_filename);
	std::string line;
	std::ifstream data(filename);
	if (data.good() == false)
	{
		std::cerr << "Unable to open covariates data file" << std::endl;
		return -1;
	}
	std::cout << std::endl << "Reading covariates to adjust from " << data_filename << std::endl;
	
	int d = 0;
	while (std::getline(data, line))
	{
		docs[d]->covariates = new double [setting->ADJN];
		docs[d]->covariates[0] = 1.0;
		for (int w = 1; w <= setting->ADJN; w++)
			docs[d]->covariates[w] = 0;
		std::stringstream  lineStream(line);
		std::string        cell;
		for (int i = 1; i <= setting->ADJN; i++)
		{
			std::getline(lineStream, cell, ',');
			docs[d]->covariates[i] = boost::lexical_cast<int>(cell);
		}
		d++;
	}
	data.close();
	return 1;
 }



int corpus::read_data(const char * data_filename, const settings * setting)
{
    int length = 0,  word = 0,
        n = 0, nd = 0, nw = 0, label = 0, time_exit = 0, time_enter = 0, begin = 0;
	double count;
	std::string filename;
	filename=string(data_filename);
	std::string line;
	std::ifstream data(filename);
	if (data.good() == false)
	{
		std::cerr << "Unable to open codes data file" << std::endl;
		return -1;
	}
    std::cout << std::endl << "reading data from " << data_filename << std::endl;

    while (std::getline(data,line))
    {
		std::stringstream  lineStream(line);
		std::string        cell;

		std::getline(lineStream, cell,',');
		label = boost::lexical_cast<int>(cell);

		std::getline(lineStream, cell, ',');
		time_enter = boost::lexical_cast<int>(cell);
		
		std::getline(lineStream, cell, ',');
		time_exit = boost::lexical_cast<int>(cell);

		std::getline(lineStream, cell,',');
		length = boost::lexical_cast<int>(cell); 

		document * doc = new document(length);
		doc->label = label;
		doc->t_enter = time_enter;
		doc->t_exit= time_exit;
		if (begin ==0 || doc->t_enter < time_start)
			time_start = doc->t_enter;
		if (begin ==0 || doc->t_exit > time_end)
			time_end = doc->t_exit;
		if (label >= num_classes)
			num_classes = label + 1;
		begin = 1;
	//	std::cout << "time_start : " << time_start << " time_end : " << time_end<< std::endl;
	//	std::cout << doc->label << "," << doc->t_enter << "," << doc->t_exit << "," << length << std::endl;;
		std::istringstream(line) >> label >> time_enter >> time_exit >> length;
		if (length > max_length) max_length = length;
		for (n = 0; n < length; n++)
		{
			std::string cell1;
			std::string cell2;
			std::getline(lineStream, cell, ',');
			std::stringstream  CellStream(cell);
			std::getline(CellStream, cell1, ':');
			std::getline(CellStream, cell2, ' ');
			word = boost::lexical_cast<int>(cell1);
			count = boost::lexical_cast<double>(cell2);
			word = word  - setting->OFFSET;
			doc->words[n] = word;
			doc->counts[n] = count;
			doc->total += count;
			if (word >= nw)
				nw = word + 1;
		}
        num_total_words += doc->total;
		doc->covariates = NULL;
		docs.push_back(doc);
        nd++;
    }
	data.close();
	event_times = time_end - time_start + 1;
	num_docs = nd;
    size_vocab = nw;
    std::cout << "Number of docs  : " << num_docs << std::endl;
	std::cout << "Number of terms : " << size_vocab << std::endl;
	std::cout << "Total number of words : "<< num_total_words << std::endl;
    assert(nd == int(docs.size()));
	std::cout << "Number of event times : " << event_times;
	std::cout << " from " << time_start << " to " << time_end << std::endl << std::endl;
	return 1;
}



