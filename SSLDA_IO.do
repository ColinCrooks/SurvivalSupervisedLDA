
cap mata mata drop loadsldamodel()
mata:
void loadsldamodel(string scalar filename)
{
    file = fopen(filename, "r")
    external numeric vector params
	params = J(5,1,.)
	external numeric matrix logprob
	external numeric vector cov_beta 
	external numeric vector beta 
	external numeric vector alpha
	external numeric vector basehaz
	xline = fget(file)
	xtoken=tokens(xline, ":")
	params[1,1]=strtoreal(xtoken[1,3])
	xline = fget(file)
	xtoken=tokens(xline, ":")
    params[2,1] = strtoreal(xtoken[1,3])
	xline = fget(file)
	xtoken=tokens(xline, ":")	
	params[3,1] = strtoreal(xtoken[1,3])
	xline = fget(file)
	xtoken=tokens(xline, ":")	
	params[4,1] = strtoreal(xtoken[1,3])
	xline= fget(file)
	xtoken=tokens(xline, ":")
	params[5,1] = strtoreal(xtoken[1,3])

	xline = fget(file)
	logprob=J(params[2,1], params[1,1] ,.)
	for (k = 1; k <= params[2,1]; k++)
	{
		xline = fget(file)
		xtoken=tokens(xline, " ")
		for ( j = 1; j<= params[1,1]; j++)
		{	

			logprob[k , j] = strtoreal(xtoken[1,j])
		}
    }
	if (params[5,1] > 0)
	{
		cov_beta = J(1,params[5,1],.)
		xline = fget(file)
		xline = fget(file)
		xtoken=tokens(xline, " ")
		
		for (k = 1; k <= params[5,1]; k++)
		{
			cov_beta[1,k] = strtoreal(xtoken[1,k])
		}
	}
	
	beta = J(1,params[1,1],.)
	xline = fget(file)
	xline = fget(file)
	xtoken=tokens(xline, " ")
	for (k = 1; k <= params[1,1]; k++)
	{
		beta[1,k] = strtoreal(xtoken[1,k])
	}
 
	basehaz = J(params[3,1],1,.)
	xline = fget(file)
	for (k = 1; k <= params[3,1]; k++)
	{
		xline = fget(file)
		basehaz[k,1] = strtoreal(xline)
	}	
	alpha=J(params[1,1], 1 ,.)
	xline = fget(file)
	xline = fget(file)
	xtoken=tokens(xline, " ")
	for (k = 1; k <= params[1,1]; k++)
	{
		xline = fget(file)
		alpha[k,1] = strtoreal(xtoken[1,k])
	}	
	prob = exp(logprob)
	st_matrix("prob",prob)
	fclose(file)
}
end

cap mata mata drop exportsldadata()
mata:
void exportsldadata(string scalar filename)
{
		outfile = fopen(filename, "w")
		length = st_data(.,"length") 
		death = st_data(.,"_d")
		t0 = st_data(.,"_t0")
		time = st_data(.,"_t")
		code = st_data(.,"code")
		count = st_data(.,"count")
		row = 1
		while( row <= rows(length)) 
		{
			s = (strofreal(death[row]),",",strofreal(t0[row]),",",strofreal(time[row]),",",strofreal(length[row]),",")
			fwrite(outfile, invtokens(s,""))
			ncodes=length[row]
			for (i = 1; i < ncodes ; i++)
			{
				s = (strofreal(code[row]),":",strofreal(count[row])," ,")
				fwrite(outfile, invtokens(s,""))
				row = row + 1
			}
			s = (strofreal(code[row]),":",strofreal(count[row])," ")
			fwrite(outfile, invtokens(s,""))
			row = row + 1
			fput(outfile," ")
		}
		fclose(outfile)
	
}
end
