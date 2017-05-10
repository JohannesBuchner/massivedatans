/***
This file is part of nested_sampling, a pythonic implementation of various
nested sampling algorithms.

Author: Johannes Buchner (C) 2013-2016
License: AGPLv3

See README and LICENSE file.
***/
#include<stdbool.h>
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
#ifdef PARALLEL
#include<omp.h>
#endif

#define IFVERBOSE if(0)
#define IFDEBUG if(0)
#define adouble double
#define bdouble double
#define sqr(x) (pow(x,2))

int like(
	const void * xp, const void * yyp, const int ndata, const int nx,
	const double A, const double mu, const double sig,
	const double noise_level,
	const void * data_maskp, 
	void * Loutp
) {
	const adouble * x = (const adouble*) xp;
	const adouble * yy = (const adouble*) yyp;
	const bool * data_mask = (const bool*) data_maskp;
	adouble * Lout = (adouble*) Loutp;
	
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int j = 0; j < nx; j++) {
		const double ypred = A * exp(-0.5 * sqr((mu - x[j])/sig));
		
		int k = 0;
		for (int i = 0; i < ndata; i++) {
			IFVERBOSE printf("data_mask %d: %d\n", i, data_mask[i]);
			if (data_mask[i]) {
				IFVERBOSE printf("y %d %d: %f %f\n", i, j, yy[i + j*ndata], ypred);
				Lout[k] += sqr((ypred - yy[i + j*ndata]) / noise_level);
				k++;
			}
		}
	}
	IFVERBOSE {
		int k = 0;
		for (int i = 0; i < ndata; i++) {
			if (data_mask[i]) {
				printf("L %d: %f\n", k, Lout[k]);
				k++;
			}
		}
	}
	return 0;
}

