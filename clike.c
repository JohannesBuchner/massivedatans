/***

Likelihood implementation in C
--------------------------------

Copyright (c) 2017 Johannes Buchner

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

// Parallelisation does not work at the moment, you are welcome to fix it

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
	
	{
	#ifdef PARALLEL
	int k = 0;
	#pragma omp parallel for
	// this is stupid because it does not actually safe model evaluations,
	// but at least it should run faster for our testing purposes.
	for (int i = 0; i < ndata; i++) {
		if (data_mask[i]) {
			Lout[k] = 0;
			for (int j = 0; j < nx; j++) {
				const double ypred = A * exp(-0.5 * sqr((mu - x[j])/sig));
				IFVERBOSE printf("y %d %d: %f %f\n", i, j, yy[i + j*ndata], ypred);
				Lout[k] += sqr((ypred - yy[i + j*ndata]) / noise_level);
			}
			k++;
		}
	}
	#else
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
	#endif
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

