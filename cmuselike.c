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
// ret = lib.like(yd, vd, ypred, data_mask, ndata, nspec, Lout)
int like(
	const void * yyp, const void * vvp, const void * ypredp, const void * data_maskp, 
	const int ndata, const int nx,
	void * Loutp
) {
	const adouble * yy = (const adouble*) yyp;
	const adouble * vv = (const adouble*) vvp;
	const adouble * ypred = (const adouble*) ypredp;
	const bool * data_mask = (const bool*) data_maskp;
	adouble * Lout = (adouble*) Loutp;
	
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < ndata; i++) {
		if (data_mask[i]) {
			// compute s
			double s1 = 0.;
			double s2 = 1e-10;
			for (int j = 0; j < nx; j++) {
				s1 += yy[i+j*ndata] * ypred[j] / vv[i+j*ndata];
				s2 += pow(ypred[j], 2) / vv[i+j*ndata];
			}
			double s = s1/s2;
			double chi = 0.;
			for (int j = 0; j < nx; j++) {
				chi += pow(yy[i+j*ndata] - s * ypred[j], 2) / vv[i+j*ndata];
			}
			Lout[i] = -0.5 * chi;
		}
	}
	return 0;
}

