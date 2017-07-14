/***

Neighbourhood helper functions accelerated with parallelised C
---------------------------------------------------------------

Copyright (c) 2017 Johannes Buchner

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

***/

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

double most_distant_nearest_neighbor(
	const void * xxp, int nsamples, int ndim
) {
	const adouble * xx = (const adouble*) xxp;
	double nearest_ds[nsamples];

	IFVERBOSE {
		for (int i = 0; i < nsamples; i++) { // one sample at a time
			printf("%d: ", i);
			for (int k = 0; k < ndim; k++) {
				printf("%e\t", xx[i*ndim + k]);
			}
			printf("\n");
		}
	}
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < nsamples; i++) { // one sample at a time
		// consider all other samples before i
		double nearest_d = 1e300;
		for (int j = 0; j < nsamples; j++) {
			if (j != i) {
				double d = 0;
				for (int k = 0; k < ndim; k++) {
					d += sqr(xx[i*ndim + k] - xx[j*ndim + k]);
				}
				if (d < nearest_d) {
					nearest_d = d;
				}
			}
		}
		IFVERBOSE printf("%d: %f\n", i, sqrt(nearest_d));
		nearest_ds[i] = sqrt(nearest_d);
	}
	double furthest_d = nearest_ds[0];

	for (int i = 1; i < nsamples; i++) {
		if (nearest_ds[i] > furthest_d)
			furthest_d = nearest_ds[i];
	}
	IFVERBOSE printf("result: %f\n", furthest_d);
	return furthest_d;
}

int is_within_distance_of(
	const void * xxp, int nsamples, int ndim, double maxdistance, const void * yp
) {
	const adouble * xx = (const adouble*) xxp;
	const adouble * y = (const adouble*) yp;

	for (int i = 0; i < nsamples; i++) { // one sample at a time
		double d = 0;
		for (int k = 0; k < ndim; k++) {
			d += sqr(xx[i*ndim + k] - y[k]);
		}
		if (sqrt(d) < maxdistance)
			return 1;
	}
	return 0;
}


int count_within_distance_of(
	const void * xxp, int nsamples, int ndim, double maxdistance, 
	const void * yyp, int nothers, void * outp, const int countmax
) {
	const adouble * xx = (const adouble*) xxp;
	const adouble * yy = (const adouble*) yyp;
	double * out = (double*) outp;

	for (int j = 0; j < nothers; j++) { // one sample at a time
		for (int i = 0; i < nsamples; i++) { // one sample at a time
			double d = 0;
			for (int k = 0; k < ndim; k++) {
				d += sqr(xx[i*ndim + k] - yy[j*ndim + k]);
			}
			if (sqrt(d) < maxdistance) {
				out[j]++;
				// printf("%d: %f\n", j, out[j]);
				if (countmax > 0 && out[j] >= countmax) {
					break;
				}
			}
		}
	}
	return 0;
}

/**
 * xxp are double points (nsamples x ndim)
 * choicep is whether the point is selected in the bootstrap round (nsamples x nbootstraps)
 */
double bootstrapped_maxdistance(
	const void * xxp, 
	int nsamples, int ndim,
	const void * choicep,
	int nbootstraps
) {
	const adouble * xx = (const adouble*) xxp;
	const adouble * chosen = (const adouble*) choicep;

	double furthest_ds[nbootstraps];
	double furthest_d_bs;
	
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for(int b = 0; b < nbootstraps; b++) {
		double nearest_ds[nsamples];
		double furthest_d = 0;
		//printf("bootstrap round %d\n", b);
		// find one that was not chosen
		for (int i = 0; i < nsamples; i++) {
			if (chosen[i*nbootstraps + b] != 0) continue;
			//printf("   considering %d\n", i);
			double nearest_d = 1e300;
			for (int j = 0; j < nsamples; j++) {
				if (chosen[j*nbootstraps + b] == 0) continue;
				double d = 0;
				for (int k = 0; k < ndim; k++) {
					d += sqr(xx[i*ndim + k] - xx[j*ndim + k]);
				}
				if (d < nearest_d) {
					nearest_d = d;
				}
			}
			//printf("    %d: %f\n", i, sqrt(nearest_d));
			nearest_ds[i] = sqrt(nearest_d);
		}
		for (int i = 1; i < nsamples; i++) {
			if (chosen[i*nbootstraps + b] != 0) continue;
			if (nearest_ds[i] > furthest_d)
				furthest_d = nearest_ds[i];
		}
		//printf("bootstrap round %d gave %f\n", b, furthest_d);
		furthest_ds[b] = furthest_d;
	}
	
	furthest_d_bs = furthest_ds[0];
	for (int i = 1; i < nbootstraps; i++) {
		if (furthest_ds[i] > furthest_d_bs)
			furthest_d_bs = furthest_ds[i];
	}

	IFVERBOSE printf("result: %f\n", furthest_d_bs);
	return furthest_d_bs;
}
