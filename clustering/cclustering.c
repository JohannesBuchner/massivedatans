/***
This file is part of nested_sampling, a pythonic implementation of various
nested sampling algorithms.

Author: Johannes Buchner (C) 2013-2016
License: AGPLv3

See README and LICENSE file.
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

// evil global variable to pass into qsort
const double * current_dists;

int compare_dists(const void * ap, const void * bp) {
	const int a = * (int *) ap;
	const int b = * (int *) bp;
	if (current_dists[a] < current_dists[b]) {
		return -1;
	} else {
		return +1;
	}
}

/**
 *
 * :param number_of_neighbors: J, how many neighbors of a point to consider
 * :param threshold_number_of_common_neighbors: K, how many neighbors two points have to have in common to be put into the same cluster.
 * 
 * clusters needs to be set to arange(n)
 * 
 */

int jarvis_patrick_clustering(
	const void * xxp, int n, int J, int K, 
	int * clusters
) {
	const adouble * dists = (const adouble*) xxp;
	int neighbors_list[n][J];
	int neighbors_list_i[n];
	
	for (int i = 0; i < n; i++) {
		// order its nearest neighbors
		for (int j = 0; j < n; j++) {
			neighbors_list_i[j] = j;
		}
		current_dists = dists + i * n;
		qsort(neighbors_list_i, n, sizeof(int), compare_dists);
		// now neighbors_list_i should be sorted, with nearest at 0
		// we want 1...J+1
		for (int j = 0; j < J; j++) {
			neighbors_list[i][j] = neighbors_list_i[j+1];
		}
	}
	for (int i = 0; i < n; i++) {
		IFVERBOSE {
			printf("cluster %d: %d\n", i, clusters[i]);
			for (int k = 0; k < J; k++) {
				printf("    has neighbor: %d\n", neighbors_list[i][k]);
			}
		}
		for (int j = 0; j < i; j++) {
			// check if i in neighbors_list
			int is_neighbor = 0;
			for (int k = 0; k < J; k++) {
				if (neighbors_list[j][k] == i) {
					is_neighbor |= 1;
				}
				if (neighbors_list[i][k] == j) {
					is_neighbor |= 2;
				}
			}
			if (is_neighbor != 3) {
				continue;
			}
			
			// count how many in common in neighbors_list
			int in_common = 0;
			for (int k = 0; k < J; k++) {
				int a = neighbors_list[i][k];
				for (int k2 = 0; k2 < J; k2++) {
					if (neighbors_list[j][k2] == a) {
						in_common++;
						break;
					}
				}
			}
			IFVERBOSE printf("%d ^ %d : have %d in common. clusters: %d %d\n", 
				i, j, in_common, clusters[i], clusters[j]);
			if (in_common < K)
				continue;
			// re-assign clusters
			int c1 = clusters[i];
			int c2 = clusters[j];
			if (c1 == c2)
				continue;
			if (c1 > c2) {
				c1 = clusters[j];
				c2 = clusters[i];
			}
			// move all from c2 to c1
			IFVERBOSE printf("moving %d -> %d\n", c2, c1);
			for (int k = 0; k < n; k++) {
				if (clusters[k] == c2) {
					clusters[k] = c1;
				}
			}
		}
	}

	return 0;
}

