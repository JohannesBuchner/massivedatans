from __future__ import print_function, division
"""

Neighbourhood helper functions
-------------------------------

Copyright (c) 2017 Johannes Buchner

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy
import scipy.spatial

def initial_maxdistance_guess(u):
	n = len(u)
	distances = scipy.spatial.distance.cdist(u, u)
	nearest = [distances[i,:].argsort()[1] for i in range(n)]
	nearest = [numpy.abs(u[k,:] - u[i,:]) for i, k in enumerate(nearest)]
	# compute distance maximum
	maxdistance = numpy.max(nearest, axis=0)
	return maxdistance

def update_maxdistance(u, ibootstrap, maxdistance, verbose = False):
	n, ndim = u.shape
	
	# bootstrap to find smallest maxdistance which includes
	# all points
	choice = list(set(numpy.random.choice(numpy.arange(n), size=n)))
	notchosen = set(range(n)) - set(choice)
	# check if included with our starting criterion
	for i in notchosen:
		dists = numpy.abs(u[i,:] - u[choice,:])
		close = numpy.all(dists < maxdistance.reshape((1,-1)), axis=1)
		assert close.shape == (len(choice),), (close.shape, len(choice))
		# find the point where we have to increase the least
		if not close.any():
			# compute maxdists -- we already did that
			# compute extension to maxdistance
			#maxdistance_suggest = [numpy.max([maxdistance, d], axis=0) for d in dists]
			maxdistance_suggest = numpy.where(maxdistance > dists, dists, maxdistance)
			assert maxdistance_suggest.shape == (len(dists), ndim)
			# compute volume increase in comparison to maxdistance
			#increase = [(numpy.log(m) - numpy.log(maxdistance)).sum()  for m in maxdistance_suggest]
			increase = numpy.log(maxdistance_suggest).sum(axis=1) - numpy.log(maxdistance).sum()
			
			# choose smallest
			nearest = numpy.argmin(increase)
			if verbose: print(ibootstrap, 'nearest:', u[i], u[nearest], increase[nearest])
			# update maxdistance
			maxdistance = numpy.where(dists[nearest] > maxdistance, dists[nearest], maxdistance)
			if verbose: print(ibootstrap, 'extending:', maxdistance)
		else:
			# we got this one, everything is fine
			pass
	return maxdistance

def find_maxdistance(u, verbose=False, nbootstraps=15):
	# find nearest point for every point
	if verbose: print('finding nearest neighbors:')
	maxdistance = initial_maxdistance_guess(u)
	#maxdistance = numpy.zeros(ndim)
	if verbose: print('initial:', maxdistance)
	for ibootstrap in range(nbootstraps):
		maxdistance = update_maxdistance(u, ibootstrap, maxdistance, verbose=verbose)
	return maxdistance

def is_within_distance_of(members, maxdistance, u, metric='euclidean'):
	dists = scipy.spatial.distance.cdist(members, us, metric=metric)
	return (dists < maxdistance).any()

def count_within_distance_of(members, maxdistance, us, metric='euclidean'):
	dists = scipy.spatial.distance.cdist(members, us, metric=metric)
	return (dists < maxdistance).sum(axis=0)

def any_within_distance_of(members, maxdistance, us, metric='euclidean'):
	dists = scipy.spatial.distance.cdist(members, us, metric=metric)
	return (dists < maxdistance).any(axis=0)

most_distant_nearest_neighbor = None
bootstrapped_maxdistance = None
try:
	import os
	from ctypes import *
	from numpy.ctypeslib import ndpointer

	if int(os.environ.get('OMP_NUM_THREADS', '1')) > 1:
		libname = 'cneighbors-parallel.so'
	else:
		libname = 'cneighbors.so'
	libfilename = os.path.join(os.path.dirname(os.path.abspath(__file__)), libname)
	lib = cdll.LoadLibrary(libfilename)
	lib.most_distant_nearest_neighbor.argtypes = [
		ndpointer(dtype=numpy.float64, ndim=2, flags='C_CONTIGUOUS'), 
		c_int, 
		c_int, 
		]
	lib.most_distant_nearest_neighbor.restype = c_double

	def most_distant_nearest_neighbor(xx):
		i, m = xx.shape
		r = lib.most_distant_nearest_neighbor(xx, i, m)
		return r

	lib.is_within_distance_of.argtypes = [
		ndpointer(dtype=numpy.float64, ndim=2, flags='C_CONTIGUOUS'), 
		c_int, 
		c_int, 
		c_double, 
		ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
		]
	lib.is_within_distance_of.restype = c_int

	def is_within_distance_of(xx, maxdistance, y):
		i, m = xx.shape
		r = lib.is_within_distance_of(xx, i, m, maxdistance, y)
		return r == 1

	lib.count_within_distance_of.argtypes = [
		ndpointer(dtype=numpy.float64, ndim=2, flags='C_CONTIGUOUS'), 
		c_int, 
		c_int, 
		c_double, 
		ndpointer(dtype=numpy.float64, ndim=2, flags='C_CONTIGUOUS'), 
		c_int, 
		ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
		c_int, 
		]

	def count_within_distance_of(xx, maxdistance, yy):
		i, m = xx.shape
		j = len(yy)
		counts = numpy.zeros(len(yy))
		r = lib.count_within_distance_of(xx, i, m, maxdistance, yy, j, counts, 0)
		counts = counts.astype(int)
		# check
		#dists = scipy.spatial.distance.cdist(xx, yy, metric='euclidean')
		#counts_true = (dists < maxdistance).sum(axis=0)
		#assert (counts == counts_true).all(), (counts, counts_true)
		return counts

	def any_within_distance_of(xx, maxdistance, yy):
		i, m = xx.shape
		j = len(yy)
		counts = numpy.zeros(len(yy))
		r = lib.count_within_distance_of(xx, i, m, maxdistance, yy, j, counts, 1)
		counts = counts > 0
		# check
		#dists = scipy.spatial.distance.cdist(xx, yy, metric='euclidean')
		#counts_true = (dists < maxdistance).any(axis=0)
		#assert (counts == counts_true).all(), (counts, counts_true)
		return counts

	lib.bootstrapped_maxdistance.argtypes = [
		ndpointer(dtype=numpy.float64, ndim=2, flags='C_CONTIGUOUS'), 
		c_int, 
		c_int, 
		ndpointer(dtype=numpy.float64, ndim=2, flags='C_CONTIGUOUS'), 
		c_int, 
		]
	lib.bootstrapped_maxdistance.restype = c_double
	
	def bootstrapped_maxdistance(xx, nbootstraps):
		nsamples, ndim = xx.shape
		chosen = numpy.zeros((nsamples, nbootstraps))
		for b in range(nbootstraps):
			chosen[numpy.random.choice(numpy.arange(nsamples), size=nsamples, replace=True),b] = 1.
		
		maxdistance = lib.bootstrapped_maxdistance(xx, nsamples, ndim, chosen, nbootstraps)
		return maxdistance

except ImportError as e:
	print('Using slow, high-memory neighborhood function nearest_rdistance_guess because import failed:', e)
except Exception as e:
	print('Using slow, high-memory neighborhood function nearest_rdistance_guess because:', e)


def nearest_rdistance_guess(u, metric='euclidean'):
	if metric == 'euclidean' and most_distant_nearest_neighbor is not None:
		return most_distant_nearest_neighbor(u)
	n = len(u)
	distances = scipy.spatial.distance.cdist(u, u, metric=metric)
	numpy.fill_diagonal(distances, 1e300)
	nearest_neighbor_distance = numpy.min(distances, axis = 1)
	rdistance = numpy.max(nearest_neighbor_distance)
	#print 'distance to nearest:', rdistance, nearest_neighbor_distance
	return rdistance

def initial_rdistance_guess(u, metric='euclidean', k = 10):
	n = len(u)
	distances = scipy.spatial.distance.cdist(u, u, metric=metric)
	if k == 1:
	#	numpy.diag(distances)
	#	nearest = [distances[i,:])[1:k] for i in range(n)]
		distances2 = distances + numpy.diag(1e100 * numpy.ones(len(distances)))
		nearest = distances2.min(axis=0)
	else:
		assert False, k
		nearest = [numpy.sort(distances[i,:])[1:k+1] for i in range(n)]
	# compute distance maximum
	rdistance = numpy.max(nearest)
	return rdistance

def update_rdistance(u, ibootstrap, rdistance, verbose = False, metric='euclidean'):
	n, ndim = u.shape
	
	# bootstrap to find smallest rdistance which includes
	# all points
	choice = set(numpy.random.choice(numpy.arange(n), size=n))
	mask = numpy.array([c in choice for c in numpy.arange(n)])
	
	distances = scipy.spatial.distance.cdist(u[mask], u[-mask], metric=metric)
	assert distances.shape == (mask.sum(), (-mask).sum())
	nearest_distance_to_members = distances.min(axis=0)
	if verbose:
		print('nearest distances:', nearest_distance_to_members.max(), nearest_distance_to_members)
	newrdistance = max(rdistance, nearest_distance_to_members.max())
	if newrdistance > rdistance and verbose:
		print(ibootstrap, 'extending:', newrdistance)
	return newrdistance

def find_rdistance(u, verbose=False, nbootstraps=15, metric='euclidean'):
	if metric == 'euclidean' and bootstrapped_maxdistance is not None:
		return bootstrapped_maxdistance(u, nbootstraps)
	# find nearest point for every point
	if verbose: print('finding nearest neighbors:')
	rdistance = 0 #initial_rdistance_guess(u)
	if verbose: print('initial:', rdistance)
	for ibootstrap in range(nbootstraps):
		rdistance = update_rdistance(u, ibootstrap, rdistance, verbose=verbose, metric=metric)
	return rdistance

if __name__ == '__main__':
	nbootstraps = 10
	numpy.random.seed(1)
	u = numpy.random.uniform(size=(200,2))
	for i in range(100):
		numpy.random.seed(i)
		a = bootstrapped_maxdistance(u, nbootstraps)
		numpy.random.seed(i)
		b = find_rdistance(u, nbootstraps=nbootstraps, metric='euclidean', verbose=False)
		print(a, b)
		assert numpy.allclose(a, b)
		
