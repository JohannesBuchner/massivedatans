from __future__ import print_function, division
"""

Main program
---------------

Copyright (c) 2017 Johannes Buchner

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy
from numpy import exp
import h5py
import sys
import json
import os
import time

print('loading data...')
ndata = int(sys.argv[2])
with h5py.File(sys.argv[1], 'r') as f:
	x = numpy.array(f['x'].value)
	y = numpy.array(f['y'][:,:ndata])


"""

Definition of the problem
- parameter space (here: 3d)
- likelihood function which consists of 
  - model function ("slow predicting function")
  - data comparison

"""

nx, ndata = y.shape
noise_level = 0.01
params = ['A', 'mu', 'sig'] #, 'noise_level']
nparams = len(params)

def gauss(x, z, A, mu, sig):
	return A * exp(-0.5 * ((mu - x / (1. + z))/sig)**2)

def priortransform(cube):
	# definition of the parameter width, by transforming from a unit cube
	cube = cube.copy()
	cube[0] = 10**(cube[0] * 2 - 2)
	cube[1] = cube[1] * 400 + 400
	cube[2] = cube[2] * 2
	return cube

# the following is a python-only implementation of the likelihood 
# @ params are the parameters (as transformed by priortransform)
# @ data_mask is which data sets to consider.
# returns a likelihood vector
def multi_loglikelihood(params, data_mask):
	A, mu, log_sig_kms = params
	# predict the model
	sig = 10**log_sig_kms
	ypred = A * exp(-0.5 * ((mu - x)/sig)**2)
	# do the data comparison
	L = -0.5 * (((ypred.reshape((-1,1)) - y[:,data_mask])/noise_level)**2).sum(axis=0)
	return L

#print multi_loglikelihood([0.88091237,  444.44207558,    2.77671952], numpy.ones(ndata)==1)
#print multi_loglikelihood([1.65758829e-01, 4.45518543e+02, 3.25894638e+00], numpy.ones(ndata)==1)
#print multi_loglikelihood([0.95572931,  443.99407818,    2.95764509], numpy.ones(ndata)==1)

# The following is a C implementation of the likelihood
from ctypes import *
from numpy.ctypeslib import ndpointer

if int(os.environ.get('OMP_NUM_THREADS', '1')) > 1 and False: # does not work correctly yet
	lib = cdll.LoadLibrary('./clike-parallel.so')
else:
	lib = cdll.LoadLibrary('./clike.so')
lib.like.argtypes = [
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=2, flags='C_CONTIGUOUS'), 
	c_int, 
	c_int, 
	c_double, 
	c_double, 
	c_double, 
	c_double, 
	ndpointer(dtype=numpy.bool, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	]

# @ params are the parameters (as transformed by priortransform)
# @ data_mask is which data sets to consider.
# returns a likelihood vector
def multi_loglikelihood(params, data_mask):
	A, mu, log_sig_kms = params
	sig = 10**log_sig_kms
	Lout = numpy.zeros(data_mask.sum())
	# do everything in C and return the resulting likelihood vector
	ret = lib.like(x, y, ndata, nx, A, mu, sig, noise_level, data_mask, Lout)
	#assert numpy.isfinite(Lout).all(), (Lout, params)
	return -0.5 * Lout

#print multi_loglikelihood([0.88091237,  444.44207558,    2.77671952], numpy.ones(ndata)==1)
#print multi_loglikelihood([1.65758829e-01, 4.45518543e+02, 3.25894638e+00], numpy.ones(ndata)==1)
#print multi_loglikelihood([0.95572931,  443.99407818,    2.95764509], numpy.ones(ndata)==1)

"""

After defining the problem, we use generic code to set up 
- Nested Sampling (Multi)Integrator
- Our special sampler
- RadFriends (constrained region draw)

We start with the latter.
"""


from multi_nested_integrator import multi_nested_integrator
from multi_nested_sampler import MultiNestedSampler

import cachedconstrainer
from cachedconstrainer import CachedConstrainer, generate_individual_constrainer, generate_superset_constrainer, MultiEllipsoidalConstrainer, MetricLearningFriendsConstrainer, generate_fresh_constrainer

constrainer_type = os.environ.get('CONSTRAINER', 'MLFRIENDS')
if constrainer_type == 'MLFRIENDS':
	def generate_fresh_constrainer():
		return MetricLearningFriendsConstrainer(
			metriclearner = 'truncatedscaling', force_shrink=True,
			rebuild_every=1000, metric_rebuild_every=20, 
			verbose=False)

	superset_constrainer = MetricLearningFriendsConstrainer(
			metriclearner = 'truncatedscaling', force_shrink=True,
			rebuild_every=1000, metric_rebuild_every=20, 
			verbose=False)
elif constrainer_type == 'MULTIELLIPSOIDS':
	def generate_fresh_constrainer():
		return MultiEllipsoidalConstrainer(rebuild_every=1000)

	superset_constrainer = generate_fresh_constrainer()
elif constrainer_type == 'SLICE':
	#from whitenedmcmc import FilteredMCMCConstrainer, HybridMLMultiEllipsoidConstrainer
	from whitenedmcmc import SliceConstrainer, FilteredMahalanobisHARMProposal, FilteredUnitIterateSliceProposal
	def generate_fresh_constrainer():
		return SliceConstrainer(proposer=FilteredUnitIterateSliceProposal(), nsteps=nparams*5)
	superset_constrainer = generate_fresh_constrainer()
else:
	assert False, constrainer_type

cachedconstrainer.generate_fresh_constrainer = generate_fresh_constrainer

cc = CachedConstrainer()
focusset_constrainer = cc.get
_, _, individual_draw_constrained = generate_individual_constrainer()
numpy.random.seed(1)
start_time = time.time()
print('setting up integrator ...')
nlive_points = int(os.environ.get('NLIVE_POINTS','400'))

# constrained region draw functions
# we try hard to keep information about current regions and subselected regions
# because recomputing the regions is expensive if the likelihood is very fast.
# There are three constrainers:
#   - the one of the superset (all data sets)
#   - one for each data set if need a individual draw (focussed draw with only one)
#   - a memory for recent clusterings, because they might recur in the next iteration(s)
# Note that this does caching not improve the algorithms efficiency
#   in fact, not recomputing regions keeps the regions larger, 
#   leading potentially to slightly more rejections. 
# However, there is substantial execution speedup.


# now set up sampler and pass the three constrainers

sampler = MultiNestedSampler(nlive_points = nlive_points, 
	priortransform=priortransform, multi_loglikelihood=multi_loglikelihood, 
	ndim=nparams, ndata=ndata,
	superset_draw_constrained = superset_constrainer.draw_constrained, 
	individual_draw_constrained = individual_draw_constrained,
	draw_constrained = focusset_constrainer, 
	nsuperset_draws = int(os.environ.get('SUPERSET_DRAWS', '10')),
	use_graph = os.environ.get('USE_GRAPH', '1') == '1'
)

superset_constrainer.sampler = sampler
cc.sampler = sampler
print('integrating ...')
max_samples = int(os.environ.get('MAXSAMPLES', 0))
min_samples = int(os.environ.get('MINSAMPLES', 0))
results = multi_nested_integrator(tolerance=0.5, multi_sampler=sampler, min_samples=min_samples, max_samples=max_samples)
duration = time.time() - start_time
print('writing output files ...')
prefix = '%s_%s_nlive%d_%d.out8' % (sys.argv[1], constrainer_type, nlive_points, ndata)
# store results
with h5py.File(prefix + '.hdf5', 'w') as f:
	f.create_dataset('logZ', data=results['logZ'], compression='gzip', shuffle=True)
	f.create_dataset('logZerr', data=results['logZerr'], compression='gzip', shuffle=True)
	u, x, L, w, mask = list(zip(*results['weights']))
	f.create_dataset('u', data=u, compression='gzip', shuffle=True)
	f.create_dataset('x', data=x, compression='gzip', shuffle=True)
	f.create_dataset('L', data=L, compression='gzip', shuffle=True)
	f.create_dataset('w', data=w, compression='gzip', shuffle=True)
	f.create_dataset('mask', data=mask, compression='gzip', shuffle=True)
	f.create_dataset('ndraws', data=sampler.ndraws)
	print('logZ = %.1f +- %.1f' % (results['logZ'][0], results['logZerr'][0]))
	print('ndraws:', sampler.ndraws, 'niter:', len(w))

print('writing statistic ...')
json.dump(dict(ndraws=sampler.ndraws, duration=duration, ndata=ndata, niter=len(w)), 
	open(prefix + '.stats.json', 'w'), indent=4)
print('done.')


