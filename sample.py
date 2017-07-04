import numpy
from numpy import exp
import h5py
import sys
import json
import os
import time

print 'loading data...'
f = h5py.File(sys.argv[1], 'r')
x = f['x'].value
y = f['y'].value

def gauss(x, z, A, mu, sig):
	return A * exp(-0.5 * ((mu - x / (1. + z))/sig)**2)

nx, ndata = y.shape
rest_wave = 440
noise_level = 0.01
params = ['A', 'mu', 'sig'] #, 'noise_level']
nparams = len(params)
def priortransform(cube):
	cube = cube.copy()
	cube[0] = 10**(cube[0] * 2 - 2)
	#cube[0] = cube[0] * 10
	cube[1] = cube[1] * 400 + 400
	cube[2] = cube[2] * 2 + 2
	##cube[3] = 10**(cube[3] * 2 - 4)
	return cube

#def priortransform(cube):
#	cube = cube.copy()
#	cube[0] = 10**(cube[0] * 10 - 5)
#	cube[1] = cube[1] * 400 + 400
#	cube[2] = cube[0] * 10 - 5
#	return cube

def model(params):
	A, mu, log_sig_kms = params
	sig = 10**log_sig_kms * rest_wave / 300000
	z = 0
	ypred = gauss(x, z, A, mu, sig)
	return ypred

def multi_loglikelihood(params, data_mask):
	#print 'Like: model'
	
	#ypred = model(params)
	# inlining for speed-up
	A, mu, log_sig_kms = params
	sig = 10**log_sig_kms * rest_wave / 300000
	ypred = A * exp(-0.5 * ((mu - x)/sig)**2)
	# end of inlining
	
	#print 'Like: data-model'
	L = -0.5 * (((ypred.reshape((-1,1)) - y[:,data_mask])/noise_level)**2).sum(axis=0)
	#print 'Like done'
	#assert L.shape == (data_mask.sum(),), (L.shape, ndata)
	return L

#print multi_loglikelihood([0.88091237,  444.44207558,    2.77671952], numpy.ones(ndata)==1)
#print multi_loglikelihood([1.65758829e-01, 4.45518543e+02, 3.25894638e+00], numpy.ones(ndata)==1)
#print multi_loglikelihood([0.95572931,  443.99407818,    2.95764509], numpy.ones(ndata)==1)

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

def multi_loglikelihood(params, data_mask):
	A, mu, log_sig_kms = params
	sig = 10**log_sig_kms * rest_wave / 300000
	Lout = numpy.zeros(data_mask.sum())
	ret = lib.like(x, y, ndata, nx, A, mu, sig, noise_level, data_mask, Lout)
	#assert ret == 0, (ret, -0.5*Lout)
	assert numpy.isfinite(Lout).all(), (Lout, params)
	return -0.5 * Lout

#print multi_loglikelihood([0.88091237,  444.44207558,    2.77671952], numpy.ones(ndata)==1)
#print multi_loglikelihood([1.65758829e-01, 4.45518543e+02, 3.25894638e+00], numpy.ones(ndata)==1)
#print multi_loglikelihood([0.95572931,  443.99407818,    2.95764509], numpy.ones(ndata)==1)

from multi_nested_integrator import multi_nested_integrator
from multi_nested_sampler import MultiNestedSampler
from hiermetriclearn import MetricLearningFriendsConstrainer

numpy.random.seed(1)
start_time = time.time()
print 'setting up integrator ...'
nlive_points = int(os.environ.get('NLIVE_POINTS','400'))
superset_constrainer = MetricLearningFriendsConstrainer(metriclearner = 'truncatedscaling', 
	rebuild_every=20, metric_rebuild_every=20, verbose=False, force_shrink=True)
focusset_constrainer = MetricLearningFriendsConstrainer(metriclearner = 'truncatedscaling', 
	rebuild_every=1, metric_rebuild_every=1, verbose=False)
individual_constrainers = {}
individual_constrainers_lastiter = {}
def individual_draw_constrained(i, it):
	if i not in individual_constrainers:
		individual_constrainers[i] = MetricLearningFriendsConstrainer(
			metriclearner = 'truncatedscaling', force_shrink=True,
			rebuild_every=5, metric_rebuild_every=5, 
			verbose=False)
		individual_constrainers[i].sampler = sampler
		individual_constrainers_lastiter[i] = it
	if it > individual_constrainers_lastiter[i] + 5:
		# force rebuild
		individual_constrainers[i].region = None
	individual_constrainers_lastiter[i] = it
	return individual_constrainers[i].draw_constrained

sampler = MultiNestedSampler(nlive_points = nlive_points, 
	priortransform=priortransform, multi_loglikelihood=multi_loglikelihood, 
	ndim=nparams, ndata=ndata,
	superset_draw_constrained = superset_constrainer.draw_constrained, 
	individual_draw_constrained = individual_draw_constrained,
	draw_constrained = focusset_constrainer.draw_constrained, 
	nsuperset_draws = int(os.environ.get('SUPERSET_DRAWS', '10')),
	use_graph = os.environ.get('USE_GRAPH', '1') == '1'
	)
focusset_constrainer.sampler = sampler
superset_constrainer.sampler = sampler
print 'integrating ...'
results = multi_nested_integrator(tolerance=0.5, multi_sampler=sampler, min_samples=0) #, max_samples=1000)
duration = time.time() - start_time
print 'writing output files ...'
# store results
with h5py.File(sys.argv[1] + '.out.hdf5', 'w') as f:
	f.create_dataset('logZ', data=results['logZ'], compression='gzip', shuffle=True)
	f.create_dataset('logZerr', data=results['logZerr'], compression='gzip', shuffle=True)
	u, x, L, w, mask = zip(*results['weights'])
	f.create_dataset('u', data=u, compression='gzip', shuffle=True)
	f.create_dataset('x', data=x, compression='gzip', shuffle=True)
	f.create_dataset('L', data=L, compression='gzip', shuffle=True)
	f.create_dataset('w', data=w, compression='gzip', shuffle=True)
	f.create_dataset('mask', data=mask, compression='gzip', shuffle=True)
	f.create_dataset('ndraws', data=sampler.ndraws)
	print 'logZ = %.1f +- %.1f' % (results['logZ'][0], results['logZerr'][0])
	print 'ndraws:', sampler.ndraws

print 'writing statistic ...'
json.dump(dict(ndraws=sampler.ndraws, duration=duration), 
	open(sys.argv[1] + '.out.stats.json', 'w'))
print 'done.'


