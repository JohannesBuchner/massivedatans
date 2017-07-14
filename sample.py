import numpy
from numpy import exp
import h5py
import sys
import json
import os
import time

print 'loading data...'
ndata = int(sys.argv[2])
with h5py.File(sys.argv[1], 'r') as f:
	x = numpy.array(f['x'].value)
	y = numpy.array(f['y'][:,:ndata])

def gauss(x, z, A, mu, sig):
	return A * exp(-0.5 * ((mu - x / (1. + z))/sig)**2)

nx, ndata = y.shape
noise_level = 0.01
params = ['A', 'mu', 'sig'] #, 'noise_level']
nparams = len(params)
def priortransform(cube):
	cube = cube.copy()
	cube[0] = 10**(cube[0] * 2 - 2)
	cube[1] = cube[1] * 400 + 400
	cube[2] = cube[2] * 2
	return cube

def model(params):
	A, mu, log_sig_kms = params
	sig = 10**log_sig_kms
	z = 0
	ypred = gauss(x, z, A, mu, sig)
	return ypred

def multi_loglikelihood(params, data_mask):
	A, mu, log_sig_kms = params
	sig = 10**log_sig_kms
	ypred = A * exp(-0.5 * ((mu - x)/sig)**2)
	L = -0.5 * (((ypred.reshape((-1,1)) - y[:,data_mask])/noise_level)**2).sum(axis=0)
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
	sig = 10**log_sig_kms
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

superset_constrainer = MetricLearningFriendsConstrainer(metriclearner = 'truncatedscaling', 
	rebuild_every=1000, metric_rebuild_every=20, verbose=False, force_shrink=True)
class CachedConstrainer(object):
	"""
	This keeps metric learners if they are used (in the last three iterations).
	Otherwise, constructs a fresh one.
	"""
	def __init__(self):
		self.iter = 0
		self.prev_prev_prev_generation = {}
		self.prev_prev_generation = {}
		self.prev_generation = {}
		self.curr_generation = {}
		self.last_mask = []
		self.last_points = []
		self.last_realmask = None
	
	def get(self, mask, realmask, points, it):
		while self.iter < it:
			# new generation
			self.prev_prev_prev_generation = self.prev_prev_generation
			self.prev_prev_generation = self.prev_generation
			self.prev_generation = self.curr_generation
			self.curr_generation = {}
			self.last_mask = []
			self.last_realmask = None
			self.last_points = []
			self.iter += 1
		
		# if we only dropped a single (or a few) data sets
		# compared to the call just before, lets reuse the same
		# this happens in the focussed draw with 1000s of data sets
		# where a single data set can accept a point; 
		# not worth to recompute the region.
		if self.last_realmask is not None and len(mask) < len(self.last_mask) and \
			len(mask) > 0.80 * len(self.last_mask) and \
			len(points) <= len(self.last_points) and \
			len(points) > 0.90 * len(self.last_points) and \
			numpy.mean(self.last_realmask == realmask) > 0.80 and \
			numpy.in1d(points, self.last_points).all():
			print 're-using previous, similar region (%.1f%% data set overlap, %.1f%% points overlap)' % (numpy.mean(self.last_realmask == realmask) * 100., len(points) * 100. / len(self.last_points), )
			k = tuple(self.last_mask.tolist())
			return self.curr_generation[k].draw_constrained
		print 'not re-using region', (len(mask), len(self.last_mask), len(points), len(self.last_points), (len(mask) < len(self.last_mask), len(mask) > 0.80 * len(self.last_mask), len(points) > 0.90 * len(self.last_points), numpy.mean(self.last_realmask == realmask) ) )
		
		# normal operation:
		k = tuple(mask.tolist())
		self.last_realmask = realmask
		self.last_mask = mask
		self.last_points = points
		
		# try to recycle
		if k in self.curr_generation:
			pass
		elif k in self.prev_generation:
			print 're-using previous1 region'
			self.curr_generation[k] = self.prev_generation[k]
		elif k in self.prev_prev_generation:
			print 're-using previous2 region'
			self.curr_generation[k] = self.prev_prev_generation[k]
		elif k in self.prev_prev_prev_generation:
			print 're-using previous3 region'
			self.curr_generation[k] = self.prev_prev_prev_generation[k]
		else:
			# nothing found, so start from scratch
			self.curr_generation[k] = MetricLearningFriendsConstrainer(
				metriclearner = 'truncatedscaling', force_shrink=True,
				rebuild_every=1000, metric_rebuild_every=20, 
				verbose=False)
			self.curr_generation[k].sampler = sampler
		
		return self.curr_generation[k].draw_constrained

focusset_constrainer = CachedConstrainer().get
individual_constrainers = {}
individual_constrainers_lastiter = {}
def individual_draw_constrained(i, it):
	if i not in individual_constrainers:
		individual_constrainers[i] = MetricLearningFriendsConstrainer(
			metriclearner = 'truncatedscaling', force_shrink=True,
			rebuild_every=1000, metric_rebuild_every=20, 
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
	draw_constrained = focusset_constrainer, 
	nsuperset_draws = int(os.environ.get('SUPERSET_DRAWS', '10')),
	use_graph = os.environ.get('USE_GRAPH', '1') == '1'
	)
superset_constrainer.sampler = sampler
print 'integrating ...'
results = multi_nested_integrator(tolerance=0.5, multi_sampler=sampler, min_samples=0) #, max_samples=1000)
duration = time.time() - start_time
print 'writing output files ...'
# store results
with h5py.File(sys.argv[1] + '.out7.hdf5', 'w') as f:
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
	print 'ndraws:', sampler.ndraws, 'niter:', len(w)

print 'writing statistic ...'
json.dump(dict(ndraws=sampler.ndraws, duration=duration, ndata=ndata, niter=len(w)), 
	open(sys.argv[1] + '_%d.out7.stats.json' % ndata, 'w'), indent=4)
print 'done.'


