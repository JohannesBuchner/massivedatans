import numpy
from numpy import exp
import h5py
import sys
import os

print 'loading data...'
f = h5py.File(sys.argv[1], 'r')
x = f['x'].value
y = f['y'].value

def gauss(x, z, A, mu, sig):
	return A * exp(-0.5 * ((mu - x / (1. + z))/sig)**2)

nx, ndata = y.shape
rest_wave = 440
noise_level = 0.4
params = ['A', 'mu', 'sig']
nparams = len(params)
def priortransform(cube):
	cube = cube.copy()
	cube[0] = cube[0] * 10
	cube[1] = cube[1] * 400 + 400
	cube[2] = cube[2] * 5
	return cube

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
	return -0.5 * Lout

#print multi_loglikelihood([0.88091237,  444.44207558,    2.77671952], numpy.ones(ndata)==1)
#print multi_loglikelihood([1.65758829e-01, 4.45518543e+02, 3.25894638e+00], numpy.ones(ndata)==1)
#print multi_loglikelihood([0.95572931,  443.99407818,    2.95764509], numpy.ones(ndata)==1)

from multi_nested_integrator import multi_nested_integrator
from multi_nested_sampler import MultiNestedSampler
from friends import FriendsConstrainer

numpy.random.seed(1)
print 'setting up integrator ...'
superset_constrainer = FriendsConstrainer(radial = True, metric = 'euclidean', jackknife=True, 
	rebuild_every=10, verbose=False)
focusset_constrainer = FriendsConstrainer(radial = True, metric = 'euclidean', jackknife=True, 
	rebuild_every=1, verbose=False)
sampler = MultiNestedSampler(nlive_points = 400, 
	priortransform=priortransform, multi_loglikelihood=multi_loglikelihood, 
	ndim=nparams, ndata=ndata,
	superset_draw_constrained = superset_constrainer.draw_constrained, 
	draw_constrained = focusset_constrainer.draw_constrained, 
	nsuperset_draws = int(os.environ.get('SUPERSET_DRAWS', '10'))
	)
focusset_constrainer.sampler = sampler
superset_constrainer.sampler = sampler
print 'integrating ...'
results = multi_nested_integrator(tolerance=0.2, multi_sampler=sampler, min_samples=500) #, max_samples=1000)

print 'writing output files ...'
# store results
with h5py.File(sys.argv[1] + '.out.hdf5', 'w') as f:
	f.create_dataset('logZ', data=results['logZ'], compression='gzip', shuffle=True)
	f.create_dataset('logZerr', data=results['logZerr'], compression='gzip', shuffle=True)
	u, x, L, w = zip(*results['weights'])
	f.create_dataset('u', data=u, compression='gzip', shuffle=True)
	f.create_dataset('x', data=x, compression='gzip', shuffle=True)
	f.create_dataset('L', data=L, compression='gzip', shuffle=True)
	f.create_dataset('w', data=w, compression='gzip', shuffle=True)
	f.create_dataset('ndraws', data=sampler.ndraws)
	print 'logZ = %.1f +- %.1f' % (results['logZ'][0], results['logZerr'][0])
	print 'ndraws:', sampler.ndraws


