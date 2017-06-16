import numpy
from numpy import exp
import h5py
import sys
print 'loading data...'
f = h5py.File(sys.argv[1], 'r')
x = f['x'].value
y = f['y'].value[:,0]

def gauss(x, z, A, mu, sig):
	return A * exp(-0.5 * ((mu - x / (1. + z))/sig)**2)

nx = y.shape
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
def priortransform(cube):
	cube = cube.copy()
	cube[0] = 10**(cube[0] * 10 - 5)
	cube[1] = cube[1] * 400 + 400
	cube[2] = 10**(cube[0] * 10 - 5)
	return cube

def model(params):
	A, mu, log_sig_kms = params
	sig = 10**log_sig_kms * rest_wave / 300000
	z = 0
	ypred = gauss(x, z, A, mu, sig)
	return ypred

def loglikelihood(params):
	#print 'Like: model'
	ypred = model(params)
	#print 'Like: data-model'
	L = -0.5 * (((ypred - y)/noise_level)**2).sum()
	assert numpy.isfinite(L), L
	#assert L.shape == (data_mask.sum(),), (L.shape, ndata)
	return L

numpy.random.seed(1)
from nested_sampling.nested_integrator import nested_integrator
from nested_sampling.nested_sampler import NestedSampler
from nested_sampling.samplers.rejection import RejectionConstrainer
from nested_sampling.samplers.friends import FriendsConstrainer
import nested_sampling.postprocess as post
print 'setting up integrator ...'
#constrainer = RejectionConstrainer()
constrainer = FriendsConstrainer(radial = True, metric = 'euclidean', jackknife=True)
sampler = NestedSampler(nlive_points = 400, 
	priortransform=priortransform, loglikelihood=loglikelihood, 
	draw_constrained = constrainer.draw_constrained, ndim=nparams)
constrainer.sampler = sampler
results = nested_integrator(tolerance=0.2, sampler=sampler)

print 'writing output files ...'
# store results
with h5py.File(sys.argv[1] + '.outsingle.hdf5', 'w') as f:
	f.create_dataset('logZ', data=results['logZ'])
	f.create_dataset('logZerr', data=results['logZerr'])
	u, x, L, w = zip(*results['weights'])
	f.create_dataset('u', data=u, compression='gzip', shuffle=True)
	f.create_dataset('x', data=x, compression='gzip', shuffle=True)
	f.create_dataset('L', data=L, compression='gzip', shuffle=True)
	f.create_dataset('w', data=w, compression='gzip', shuffle=True)
	f.create_dataset('ndraws', data=sampler.ndraws)


