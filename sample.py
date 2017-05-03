import numpy
from numpy import exp
import h5py
import sys
f = h5py.File(sys.argv[1])
x = f['x'].value
y = f['y'].value

def gauss(x, z, A, mu, sig):
	return A * exp(-0.5 * ((mu - x / (1. + z))/sig)**2)

rest_wave = 440
noise_level = 0.01
params = ['A', 'mu', 'sig']
nparams = len(params)
def priortransform(cube):
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

def loglikelihood(params):
	ypred = model(params)
	return ((ypred - y.reshape((1, -1)))/noise_level)**2


from nested_sampling.nested_integrator import nested_integrator
from nested_sampling.nested_sampler import NestedSampler
from nested_sampling.samplers.rejection import RejectionConstrainer
from nested_sampling.samplers.friends import FriendsConstrainer

constrainer = FriendsConstrainer(radial = True, metric = 'euclidean', jackknife=True)
sampler = NestedSampler(nlive_points = 400, 
	priortransform=priortransform, loglikelihood=loglikelihood, 
	draw_constrained = constrainer.draw_constrained, ndim=2)
constrainer.sampler = sampler
results = nested_integrator(tolerance=0.1, sampler=sampler)
print results

