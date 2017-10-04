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
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

do_plotting = False

print 'loading data...'
ndata = int(sys.argv[2])
f = pyfits.open(sys.argv[1])
datasection = f['DATA'] 
y = datasection.data # values
nspec, npixx, npixy = y.shape
noise_level = f['STAT'].data # variance

if do_plotting:
	print 'plotting image...'
	plt.figure(figsize=(20,20))
	plt.imshow(y[0,:,:])
	plt.savefig('musefuse_img0.png', bbox_inches='tight')
	plt.close()

print 'applying subselection ...'
y = y[:,80:200,170:240]
noise_level = noise_level[:,80:200,170:240]
y = y[:,70:80,35:45]
noise_level = noise_level[:,70:80,35:45]
#y = y[:,150:170,145:170]
#noise_level = noise_level[:,150:170,145:170]

if do_plotting:
	print 'plotting selection ...'
	plt.figure(figsize=(20,20))
	plt.imshow(y[0,:,:])
	plt.savefig('musefuse_sel_img0.png', bbox_inches='tight')
	plt.close()

nspec, npixx, npixy = y.shape

y = y.reshape((nspec, -1))
noise_level = noise_level.reshape((nspec, -1))
x = datasection.header['CD3_3'] * numpy.arange(nspec) + datasection.header['CRVAL3']
#good = numpy.logical_and(numpy.isfinite(noise_level).all(axis=0), numpy.isfinite(y).all(axis=0))
print '    finding NaNs...'
good = numpy.isfinite(noise_level).all(axis=0)
assert good.shape == (npixx*npixy,), good.shape
goodids = numpy.where(good)[0]
#numpy.random.shuffle(goodids)

print '    truncating data to %d sets...' % ndata, goodids[:ndata]
# truncate data
y = y[:,goodids[:ndata]]
noise_level = noise_level[:,goodids[:ndata]]
assert (noise_level>0).all(), noise_level

assert y.shape == (nspec, ndata), (y.shape, nspec, ndata)
assert noise_level.shape == (nspec, ndata)

#noise_level[noise_level > 2 * numpy.median(vd[:,i]] = 1000

print '    cleaning data'
noise_level2 = noise_level.copy()
w = 10
for j in range(nspec):
	lo = j - w
	hi = j + w
	if lo < 0:
		lo = 0
	if hi > nspec:
		hi = nspec
	seg = noise_level[lo:hi,:]
	med = numpy.median(seg, axis=0)
	diff = numpy.abs(med.reshape((1, -1)) - seg)
	meddiff = numpy.median(diff, axis=0)
	diff = numpy.abs(noise_level[j,:] - med)
	v = (diff > 5 * meddiff) * 1e10
	#k = j
	if v.any():
		print '    updating noise level at', j #, meddiff, diff
		for k in range(max(0, j-3), min(nspec-1, j+3)+1):
			noise_level2[k,:] += v

noise_level2[1600:1670,:] += 1e10
noise_level2[1730:1780,:] += 1e10
noise_level2[1950:2000,:] += 1e10

if do_plotting:
	for i in range(ndata):
		plt.figure()
		xi = numpy.arange(len(y[:,i]))
		plt.plot(xi, y[:,i], color='k', lw=2)
		sigma = noise_level[:,i]**0.5
		plt.fill_between(xi, y[:,i] - sigma, y[:,i] + sigma, alpha=0.3, color='red')
		sigma = noise_level2[:,i]**0.5
		plt.fill_between(xi, y[:,i] - sigma, y[:,i] + sigma, alpha=0.3, color='gray')
		idx = numpy.where(noise_level2[:,i] != noise_level[:,i])[0]
		print idx
		lo, hi = y[:,i].min(), y[:,i].max()
		plt.vlines(idx, lo, hi, color='g', alpha=0.1)
		plt.ylim(lo, hi)
		plt.xlim(500, 2000)
		plt.savefig('musefuse_data%d.pdf' % (i+1), bbox_inches='tight')
		plt.close()

#noise_level2[noise_level2 > noise_level.max()] = noise_level.max()
noise_level = noise_level2

"""

Definition of the problem
- parameter space (here: 3d)
- likelihood function which consists of 
  - model function ("slow predicting function")
  - data comparison

"""

params = ['O', 'Z', 'SFtau', 'SFage', 'EBV'] #, 'misfit']
nparams = len(params)

print 'loading model...', sys.argv[3]
with h5py.File(sys.argv[3]) as f:
	models = f['templates'].value
	wavelength = f['wavelength'].value
	assert numpy.allclose(x, wavelength)
	sftaus = numpy.log10(f['taus'].value)
	sfages = f['sfages'].value / 1e9
	Zs = numpy.log10([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05, 0.1])

nZ, nSFtau, nSFage, nspec2 = models.shape
assert nspec2 == nspec
models /= 1e-10 + models[:,:,:,2000].reshape((nZ, nSFtau, nSFage, 1)) # normalise somewhere to one

nspec = 1500
models = models[:,:,:,500:2000]
y = y[500:2000,:]
wavelength = wavelength[500:2000]
noise_level = noise_level[500:2000,:]

wavelength = wavelength / 10.
calzetti_result = numpy.zeros_like(wavelength)
mask = (wavelength < 630)
calzetti_result[mask] = 2.659 * (-2.156 + 1.509e3 / wavelength[mask] -
    0.198e6 / wavelength[mask] ** 2 +
    0.011e9 / wavelength[mask] ** 3) + 4.05

# Attenuation between 630 nm and 2200 nm
mask = (wavelength >= 630)
calzetti_result[mask] = 2.659 * (-1.857 + 1.040e3 / wavelength[mask]) + 4.05

import scipy.interpolate
#model_interp = scipy.interpolate.RegularGridInterpolator([numpy.arange(nZ), numpy.arange(nSFtau), numpy.arange(nSFage)], models)
model_interp = scipy.interpolate.RegularGridInterpolator([Zs, sftaus, sfages], models)

def model(Z, SFtau, SFage, EBV):
	#iZ = int(Z)
	#iSFtau = int(SFtau)
	#iSFage = int(SFage)
	#template = models[iZ, iSFtau, iSFage]
	#print 'requesting interpolation', Z, SFtau, SFage
	template = model_interp([Z, SFtau, SFage])[0]
	assert template.shape == (nspec,), template.shape
	# apply calzetti law
	exttemplate = template * 10**(-2.5 * calzetti_result * EBV)
	assert numpy.all(numpy.isfinite(exttemplate)), exttemplate
	return exttemplate

def priortransform(cube):
	# definition of the parameter width, by transforming from a unit cube
	cube = cube.copy()
	cube[0] = 10**(cube[0] * 4 - 2) # plateau
	cube[1] = cube[1] * (Zs.max() - Zs.min()) + Zs.min()
	cube[2] = cube[2] * (sftaus.max() - sftaus.min()) + sftaus.min()
	cube[3] = cube[3] * (sfages.max() - sfages.min()) + sfages.min()
	cube[4] = cube[4] * 2 # E(B-V)
	#cube[5] = cube[5] * 4 - 1 # misfit
	return cube

# the following is a python-only implementation of the likelihood 
# @ params are the parameters (as transformed by priortransform)
# @ data_mask is which data sets to consider.
# returns a likelihood vector
Lmax = -1e100
def multi_loglikelihood(params, data_mask):
	global Lmax
	O, Z, SFtau, SFage, EBV = params
	# predict the model
	ypred = model(Z, SFtau, SFage, EBV)
	# do the data comparison
	#print ypred.shape, y.shape, data_mask
	ndata = data_mask.sum()
	if (ypred == 0).all():
		# give low probability to solutions with no stars
		return numpy.ones(ndata) * -1e100
	ypred += O
	
	yd = y[:,data_mask]
	vd = noise_level[:,data_mask] #+ 10**logvar
	#vd[vd > 2 * numpy.median(vd)] = 1000
	
	# simple likelihood, would need a normalisation factor:
	# L = -0.5 * numpy.nansum((ypred.reshape((-1,1)) - yd)**2/vd, axis=0)
	L = numpy.zeros(ndata)
	
	for i in numpy.arange(ndata):
		# scaled likelihood, like LePhare
		# s = sum[OjMj/sigmaj^2] / sum[Mj^2/sigmaj^2]
		s = numpy.nansum(yd[:,i] * ypred / vd[:,i]) / (numpy.nansum(ypred**2 / vd[:,i]) + 1e-10)
		assert numpy.isfinite(s), (s, ypred, ypred**2, yd[:,i], vd[:,i])
		# chi2 = sum[(Oi - s*Mi)^2 / sigmai^2]
		chi2 = numpy.nansum((yd[:,i] - s * ypred)**2 / vd[:,i]) # + numpy.log(2*numpy.pi*vd))
		L[i] = -0.5 * chi2 + numpy.random.uniform() * 1e-5
		if i == 0 and L[i] > Lmax:
			Lmax = L[i]
			print 'plotting...'
			plt.figure(figsize=(20,20))
			plt.subplot(3, 1, 1)
			plt.title(str(params) + ' : chi2:' + str(chi2))
			#mask = vd[:,i] < 2 * numpy.median(vd[:,i])
			#mask = numpy.isfinite(vd[:,i])
			mask = Ellipsis
			plt.plot(yd[mask,i])
			plt.plot(s * ypred[mask])
			plt.ylim(yd[mask,i].min(), yd[mask,i].max())
			plt.subplot(3, 1, 2)
			plt.plot(ypred[mask])
			plt.subplot(3, 1, 3)
			plt.plot(vd[mask,i])
			plt.savefig('musefuse.pdf', bbox_inches='tight')
			plt.close()
			time.sleep(0.1)
		#print chi2
	assert L.shape == (ndata,), (L.shape, ypred.shape, y.shape, data_mask)
	return L

def multi_loglikelihood_vectorized(params, data_mask):
	O, Z, SFtau, SFage, EBV = params
	# predict the model
	ypred = model(Z, SFtau, SFage, EBV)
	# do the data comparison
	ndata = data_mask.sum()
	if (ypred == 0).all():
		# give low probability to solutions with no stars
		return numpy.ones(ndata) * -1e100
	ypred += O
	
	yd = y[:,data_mask]
	vd = noise_level[:,data_mask]
	assert numpy.isfinite(yd).all()
	assert numpy.isfinite(vd).all()
	assert numpy.isfinite(ypred).all()
	
	ypreds = ypred.reshape((-1,1))
	s = numpy.sum(yd * ypreds / vd, axis=0) / (numpy.sum(ypreds**2 / vd, axis=0) + 1e-10)
	assert s.shape == (ndata,), s.shape
	assert numpy.isfinite(s).all()
	chi2 = numpy.sum((yd - s.reshape((1,-1)) * ypreds)**2 / vd, axis=0)
	L = -0.5 * chi2 + numpy.random.uniform() * 1e-5
	
	assert L.shape == (ndata,), (L.shape, ypred.shape, y.shape, data_mask.sum())
	return L

if False:
	print 'testing vectorised code...'
	data_mask_all = numpy.ones(ndata) == 1
	for i in range(100):
		cube = numpy.random.uniform(size=nparams)
		params = priortransform(cube)
		L = multi_loglikelihood(params, data_mask_all)
		L2 = multi_loglikelihood_vectorized(params, data_mask_all)
		assert numpy.allclose(L, L2), (L, L2, cube, params)

multi_loglikelihood = multi_loglikelihood_vectorized

"""

After defining the problem, we use generic code to set up 
- Nested Sampling (Multi)Integrator
- Our special sampler
- RadFriends (constrained region draw)

We start with the latter.
"""


from multi_nested_integrator import multi_nested_integrator
from multi_nested_sampler import MultiNestedSampler
from hiermetriclearn import MetricLearningFriendsConstrainer
from cachedconstrainer import CachedConstrainer, generate_individual_constrainer

superset_constrainer = MetricLearningFriendsConstrainer(metriclearner = 'truncatedscaling', 
	rebuild_every=1000, metric_rebuild_every=20, verbose=False, force_shrink=True)

cc = CachedConstrainer()
focusset_constrainer = cc.get
_, _, individual_draw_constrained = generate_individual_constrainer()
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
print 'integrating ...'
results = multi_nested_integrator(tolerance=0.5, multi_sampler=sampler, min_samples=0) #, max_samples=5000)
duration = time.time() - start_time
print 'writing output files ...'
# store results
with h5py.File(sys.argv[1] + '.out_%d.hdf5' % ndata, 'w') as f:
	f.create_dataset('logZ', data=results['logZ'], compression='gzip', shuffle=True)
	f.create_dataset('logZerr', data=results['logZerr'], compression='gzip', shuffle=True)
	u, x, L, w, mask = zip(*results['weights'])
	f.create_dataset('u', data=u, compression='gzip', shuffle=True)
	f.create_dataset('x', data=x, compression='gzip', shuffle=True)
	f.create_dataset('L', data=L, compression='gzip', shuffle=True)
	f.create_dataset('w', data=w, compression='gzip', shuffle=True)
	f.create_dataset('mask', data=mask, compression='gzip', shuffle=True)
	f.create_dataset('ndraws', data=sampler.ndraws)
	f.create_dataset('fiberids', data=goodids[:ndata], compression='gzip', shuffle=True)
	
	print 'logZ = %.1f +- %.1f' % (results['logZ'][0], results['logZerr'][0])
	print 'ndraws:', sampler.ndraws, 'niter:', len(w)

print 'writing statistic ...'
json.dump(dict(ndraws=sampler.ndraws, duration=duration, ndata=ndata, niter=len(w)), 
	open(sys.argv[1] + '_%d.out7.stats.json' % ndata, 'w'), indent=4)
print 'done.'


