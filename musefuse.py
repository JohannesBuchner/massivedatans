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
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

do_plotting = False

print('loading data...')
f = pyfits.open(sys.argv[1])
datasection = f['DATA'] 
y = datasection.data # values
y = y[:3600,:,:]
nspec, npixx, npixy = y.shape
noise_level = f['STAT'].data # variance
noise_level = noise_level[:3600,:,:]
good = numpy.isfinite(noise_level).all(axis=0)
print('   %.2f%% good...' % (100*good.mean()))
#print numpy.where(~numpy.isfinite(noise_level[:,40,40]))
#print noise_level[~numpy.isfinite(noise_level[:,40,40]),40,40]

if do_plotting:
	print('plotting image...')
	plt.figure(figsize=(20,20))
	plt.imshow(y[0,:,:])
	plt.savefig('musefuse_img0.png', bbox_inches='tight')
	plt.close()

regionfile = sys.argv[2]
import pyregion
region = pyregion.parse(open(regionfile).read())
mask = region.get_mask(shape=(npixx, npixy))

maskx = mask.any(axis=0)
masky = mask.any(axis=1)
i = numpy.where(maskx)[0]
ilo, ihi = i.min(), i.max() + 1
j = numpy.where(masky)[0]
jlo, jhi = j.min(), j.max() + 1
print((mask.sum(), ilo, ihi, jlo, jhi, y.shape, npixx, npixy))
#ndata = mask.sum()

#ymask = mask.reshape((1, npixx, npixy))
ymask = numpy.array([mask] * len(y))
y[~ymask] = numpy.nan
if do_plotting:
	print('plotting selection ...')
	plt.figure(figsize=(20,20))
	plt.imshow(y[0,ilo:ihi,jlo:jhi])
	plt.colorbar()
	plt.savefig('musefuse_sel_img0.png', bbox_inches='tight')
	plt.close()

print('applying subselection ...')
y = y[ymask]
noise_level = noise_level[ymask]
print('    subselection gave %s ...' % (y.shape))
y = y.reshape((nspec, -1))
noise_level = noise_level.reshape((nspec, -1))
x = datasection.header['CD3_3'] * numpy.arange(nspec) + datasection.header['CRVAL3']
wavelength = x
#good = numpy.logical_and(numpy.isfinite(noise_level).all(axis=0), numpy.isfinite(y).all(axis=0))
print('    finding NaNs...')
good = numpy.isfinite(noise_level).all(axis=0)
print('    found %d finite spaxels ...' % (good.sum()))
#assert good.shape == (ymask.sum(),), good.shape
goodids = numpy.where(good)[0]
#numpy.random.shuffle(goodids)

ndata = int(os.environ.get('MAXDATA', len(goodids)))
print('    truncating data to %d sets...' % ndata, goodids[:ndata])
## truncate data
y = y[:,goodids[:ndata]]
noise_level = noise_level[:,goodids[:ndata]]
assert (noise_level>0).all(), noise_level

assert y.shape == (nspec, ndata), (y.shape, nspec, ndata)
assert noise_level.shape == (nspec, ndata)

assert ndata > 0, 'No valid data!?'

#noise_level[noise_level > 2 * numpy.median(vd[:,i]] = 1000

print('    cleaning data')
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
	if False and v.any():
		print('    updating noise level at', j) #, meddiff, diff
		for k in range(max(0, j-3), min(nspec-1, j+3)+1):
			noise_level2[k,:] += v

noise_level2[1600:1670,:] += 1e10
noise_level2[1730:1780,:] += 1e10
noise_level2[1950:2000,:] += 1e10
noise_level2[1750+500:2200+500,:] += 1e10
noise_level2[2300+500:2500+500,:] += 1e10
#noise_level2[noise_level2 > noise_level.max()] = noise_level.max()

if do_plotting:
	for i in range(ndata):
		plt.figure()
		xi = numpy.arange(len(y[:,i]))
		plt.plot(xi, y[:,i], color='k', lw=1)
		sigma0 = noise_level[:,i]**0.5
		plt.fill_between(xi, y[:,i] - sigma0, y[:,i] + sigma0, alpha=0.3, color='red')
		sigma = noise_level2[:,i]**0.5
		plt.fill_between(xi, y[:,i] - sigma, y[:,i] + sigma, alpha=0.3, color='gray')
		idx = numpy.where(noise_level2[:,i] != noise_level[:,i])[0]
		lo, hi = y[:,i].min(), y[:,i].max()
		plt.plot(xi, lo+sigma0, color='b')
		plt.plot(xi, lo+0*sigma0, color='b')
		plt.vlines(idx, lo, hi, color='g', alpha=0.1, lw=0.1)
		plt.ylim(lo, hi)
		#plt.xlim(500, 3500)
		plt.savefig('musefuse_data%d.pdf' % (i+1), bbox_inches='tight')
		plt.close()

noise_level = noise_level2

"""

Definition of the problem
- parameter space (here: 3d)
- likelihood function which consists of 
  - model function ("slow predicting function")
  - data comparison

"""

paramnames = ['Z', 'logSFtau', 'SFage', 'z', 'EBV'] #, 'misfit']
nparams = len(paramnames)

zlo = float(sys.argv[3])
zhi = float(sys.argv[4])
filenames = sys.argv[5:]
grid = []

for iZ, filename in enumerate(filenames):
	print(filename)
	data = numpy.loadtxt(filename)
	model_wavelength = data[:,0]
	model_templates = data[:,1:].transpose()
	grid.append(model_templates)

inversewavelength_grid = numpy.linspace(1/10000., 1/4000., 2000)
# sigma is applied on that grid
# to convert to km/s, we need the wavelength, e.g. at 4000 and the element size
inversewavelength_gridwidth_A = 0.24 / 5 # A at 4000 (the end of this grid)

Zs = numpy.log10([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05, 0.1])
sftaus = numpy.log10(numpy.array([1, 4, 10, 40, 100, 400, 1000, 4000]) * 1.e6)
sfages = numpy.linspace(0, 13, 26)
ages = numpy.array([0.000E+00, 1.000E+05, 1.412E+05, 1.585E+05, 1.778E+05, 1.995E+05, 2.239E+05, 2.512E+05, 2.818E+05, 3.162E+05, 3.548E+05, 3.981E+05, 4.467E+05, 5.012E+05, 5.623E+05, 6.310E+05, 7.080E+05, 7.943E+05, 8.913E+05, 1.000E+06, 1.047E+06, 1.096E+06, 1.148E+06, 1.202E+06, 1.259E+06, 1.318E+06, 1.380E+06, 1.445E+06, 1.514E+06, 1.585E+06, 1.660E+06, 1.738E+06, 1.820E+06, 1.906E+06, 1.995E+06, 2.089E+06, 2.188E+06, 2.291E+06, 2.399E+06, 2.512E+06, 2.630E+06, 2.754E+06, 2.884E+06, 3.020E+06, 3.162E+06, 3.311E+06, 3.467E+06, 3.631E+06, 3.802E+06, 3.981E+06, 4.169E+06, 4.365E+06, 4.571E+06, 4.786E+06, 5.012E+06, 5.248E+06, 5.495E+06, 5.754E+06, 6.026E+06, 6.310E+06, 6.607E+06, 6.918E+06, 7.244E+06, 7.586E+06, 7.943E+06, 8.318E+06, 8.710E+06, 9.120E+06, 9.550E+06, 1.000E+07, 1.047E+07, 1.096E+07, 1.148E+07, 1.202E+07, 1.259E+07, 1.318E+07, 1.380E+07, 1.445E+07, 1.514E+07, 1.585E+07, 1.660E+07, 1.738E+07, 1.820E+07, 1.906E+07, 1.995E+07, 2.089E+07, 2.188E+07, 2.291E+07, 2.399E+07, 2.512E+07, 2.630E+07, 2.754E+07, 2.900E+07, 3.000E+07, 3.100E+07, 3.200E+07, 3.300E+07, 3.400E+07, 3.500E+07, 3.600E+07, 3.700E+07, 3.800E+07, 3.900E+07, 4.000E+07, 4.250E+07, 4.500E+07, 4.750E+07, 5.000E+07, 5.250E+07, 5.500E+07, 5.709E+07, 6.405E+07, 7.187E+07, 8.064E+07, 9.048E+07, 1.015E+08, 1.139E+08, 1.278E+08, 1.434E+08, 1.609E+08, 1.805E+08, 2.026E+08, 2.273E+08, 2.550E+08, 2.861E+08, 3.210E+08, 3.602E+08, 4.042E+08, 4.535E+08, 5.088E+08, 5.709E+08, 6.405E+08, 7.187E+08, 8.064E+08, 9.048E+08, 1.015E+09, 1.139E+09, 1.278E+09, 1.434E+09, 1.609E+09, 1.680E+09, 1.700E+09, 1.800E+09, 1.900E+09, 2.000E+09, 2.100E+09, 2.200E+09, 2.300E+09, 2.400E+09, 2.500E+09, 2.600E+09, 2.750E+09, 3.000E+09, 3.250E+09, 3.500E+09, 3.750E+09, 4.000E+09, 4.250E+09, 4.500E+09, 4.750E+09, 5.000E+09, 5.250E+09, 5.500E+09, 5.750E+09, 6.000E+09, 6.250E+09, 6.500E+09, 6.750E+09, 7.000E+09, 7.250E+09, 7.500E+09, 7.750E+09, 8.000E+09, 8.250E+09, 8.500E+09, 8.750E+09, 9.000E+09, 9.250E+09, 9.500E+09, 9.750E+09, 1.000E+10, 1.025E+10, 1.050E+10, 1.075E+10, 1.100E+10, 1.125E+10, 1.150E+10, 1.175E+10, 1.200E+10, 1.225E+10, 1.250E+10, 1.275E+10, 1.300E+10, 1.325E+10, 1.350E+10, 1.375E+10, 1.400E+10, 1.425E+10, 1.450E+10, 1.475E+10, 1.500E+10, 1.525E+10, 1.550E+10, 1.575E+10, 1.600E+10, 1.625E+10, 1.650E+10, 1.675E+10, 1.700E+10, 1.725E+10, 1.750E+10, 1.775E+10, 1.800E+10, 1.825E+10, 1.850E+10, 1.875E+10, 1.900E+10, 1.925E+10, 1.950E+10, 1.975E+10, 2.000E+10])[::2]

nZ = len(Zs)
nSFage = len(sfages)
nSFtau = len(sftaus)
#nspec2 = models.shape
#assert nspec2 == nspec
#models /= 1e-10 + models[:,:,:,2000].reshape((nZ, nSFage, nSFtau, 1)) # normalise somewhere to one

"""
nspec = 3000
#models = models[:,:,:,500:3500]
y = y[500:3500,:]
wavelength = wavelength[500:3500]
noise_level = noise_level[500:3500,:]
"""
y = y.astype(numpy.float64).copy()
noise_level = noise_level.astype(numpy.float64).copy()

wavelength = wavelength / 10.
model_wavelength = model_wavelength / 10.
calzetti_result = numpy.zeros_like(model_wavelength)
mask = (model_wavelength < 630)
calzetti_result[mask] = 2.659 * (-2.156 + 1.509e3 / model_wavelength[mask] -
    0.198e6 / model_wavelength[mask] ** 2 +
    0.011e9 / model_wavelength[mask] ** 3) + 4.05

# Attenuation between 630 nm and 2200 nm
mask = (model_wavelength >= 630)
calzetti_result[mask] = 2.659 * (-1.857 + 1.040e3 / model_wavelength[mask]) + 4.05

import scipy.interpolate, scipy.ndimage

def model(Z, SFtau, sfage, z, EBV):
	iZ = numpy.where(Zs <= Z)[-1][-1]
	#print('   selecting Z: %d' % iZ) 
	model_templates = grid[iZ]
	#print('   template max value:', model_templates.max(), model_templates.shape)
	assert numpy.all(model_templates>=0), model_templates
	# convolve the template
	
	# SFage = 0-13 (Gyrs).
	#print('   selecting sfage: %.2f' % sfage) 
	# ----123456789SFage________ --age-->
	tsinceSF = sfage * 1.e9 - ages
	tsinceSF[tsinceSF <= 0] = 0
	# star formation history is a (delayed) exponential decline.
	SFtau = float(SFtau)
	#print('   selecting SFtau: %.2f' % SFtau) 
	sfh = tsinceSF / SFtau**2 * numpy.exp(-tsinceSF/SFtau)
	sfh /= sfh.max()
	assert numpy.all(sfh>=0), sfh
	#print('   ages: ', ages)
	#print('   tsinceSF: ', tsinceSF)
	#print('   sfh: ', sfh)
	# before sfage, no stars
	age_weight = ages[1:] - ages[:-1]
	assert numpy.all(age_weight>=0), age_weight
	
	# weight stellar templates with this SFH
	#print(model_templates.shape, sfh.shape, age_weight.shape)
	template = numpy.sum(model_templates[:-1] * \
		sfh[:-1].reshape((-1,1)) * age_weight.reshape((-1,1)), axis=0)
	assert template.shape == (len(model_wavelength),), template.shape
	#print('   template max value after sfh convolution:', template.max())
	# normalise template at the highest wavelength
	template /= 1e-10 + template[2050]

	# apply calzetti extinction law at restframe
	template = template * 10**(-2.5 * calzetti_result * EBV)
	#print('   template max value after extinction:', template.max())
	
	#template = numpy.interp(x=inversewavelength_grid, xp=1./model_wavelength[::-1], fp=template[::-1])
	#
	## add Doppler blurring
	## sigma_4000 is something like a readshift:
	## f = f_0 * (1 + v/c)
	#sigma = 1 + v / 300000.
	## if sigma is 1A at 4000A, then on the 1/lam grid it should be this wide:
	#sigma_grid = sigma * 4000 / inversewavelength_gridwidth_A
	## convolve:
	#template = scipy.ndimage.filters.gaussian_filter1d(template, sigma_grid)
	
	# convert back to lambda
	
	# redshift / Doppler shift
	# interpolate template onto data grid
	# we go to the model at the restframe wavelength, which is bluer
	# template = numpy.interp(x=wavelength / (1 + z), xp=inversewavelength_grid, fp=template)
	template = numpy.interp(x=wavelength / (1 + z), xp=model_wavelength, fp=template)
	#print('   template max value after redshifting:', template.max())

	#template = model_interp([Z, sfage, SFtau])[0]
	assert template.shape == (nspec,), template.shape
	#assert numpy.all(numpy.isfinite(exttemplate)), exttemplate
	return template

if True:
	#O = 20
	Z, SFtau, SFage, z, EBV = -2, 1.e8, 1, 0, 0
	for Z in [-4, -2, -1]:
		ypred = model(Z, SFtau, SFage, z, EBV)
		plt.plot(wavelength, ypred, label='Z=%s' % Z)
	plt.legend(loc='best')
	plt.savefig('musefuse_model_Z.pdf', bbox_inches='tight')
	plt.close()
	Z = -2
	for SFtau in [6., 6.1, 6.3, 6.5, 7., 8., 9.]:
		ypred = model(Z, 10**SFtau, SFage, z, EBV)
		plt.plot(wavelength, ypred, label='SFtau=${10}^{%s}$' % SFtau)
	plt.legend(loc='best')
	plt.savefig('musefuse_model_SFtau.pdf', bbox_inches='tight')
	plt.close()
	SFtau = 1e8
	for SFage in [0.001, 0.01, 0.1, 1, 6, 12]:
		ypred = model(Z, SFtau, SFage, z, EBV)
		plt.plot(wavelength, ypred, label='SFage=%s' % SFage)
	plt.legend(loc='best')
	plt.savefig('musefuse_model_SFage.pdf', bbox_inches='tight')
	plt.close()
	SFage = 1
	for z in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
		ypred = model(Z, SFtau, SFage, z, EBV)
		plt.plot(wavelength, ypred, label='z=%s' % z)
	plt.legend(loc='best')
	plt.savefig('musefuse_model_z.pdf', bbox_inches='tight')
	plt.close()
	z = 0.
	for EBV in [0, 0.5, 1]:
		ypred = model(Z, SFtau, SFage, z, EBV)
		plt.plot(wavelength, ypred, label='EBV=%s' % EBV)
	plt.legend(loc='best')
	plt.savefig('musefuse_model_EBV.pdf', bbox_inches='tight')
	plt.close()


def priortransform(cube):
	# definition of the parameter width, by transforming from a unit cube
	cube = cube.copy()
	#cube[0] = 10**(cube[0] * 4 - 2) # plateau
	cube[0] = cube[0] * (Zs.max() - Zs.min()) + Zs.min()
	cube[1] = cube[1] * (sftaus.max() - sftaus.min()) + sftaus.min()
	cube[2] = cube[2] * (sfages.max() - sfages.min()) + sfages.min()
	#cube[4] = cube[4] * 3 + 1 # v (km/s)
	cube[3] = cube[3] * (zhi - zlo) + zlo # z
	cube[4] = cube[4] * 2 # E(B-V)
	#cube[8] = cube[8] * 4 - 1 # misfit
	return cube

def priortransform_simple(cube):
	# definition of the parameter width, by transforming from a unit cube
	cube = cube.copy()
	#cube[0] = 10**(cube[0] * 4 - 2) # plateau
	cube[0] = cube[0] * (sftaus.max() - sftaus.min()) + sftaus.min()
	cube[1] = cube[1] * (sfages.max() - sfages.min()) + sfages.min()
	cube[2] = cube[2] * (zhi - zlo) + zlo # z
	cube[3] = cube[3] * 2 # E(B-V)
	return cube

# the following is a python-only implementation of the likelihood 
# @ params are the parameters (as transformed by priortransform)
# @ data_mask is which data sets to consider.
# returns a likelihood vector
Lmax = -1e100
Lmax = -1e100 * numpy.ones(ndata)
def multi_loglikelihood(params, data_mask):
	global Lmax
	O, Z, logSFtau, SFage, z, EBV = params
	SFtau = 10**logSFtau
	# predict the model
	ypred = model(Z, SFtau, SFage, z, EBV)
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
		j = numpy.where(data_mask)[0][i]
		if L[i] > Lmax[j]:
			Lmax[j] = L[i]
			print('plotting...')
			plt.figure(figsize=(20,20))
			plt.subplot(3, 1, 1)
			plt.title(str(params) + ' : chi2:' + str(chi2))
			#mask = vd[:,i] < 2 * numpy.median(vd[:,i])
			#mask = numpy.isfinite(vd[:,i])
			mask = Ellipsis
			plt.plot(wavelength, yd[mask,i], color='k', alpha=0.5)
			plt.plot(wavelength, s * ypred[mask], color='r')
			plt.ylim(yd[mask,i].min(), yd[mask,i].max())
			plt.subplot(3, 1, 2)
			plt.plot(wavelength, ypred[mask], color='k')
			plt.subplot(3, 1, 3)
			plt.plot(wavelength, vd[mask,i], color='k')
			plt.yscale('log')
			plt.savefig('musefuse_bestfit_%d.pdf' % (i+1), bbox_inches='tight')
			plt.close()
			time.sleep(0.1)
		#print chi2
	assert L.shape == (ndata,), (L.shape, ypred.shape, y.shape, data_mask)
	return L

def multi_loglikelihood_vectorized(params, data_mask):
	global Lmax
	O, Z, logSFtau, SFage, z, EBV = params
	SFtau = 10**logSFtau
	# predict the model
	ypred = model(Z, SFtau, SFage, z, EBV)
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
	
	#for j, i in enumerate(numpy.where(L > Lmax[data_mask])[0]):
	for j, i in enumerate(numpy.where(data_mask)[0]):
		if not (L[j] > Lmax[i]): continue
		Lmax[i] = L[j]
		if i % (1 + ndata // 3) != 0: continue
		print('updating bestfit plot of %d ... chi2: %.2f' % (i+1, chi2[j]))
		#print '   ', yd.shape, yd[:,j].shape, ypred.shape
		plt.figure(figsize=(20,20))
		plt.subplot(3, 1, 1)
		plt.title('%s : chi2: %.2f' % (params, chi2[j]))
		#mask = vd[:,i] < 2 * numpy.median(vd[:,i])
		#mask = numpy.isfinite(vd[:,i])
		mask = Ellipsis
		plt.plot(wavelength, yd[mask,j], color='k', alpha=0.5)
		plt.plot(wavelength, s[j] * ypred[mask], color='r')
		plt.ylim(yd[mask,j].min(), yd[mask,j].max())
		plt.subplot(3, 1, 2)
		plt.plot(wavelength, ypred[mask], color='k')
		plt.subplot(3, 1, 3)
		plt.plot(wavelength, vd[mask,j], color='k')
		plt.yscale('log')
		plt.savefig('musefuse_bestfit_%d.pdf' % (i+1), bbox_inches='tight')
		plt.close()
		time.sleep(0.1)

	return L

def multi_loglikelihood_vectorized_short(params, data_mask):
	O, Z, logSFtau, SFage, z, EBV = params
	SFtau = 10**logSFtau
	# predict the model
	ypred = model(Z, SFtau, SFage, z, EBV)
	# do the data comparison
	if (ypred == 0).all():
		# give low probability to solutions with no stars
		return numpy.ones(data_mask.sum()) * -1e100
	ypred += O
	
	yd = y[:,data_mask]
	vd = noise_level[:,data_mask]
	ypreds = ypred.reshape((-1,1))
	s = numpy.sum(yd * ypreds / vd, axis=0) / (numpy.sum(ypreds**2 / vd, axis=0) + 1e-10)
	chi2 = numpy.sum((yd - s.reshape((1,-1)) * ypreds)**2 / vd, axis=0)
	L = -0.5 * chi2 + numpy.random.uniform() * 1e-5
	return L

import numexpr as ne
def multi_loglikelihood_numexpr(params, data_mask):
	O, Z, logSFtau, SFage, z, EBV = params
	SFtau = 10**logSFtau
	# predict the model
	ypred = model(Z, SFtau, SFage, z, EBV)
	# do the data comparison
	if (ypred == 0).all():
		# give low probability to solutions with no stars
		return numpy.ones(data_mask.sum()) * -1e100
	ypred += O
	
	yd = y[:,data_mask]
	vd = noise_level[:,data_mask]
	ypreds = ypred.reshape((-1,1))
	s1 = ne.evaluate("sum(yd * ypreds / vd, axis=0)")
	s2 = ne.evaluate("sum(ypreds**2 / vd, axis=0)")
	s = ne.evaluate("s1 / (s2 + 1e-10)").reshape((1,-1))
	return ne.evaluate("sum((yd - s * ypreds)**2 / (-2 * vd), axis=0)")

from ctypes import *
from numpy.ctypeslib import ndpointer
if int(os.environ.get('OMP_NUM_THREADS', '1')) > 1:
	lib = cdll.LoadLibrary('./cmuselike-parallel.so')
else:
	lib = cdll.LoadLibrary('./cmuselike.so')
lib.like.argtypes = [
	ndpointer(dtype=numpy.float64, ndim=2, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=2, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.bool, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	]

Lout = numpy.zeros(ndata)
def multi_loglikelihood_clike(params, data_mask):
	global Lout
	#O = 0
	Z, logSFtau, SFage, z, EBV = params
	SFtau = 10**logSFtau
	# predict the model
	ypred = model(Z, SFtau, SFage, z, EBV)
	# do the data comparison
	if not numpy.any(ypred):
		# give low probability to solutions with no stars
		return numpy.ones(data_mask.sum()) * -1e100
	#ypred += O
	
	# do everything in C and return the resulting likelihood vector
	ret = lib.like(y, noise_level, ypred, data_mask, ndata, nspec, Lout)
	return Lout[data_mask] + numpy.random.normal(0, 1e-5, size=data_mask.sum())

def multi_loglikelihood_simple_clike(params, data_mask):
	logSFtau, SFage, z, EBV = params
	#Z = 0.012 # solar
	Z = 0.004 # Patricio2018
	params = Z, logSFtau, SFage, z, EBV
	return multi_loglikelihood_clike(params, data_mask)

if False:
	data_mask_all = numpy.ones(ndata) == 1
	print('testing vectorised code...')
	for i in range(100):
		cube = numpy.random.uniform(size=nparams)
		params = priortransform(cube)
		L = multi_loglikelihood(params, data_mask_all)
		L2 = multi_loglikelihood_vectorized(params, data_mask_all)
		assert numpy.allclose(L, L2), (L, L2, cube, params)
		L2 = multi_loglikelihood_vectorized_short(params, data_mask_all)
		assert numpy.allclose(L, L2), (L, L2, cube, params)
		L2 = multi_loglikelihood_numexpr(params, data_mask_all)
		assert numpy.allclose(L, L2), (L, L2, cube, params)
		L2 = multi_loglikelihood_clike(params, data_mask_all)
		assert numpy.allclose(L, L2), (L, L2, cube, params)
	test_cubes = [priortransform(numpy.random.uniform(size=nparams)) for i in range(1000)]
	a = time.time()
	[multi_loglikelihood(cube, data_mask_all) for cube in test_cubes]
	print('original python code:', time.time() - a)
	a = time.time()
	[multi_loglikelihood_vectorized(cube, data_mask_all) for cube in test_cubes]
	print('vectorised python code:', time.time() - a)
	a = time.time()
	[multi_loglikelihood_vectorized_short(cube, data_mask_all) for cube in test_cubes]
	print('shortened vectorised python code:', time.time() - a)
	a = time.time()
	[multi_loglikelihood_numexpr(cube, data_mask_all) for cube in test_cubes]
	print('numexpr code:', time.time() - a)
	a = time.time()
	[multi_loglikelihood_clike(cube, data_mask_all) for cube in test_cubes]
	print('C code:', time.time() - a)

#multi_loglikelihood = multi_loglikelihood_vectorized_short
#multi_loglikelihood = multi_loglikelihood_numexpr
multi_loglikelihood = multi_loglikelihood_clike

prefix = sys.argv[1]

modelname = os.environ.get('MODEL', 'FULL')
if modelname == 'ZSOL':
	paramnames = ['logSFtau', 'SFage', 'z', 'EBV']
	nparams = len(paramnames)
	prefix = prefix + '_zsol_'
	print('Switching to Zsol model')
	multi_loglikelihood = multi_loglikelihood_simple_clike
	priortransform = priortransform_simple
elif modelname == 'FULL':
	prefix = prefix + '_full_'
	pass
else:
	assert False, modelname

"""

After defining the problem, we use generic code to set up 
- Nested Sampling (Multi)Integrator
- Our special sampler
- RadFriends (constrained region draw)

We start with the latter.
"""


from multi_nested_integrator import multi_nested_integrator
from multi_nested_sampler import MultiNestedSampler
from cachedconstrainer import CachedConstrainer, generate_individual_constrainer, generate_superset_constrainer

superset_constrainer = generate_superset_constrainer()

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
max_samples = int(os.environ.get('MAXSAMPLES', 100000))
min_samples = int(os.environ.get('MINSAMPLES', 0))
results = multi_nested_integrator(tolerance=0.5, multi_sampler=sampler, min_samples=min_samples, max_samples=max_samples)
duration = time.time() - start_time
print('writing output files ...')
# store results
with h5py.File(prefix + '.out_%d.hdf5' % ndata, 'w') as f:
	f.create_dataset('logZ', data=results['logZ'], compression='gzip', shuffle=True)
	f.create_dataset('logZerr', data=results['logZerr'], compression='gzip', shuffle=True)
	u, x, L, w, mask = list(zip(*results['weights']))
	f.create_dataset('u', data=u, compression='gzip', shuffle=True)
	f.create_dataset('x', data=x, compression='gzip', shuffle=True)
	f.create_dataset('L', data=L, compression='gzip', shuffle=True)
	f.create_dataset('w', data=w, compression='gzip', shuffle=True)
	f.create_dataset('mask', data=mask, compression='gzip', shuffle=True)
	f.create_dataset('ndraws', data=sampler.ndraws)
	f.create_dataset('fiberids', data=goodids[:ndata], compression='gzip', shuffle=True)
	f.create_dataset('duration', data=duration)
	f.create_dataset('ndata', data=ndata)
	
	print('logZ = %.1f +- %.1f' % (results['logZ'][0], results['logZerr'][0]))
	print('ndraws:', sampler.ndraws, 'niter:', len(w))

print('writing statistic ...')
json.dump(dict(ndraws=sampler.ndraws, duration=duration, ndata=ndata, niter=len(w)), 
	open(prefix + '.out_%d.stats.json' % ndata, 'w'), indent=4)
print('done.')


