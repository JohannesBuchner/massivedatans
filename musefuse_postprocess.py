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

print('loading data...')
f = pyfits.open(sys.argv[1])
datasection = f['DATA'] 
y = datasection.data # values
y = y[:3600,:,:]
noise_level = f['STAT'].data # variance
noise_level = noise_level[:3600,:,:]
nspec, npixx, npixy = y.shape

print('applying subselection ...')
## replaced by mask
#y = y[:,80:200,170:240]
#noise_level = noise_level[:,80:200,170:240]
#y = y[:,70:80,35:45]
#noise_level = noise_level[:,70:80,35:45]

regionfile = sys.argv[2]
import pyregion
region = pyregion.parse(open(regionfile).read())
mask = region.get_mask(shape=(npixx, npixy))
ymask = numpy.array([mask] * len(y))
xids, yids = numpy.where(mask)
y = y[ymask]
y = y.reshape((nspec, -1))
noise_level = noise_level[ymask]
noise_level = noise_level.reshape((nspec, -1))

#nspec, npixx, npixy = y.shape
print((y.shape))
outputimg = numpy.zeros((npixx, npixy)) * numpy.nan

#y = y.reshape((nspec, -1))
#noise_level = noise_level.reshape((nspec, -1))
outputimg_flat = outputimg #.reshape((-1))
x = datasection.header['CD3_3'] * numpy.arange(nspec) + datasection.header['CRVAL3']
print('    finding NaNs...')
good = numpy.isfinite(noise_level).all(axis=0)
#assert good.shape == (npixx*npixy,), good.shape
#goodids = numpy.where(good)[0]
goodids = list(zip(xids[good], yids[good]))
print((len(good), len(goodids)))

y = y[:,good]
noise_level = noise_level[:,good]
ndata = os.environ.get('MAXDATA', len(goodids))
print('    truncating data to %d sets...' % ndata, goodids[:ndata])
## truncate data
y = y[:,:ndata]
noise_level = noise_level[:,:ndata]
goodids = goodids[:ndata]

print((y.shape))

prefix = sys.argv[1]
modelname = os.environ.get('MODEL', 'FULL')
if modelname == 'ZSOL':
	paramnames = ['logSFtau', 'SFage', 'z', 'EBV']
	prefix = prefix + '_zsol_'
elif modelname == 'FULL':
	paramnames = ['Z', 'logSFtau', 'SFage', 'z', 'EBV']
	prefix = prefix + '_full_'
else:
	assert False

filename = prefix + '.out_%d.hdf5' % ndata
f = h5py.File(filename, 'r')

nsamplesmax, nids, nparams = f['x'].shape
assert nids == len(goodids), (nids, goodids)

output_Z = outputimg_flat.copy()
output_Zerr = outputimg_flat.copy()
output_means = {}
output_errs = {}
for pi in range(nparams):
	output_means[pi] = outputimg_flat.copy()
	output_errs[pi] = outputimg_flat.copy()

weights = numpy.transpose(f['w'].value + f['L'].value)
print(weights.shape)

#def pointfactory():
#	x = f['x'].value
#	for i in range(nids):
#		yield x[:,i,:]
points = numpy.swapaxes(f['x'].value, 0, 1)



for i, (w, logZ, logZerr, x) in enumerate(zip(weights, f['logZ'].value, f['logZerr'].value, points)):
	xi, yi = goodids[i]
	mask = numpy.isfinite(w)
	jparent = numpy.where(mask)[0]
	w = w[jparent]
	w = numpy.exp(w - w.max())
	w = w / w.sum()
	j = numpy.random.choice(jparent, size=4000, p=w)
	print('   %d/%d: spaxel %s: from %d samples drew %d unique posterior points' % (i+1, nids, (xi, yi), len(jparent), len(numpy.unique(j))))
	
	print('        logZ = %.1f +- %.1f' % (logZ, logZerr))
	output_Z[xi, yi] = logZ
	output_Zerr[xi, yi] = logZerr
	#x = f['x'][:,i,:]
	xequal = x[j,:]
	for k in range(nparams):
		v = xequal[:,k]
		output_means[k][xi, yi] = v.mean()
		output_errs[k][xi, yi] = v.std()
		print('          param %d = %.3f +- %.3f (%s)' % (k, v.mean(), v.std(), paramnames[k]))
	if i < 5:
		numpy.savetxt(prefix + '.outsamples_%d.txt' % i, xequal)
	#if i > 1000: break

output_Z = output_Z.reshape((npixx, npixy))
output_Zerr = output_Zerr.reshape((npixx, npixy))
for pi in range(nparams):
	output_means[pi] = output_means[pi].reshape((npixx, npixy))
	output_errs[pi] = output_errs[pi].reshape((npixx, npixy))

filename = prefix + '.outimg_%d.hdf5' % ndata
print('writing image files ...')
def makeimg(name, img, title=None):
	outfilename = prefix + '.outimg_%d_%s.pdf' % (ndata, name)
	print('creating %s ...' % outfilename)
	plt.figure()
	if title is None:
		title = name
	plt.title(title)
	plt.imshow(img, cmap=plt.cm.RdBu)
	plt.colorbar()
	plt.savefig(outfilename, bbox_inches='tight')
	plt.close()
	
# store results
with h5py.File(filename, 'w') as fimg:
	fimg.create_dataset('logZ', data=output_Z, compression='gzip', shuffle=True)
	makeimg('logZ', output_Z)
	fimg.create_dataset('logZerr', data=output_Zerr, compression='gzip', shuffle=True)
	makeimg('logZerr', output_Zerr)
	for k in range(nparams):
		fimg.create_dataset('param%d' % k, data=output_means[k], compression='gzip', shuffle=True)
		makeimg('param%d' % k, output_means[k], title=paramnames[k])
		fimg.create_dataset('param%derr' % k, data=output_errs[k], compression='gzip', shuffle=True)
		makeimg('param%derr' % k, output_errs[k], title=paramnames[k] + ' errors')
	fimg.attrs['nparams'] = nparams


