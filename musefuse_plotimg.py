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
import h5py
import sys
import matplotlib.pyplot as plt



print 'loading data...'
ndata = int(sys.argv[2])
#f = pyfits.open(sys.argv[1])
filename = sys.argv[1] + '.outimg_%d.hdf5' % ndata
print 'writing image files ...'
def makeimg(prefix, img):
	outfilename = sys.argv[1] + '.outimg_%d_%s.pdf' % (ndata, prefix)
	print 'creating %s ...' % outfilename
	plt.figure()
	plt.title(prefix)
	plt.imshow(img, cmap=plt.cm.RdBu)
	plt.colorbar()
	plt.savefig(outfilename, bbox_inches='tight')
	plt.close()
	
# store results
with h5py.File(filename, 'r') as fimg:
	nparams = fimg.attrs['nparams']
	for name in ['logZ'] + ['param%d' % k for k in range(nparams)]:
		makeimg(name, fimg[name]) 
		makeimg(name + 'err', fimg[name + 'err']) 


