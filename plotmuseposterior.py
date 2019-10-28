from __future__ import print_function, division
import json
import numpy
from numpy import log, log10, arctan, pi, exp
import sys
import matplotlib.pyplot as plt
import h5py
import scipy.stats
import corner

filename = sys.argv[1]
with h5py.File(filename, 'r') as f:
	logZ = f['logZ'].value
	for i in range(len(logZ)):
		print('   %d ...' % i)
		w = f['w'][:,i] + f['L'][:,i]
		mask = numpy.isfinite(w)
		if mask.sum() < 4000:
			continue
		jparent = numpy.where(mask)[0]
		w = w[jparent]
		#print w, w.min(), w.max()
		w = numpy.exp(w - w.max())
		w = w / w.sum()
		j = numpy.random.choice(jparent, size=100000, p=w)
		
		O = numpy.log10(f['x'][:,i,0][j])
		Z = f['x'][:,i,1][j]
		SFtau = f['x'][:,i,2][j]
		SFage = numpy.log10(f['x'][:,i,3][j])
		EBV = f['x'][:,i,4][j]
		print(w.shape, O.shape, Z.shape, SFtau.shape, SFage.shape, EBV.shape)
		data = numpy.transpose([O, Z, SFtau, SFage, EBV])
		
		# make marginal plots
		
		figure = corner.corner(data, 
			labels=[r"Continuum", r"logZ", r"SFtau", r"SFage", r'EBV'],
                	quantiles=[0.16, 0.5, 0.84],
                	show_titles=True, title_kwargs={"fontsize": 12})
		figure.savefig('museposterior_%d.pdf' % (i+1), bbox_inches='tight')
		plt.close()

