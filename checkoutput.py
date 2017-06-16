import h5py
import sys
import numpy
from numpy import log, log10, exp, pi
import matplotlib.pyplot as plt

for filename in sys.argv[1:]:
	with h5py.File(filename) as f:
		print filename
		logZ = f['logZ'].value
		logZerr = f['logZerr'].value
		L = f['L'].value
		
		if len(logZ.shape) > 0:
			logZ = logZ[0]
			logZerr = logZerr[0]
			L = L[:,0]
			#print f['x'][-1,0]
		else:
			#print f['x'][-1]
			pass
		ndraws = f['ndraws']
		print 'logZ = %.1f +- %.1f' % (logZ, logZerr)
		print 'ndraws:', ndraws
		#plt.plot(L)
		w = f['w'][:,0]
		w = exp(w - w.max())
		w.sort()
		w /= w.sum()
		i = numpy.random.choice(numpy.arange(len(w)), size=1000, replace=True, p=w)
		A, mu, logsigma = f['x'][:,0,:].transpose()
		A = log10(A[i])
		mu = mu[i]
		logsigma = logsigma[i]
		print 'A', A.mean(), A.std()
		print 'mu', mu.mean(), mu.std()
		print 'logsigma', logsigma.mean(), logsigma.std()
		#plt.plot(A, mu, 'x ')
		#plt.show()
		print f['w'].shape, f['x'].shape


