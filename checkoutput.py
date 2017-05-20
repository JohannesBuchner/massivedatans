import h5py
import sys
import numpy
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
		ndraws = f['ndraws']
		print 'logZ = %.1f +- %.1f' % (logZ, logZerr)
		print 'ndraws:', ndraws
		plt.plot(L)

plt.show()

