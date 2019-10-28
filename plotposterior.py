from __future__ import print_function, division
import json
import numpy
from numpy import log, log10
import sys
import matplotlib.pyplot as plt
import h5py
import scipy.stats

xx = []
yy = []

filename = sys.argv[1]
colors = ['yellow', 'pink', 'cyan', 'magenta']
cmap = plt.cm.gray
zs = []
plt.figure(figsize=(6,4))
with h5py.File(filename, 'r') as f:
	logZ = f['logZ'].value
	for i in range(len(logZ)):
		w = f['w'][:,i] + f['L'][:,i]
		mask = numpy.isfinite(w)
		jparent = numpy.where(mask)[0]
		w = w[jparent]
		#print w, w.min(), w.max()
		w = numpy.exp(w - w.max())
		w = w / w.sum()
		j = numpy.random.choice(jparent, size=1000, p=w)
		mu = f['x'][:,i,1][j]
		if mu.std() < 50:
			zs.append(mu.mean() / 440 - 1)
		#if mu.std() > 40:
		#	print 'skipping unconstrained: %.1f' % mu.std()
		#	continue
		#A = log10(f['x'][:,i,0][j])
		A = f['x'][:,i,0][j] * 100
		#if i < 4:
		#	plt.plot(mu[:100], A[:100], '. ', color='r', alpha=0.2)
		if i < 4:
			color = colors[i]
		else:
			color = cmap(0.8 * min(50, mu.std())/50.)
		plt.errorbar(x=numpy.mean(mu), xerr=mu.std(), 
			y=A.mean(), yerr=A.std(),
			capsize=0, color=color,
			elinewidth=4 if i < 4 else 1)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Line amplitude')
plt.xlim(400, 800)
plt.ylim(1, 20)
plt.yticks([1,2,10], [1,2,10])
plt.yscale('log')
plt.savefig('plotposterior.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(5,1.5))
plt.hist(zs, bins=10, histtype='step', label='Well-constrained lines', normed=True)
alpha, beta, scale = 2., 7., 1
x = numpy.linspace(0, 2, 1000)
plt.plot(x, scipy.stats.beta(alpha, beta).pdf(x), '-', color='k', label='Input redshift distribution')
plt.ylabel('Frequency')
plt.xlabel('Redshift')
plt.xlim(0, 1)
plt.savefig('plotposteriorz.pdf', bbox_inches='tight')
plt.close()




