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

filename_in = sys.argv[1]
filename = sys.argv[2]
plt.figure(figsize=(6,4))
f = h5py.File(filename_in, 'r')
logZ0 = numpy.sum(-0.5 * (f['y'].value/0.01)**2, axis=0)
f = h5py.File(filename, 'r')
logZ1 = f['logZ'].value
B = numpy.log10(numpy.exp(logZ1 - logZ0))
B[B > 4] = 4
bins = numpy.linspace(B.min(), 10, 40)
plt.hist(B, bins=bins, color='k', histtype='step', normed=True)

filename_in = sys.argv[3]
filename = sys.argv[4]
f = h5py.File(filename_in, 'r')
logZ0 = numpy.sum(-0.5 * (f['y'].value/0.01)**2, axis=0)
f = h5py.File(filename, 'r')
logZ1 = f['logZ'].value
B = numpy.log10(numpy.exp(logZ1 - logZ0))
Blim = sorted(B)[int(len(B)*0.999)]
Blim = B.max()
print(10**Blim)
bins = numpy.linspace(-5, 5, 100)
plt.hist(B, bins=bins, color='r', histtype='step', normed=True)
x = list(range(-1, 5))
plt.vlines(Blim, 0, 4, color='green', linestyles=[':'])
plt.ylim(0, 4)
plt.yticks([0, 1, 2, 3, 4])
y = ['${10}^{%d}$' % xi for xi in x]
plt.xticks(x, y)
plt.xlim(-2, 4.5)
plt.xlabel('Bayes factor B')
plt.ylabel('Frequency')
plt.savefig('plotevidences.pdf', bbox_inches='tight')
plt.close()





