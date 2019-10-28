from __future__ import print_function, division
import json
import numpy
from numpy import log
import sys
import matplotlib.pyplot as plt

xx = []
yy = []

for filename in sys.argv[1:]:
	data = json.load(open(filename))
	if 'ndata' in data:
		x = data['ndata']
	else:
		x = int(filename.split('.')[0].split('_')[-1])
	#y = json.load(open(filename))['ndraws']
	#if 'duration' not in data:
	#	continue
	#y = data['duration']
	y = data['ndraws']
	xx.append(x)
	yy.append(y)

i = numpy.argsort(xx)
xx = numpy.array(xx)[i]
yy = numpy.array(yy)[i]

plt.figure(figsize=(5,5))
plt.plot(xx, xx * max(yy/xx), '-', label='linear cost', color='k')
plt.plot(xx, numpy.sqrt(xx) * numpy.nanmax(yy / numpy.sqrt(xx)), ':', label='sqrt cost', color='gray')
#plt.plot(xx, xx**0.333 * numpy.nanmax(yy / xx**0.333), '--', label='cubic root cost')
#plt.plot(xx, log(xx) * numpy.nanmax(yy / log(xx)), '-.', label='log cost')
plt.ylabel('Model Evaluations')
plt.xlabel('Data Sets')
plt.yscale('log')
plt.xscale('log')
#plt.xlim(0.9, 10000)
plt.xlim(0.8, max(xx)*1.5)
plt.plot(xx, yy, 'o ', label='our algorithm', color='r')
plt.legend(loc='upper left', numpoints=1, prop=dict(size=10))
plt.savefig('plotscaling.pdf', bbox_inches='tight')
plt.close()




