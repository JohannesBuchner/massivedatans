from __future__ import print_function, division
import numpy
import matplotlib.pyplot as plt
#from nested_sampling.clustering.neighbors import find_rdistance, is_within_distance_of, count_within_distance_of, any_within_distance_of
from nested_sampling.samplers.hiermetriclearn import ClusterResult, RadFriendsRegion


CX = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
CS = [0.2, 0.2, 0.2, 0.2, 0.15, 0.2, 0.15, 0.2, 0.2] 
CY = [0.2, 0, 0, 0, 0.1, 0.3, 1, 1.4, 2]
CW = [1, 2, 2, 2, 2, 2, 20, 2, 2]

CX = numpy.linspace(0, 4, 20)
CY = CX*-0.2 + CX**2*0.3
#plt.plot(x, x*-0.2 + x**2*0.2)
CW = CX * 0 + 2 + 10*CY**2
CW = 1./CW
CW[0] = 0.5
CW[1] = 1
#CW[-5] = 20
CS = CX * 0 + 0.2
#CS[-5] = 0.12


def likelihood(x, y):
	l = 0
	for cx, cy, cw, cs in zip(CX, CY, CW, CS):
		l += cw * numpy.exp(-0.5 * (((cx - x)/cs)**2 + ((cy - y)/cs)**2))
	return numpy.log(l)


x = numpy.linspace(-2.5, 6.5, 100)
y = numpy.linspace(-2.5, 6.5, 100)
X, Y = numpy.meshgrid(x, y)
XY = numpy.array(numpy.transpose([X.flatten(), Y.flatten()]), order='C')
print(XY.dtype)
L = likelihood(X, Y)
Lsorted = L[30:-30,30:-30].flatten()
Lsorted.sort()
levels = Lsorted[::Lsorted.size/7-1].tolist() # + [L.max()]
levels = levels[2:]
#levels = L.max() - numpy.arange(5) * 4 - 2
plt.figure(figsize=(6, 3), frameon=False)
plt.axis('off')
plt.contour(X, Y, L, levels)
plt.savefig('plotcontour.png', bbox_inches='tight')
plt.savefig('plotcontour.pdf', bbox_inches='tight')
plt.close()

numpy.random.seed(1)
N = 10000
x = numpy.random.uniform(-2, 6, size=N)
y = numpy.random.uniform(-2, 6, size=N)
l = likelihood(x, y)
Nlive = 100
for i in range(len(levels)):
	plt.figure(figsize=(6, 2.2), frameon=False)
	plt.axis('off')
	plt.text(-2, 4, 'Iteration %d' % (i*100))
	#plt.text(-2, 4, '(%d)' % (i+1))
	mask = l > levels[i]
	xlevel = x[mask][:Nlive]
	ylevel = y[mask][:Nlive]
	live_points = numpy.array(numpy.transpose([xlevel, ylevel]), order='C')
	plt.contour(X, Y, L, levels[i:i+1], colors=['k'], linestyles=[':'])
	plt.plot(xlevel, ylevel, '.', color='k')
	# do radfriends with these points
	region = RadFriendsRegion(live_points)
	mask = region.are_inside(XY)
	maskregion = mask.reshape(X.shape)
	plt.contour(X, Y, maskregion*1., [0.5], colors=['orange'], linestyles=['-'])
	
	plt.ylim(-2.5, 6.2)
	plt.xlim(-3, 7)
	plt.savefig('plotcontour_%d.png' % (i+1), bbox_inches='tight')
	plt.savefig('plotcontour_%d.pdf' % (i+1), bbox_inches='tight')
	plt.close()



