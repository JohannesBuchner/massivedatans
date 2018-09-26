from __future__ import print_function, division
import numpy
import matplotlib.pyplot as plt

CX = [2, 2.2]
CSX = [0.5, 0.5] 
CSY = [0.2, 0.2] 
CY = [1.1, 1.2]

def likelihood(x, y):
	l = 0
	for cx, cy, csx, csy in zip(CX, CY, CSX, CSY):
		l = -0.5 * (((cx - x)/csx)**2 + ((cy - y)/csy)**2)
		yield l

x = numpy.linspace(-2.5, 6.5, 100)
y = numpy.linspace(-2.5, 6.5, 100)
X, Y = numpy.meshgrid(x, y)
XY = numpy.array(numpy.transpose([X.flatten(), Y.flatten()]), order='C')
L1, L2 = likelihood(X, Y)
Lsorted = L1[30:-30,30:-30].flatten()
Lsorted.sort()
levels = Lsorted[::Lsorted.size/7-1].tolist() # + [L.max()]
levels = levels[2:]
#levels = L.max() - numpy.arange(5) * 4 - 2
plt.figure(figsize=(6, 3), frameon=False)
plt.axis('off')
plt.contour(X, Y, L1, levels)
plt.contour(X, Y, L2, levels)
plt.savefig('plotjointcontour.png', bbox_inches='tight')
plt.savefig('plotjointcontour.pdf', bbox_inches='tight')
plt.close()

numpy.random.seed(1)
N = 10000
x = numpy.random.uniform(-2, 6, size=N)
y = numpy.random.uniform(-2, 6, size=N)
l1, l2 = likelihood(x, y)
Nlive = 100
for i in range(len(levels)):
	plt.figure(figsize=(6, 2.2), frameon=False)
	plt.axis('off')
	#plt.text(-2, 4, 'Iteration %d' % (i*100))
	#plt.text(-2, 4, '(%d)' % (i+1))
	mask1 = l1 > levels[i]
	mask2 = l2 > levels[i]
	maskboth = numpy.logical_and(mask1, mask2)
	maskone = numpy.logical_or(mask1, mask2)
	N1 = 0
	N2 = 0
	for j in range(N):
		if mask1[j] and mask2[j]: # joint
			plt.plot(x[j], y[j], '.', color='k')
			N1 += 1
			N2 += 1
		elif mask1[j] and N1 < Nlive: 
			plt.plot(x[j], y[j], 'x', color='cyan')
			N1 += 1
		elif mask2[j] and N2 < Nlive: 
			plt.plot(x[j], y[j], '+', color='magenta')
			N2 += 1
		else:
			pass
		if N1 >= Nlive and N2 >= Nlive:
			break
	plt.contour(X, Y, L1, levels[i:i+1], colors=['cyan'], linestyles=[':'])
	plt.contour(X, Y, L2, levels[i:i+1], colors=['magenta'], linestyles=[':'])
	plt.ylim(-2.5, 6.2)
	plt.xlim(-3, 7)
	plt.savefig('plotjointcontour_%d.png' % (i+1), bbox_inches='tight')
	plt.savefig('plotjointcontour_%d.pdf' % (i+1), bbox_inches='tight')
	plt.close()



