from __future__ import print_function, division
import numpy
import matplotlib.pyplot as plt
import h5py
from numpy import exp
import sys

def gauss(x, z, A, mu, sig):
	xT = x.reshape((1,-1))
	zT = z.reshape((-1,1))
	AT = A.reshape((-1,1))
	muT = mu.reshape((-1,1))
	sigT = sig.reshape((-1,1))
	return AT * exp(-0.5 * ((muT - xT / (1. + zT))/sigT)**2)

x = numpy.linspace(400, 800, 200)

N = 40
N = int(sys.argv[1])
numpy.random.seed(N)
alpha, beta, scale = 2., 7., 1
z = numpy.random.beta(alpha, beta, size=N) * scale
#z = numpy.zeros(N) + 0.01
rest_wave = 440
print('generating parameters ...')
# in km/s
width_broad = 4000 * rest_wave / 300000 * numpy.ones(N)
width_narrow = 400 * rest_wave / 300000 * numpy.ones(N)
# convert to nm
mean_broad  = rest_wave * numpy.ones(N)
mean_narrow = rest_wave * numpy.ones(N)
width_broad = width_broad
width_narrow = width_narrow
noise_level = 0.01
#signal_level = numpy.random.exponential(size=N) * 0.4
#signal_level = numpy.ones(N) * 0.04
signal_level = numpy.random.normal(0.2, 0.2, size=10*N)
signal_level = signal_level[signal_level>0.1][:N]
#signal_level = numpy.random.uniform(size=N) * 0.5
#is_type1 = numpy.random.uniform(size=N) < 0.5
height_broad  = 10**-1 * signal_level
height_narrow = signal_level

#X = numpy.array([x])

print('generating signal ...')
ym =  gauss(A=height_broad, mu=mean_broad, x=x, z=z, sig=width_broad)
ym += gauss(A=height_narrow, mu=mean_narrow, x=x, z=z, sig=width_narrow)
ym = numpy.transpose(ym)
print(ym.shape)

# add noise
print('adding noise...')
y = ym.copy()
for i in range(N):
	y[:,i] += numpy.random.normal(0, noise_level, size=len(x))

print('plotting ...')
colors = ['yellow', 'pink', 'cyan', 'magenta']
for i in range(min(N, 4)):
	#plt.plot(x, y[:,i], '.-')
	plt.plot( rest_wave * (1+z[i]), 0.15 * height_narrow[i] / noise_level, 'v', color=colors[i], ms=12)
	plt.plot(x, y[:,i] / noise_level, '-', color=colors[i])
plt.ylabel('Detector signal')
plt.xlabel('Wavelength [nm]')
plt.savefig('genfaint.pdf', bbox_inches='tight')
plt.close()

#print x.shape, y.shape, z.shape
with h5py.File('data_faint_%s.hdf5' % sys.argv[1], 'w') as f:
	f.create_dataset('x', data=x, compression='gzip', shuffle=True)
	f.create_dataset('y', data=y, compression='gzip', shuffle=True)
	f.create_dataset('z', data=z, compression='gzip', shuffle=True)
	f.create_dataset('mean_broad', data=mean_broad, compression='gzip', shuffle=True)
	f.create_dataset('width_broad', data=width_broad, compression='gzip', shuffle=True)
	f.create_dataset('height_broad', data=height_broad, compression='gzip', shuffle=True)
	f.create_dataset('mean_narrow', data=mean_narrow, compression='gzip', shuffle=True)
	f.create_dataset('width_narrow', data=width_narrow, compression='gzip', shuffle=True)
	f.create_dataset('height_narrow', data=height_narrow, compression='gzip', shuffle=True)


