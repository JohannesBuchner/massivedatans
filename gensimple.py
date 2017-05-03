import numpy
import matplotlib.pyplot as plt
import h5py
from numpy import exp

def gauss(x, z, A, mu, sig):
	xT = x.reshape((1,-1))
	zT = z.reshape((-1,1))
	AT = A.reshape((-1,1))
	muT = mu.reshape((-1,1))
	sigT = sig.reshape((-1,1))
	return AT * exp(-0.5 * ((muT - xT / (1. + zT))/sigT)**2)

x = numpy.linspace(400, 800, 1000)

N = 40
numpy.random.seed(1)
z = numpy.zeros(N) + 0.01
rest_wave = 440
# in km/s
width_broad = 10**3 * rest_wave / 300000 * numpy.ones(N)
width_narrow = 10**1 * rest_wave / 300000 * numpy.ones(N)
# convert to nm
mean_broad  = rest_wave * numpy.ones(N)
mean_narrow = rest_wave * numpy.ones(N)
width_broad = width_broad
width_narrow = width_narrow
noise_level = 0.01
#signal_level = numpy.random.exponential(size=N) * 10
signal_level = numpy.ones(N) * 10
#is_type1 = numpy.random.uniform(size=N) < 0.5
height_broad  = 10**-1 * signal_level
height_narrow = signal_level

#X = numpy.array([x])

ym =  gauss(A=height_broad, mu=mean_broad, x=x, z=z, sig=width_broad)
ym += gauss(A=height_narrow, mu=mean_narrow, x=x, z=z, sig=width_narrow)
ym = numpy.transpose(ym)
print ym.shape

# add noise
print 'adding noise'
y = numpy.random.normal(0, noise_level, size=ym.shape) + ym
print 'plotting ...'
for i in range(min(N, 20)):
	#plt.plot(x, y[:,i], '.-')
	plt.plot(x, y[:,i], '-')
plt.savefig('gen.pdf', bbox_inches='tight')
plt.close()

print x.shape, y.shape, z.shape
with h5py.File('data.hdf5', 'w') as f:
	f.create_dataset('x', data=x, compression='gzip', shuffle=True)
	f.create_dataset('y', data=y, compression='gzip', shuffle=True)
	f.create_dataset('z', data=z, compression='gzip', shuffle=True)
	f.create_dataset('mean_broad', data=mean_broad, compression='gzip', shuffle=True)
	f.create_dataset('width_broad', data=width_broad, compression='gzip', shuffle=True)
	f.create_dataset('height_broad', data=height_broad, compression='gzip', shuffle=True)
	f.create_dataset('mean_narrow', data=mean_narrow, compression='gzip', shuffle=True)
	f.create_dataset('width_narrow', data=width_narrow, compression='gzip', shuffle=True)
	f.create_dataset('height_narrow', data=height_narrow, compression='gzip', shuffle=True)


