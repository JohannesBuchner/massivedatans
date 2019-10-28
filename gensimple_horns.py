from __future__ import print_function, division
import numpy
import matplotlib.pyplot as plt
import h5py
from numpy import exp, pi, arctan
import sys

def gauss(x, A, mu, sig):
	xT = x.reshape((1,-1))
	AT = A.reshape((-1,1))
	muT = mu.reshape((-1,1))
	sigT = sig.reshape((-1,1))
	return AT * exp(-0.5 * ((muT - xT)/sigT)**2)

x = numpy.linspace(400, 800, 200)

N = 40
N = int(sys.argv[1])
numpy.random.seed(N)
z = arctan(numpy.random.uniform(-pi, pi, size=N)) * 0.1
rest_wave = 656
print('generating parameters ...')
width_narrow = 5.0 * numpy.ones(N)
mean_narrow = rest_wave * (1 + z)
width_narrow = width_narrow
noise_level = 0.01
signal_level = 0.02 / numpy.random.power(3, size=N)
height_narrow = signal_level

print('generating signal ...')
ym = gauss(A=height_narrow, mu=mean_narrow, x=x, sig=width_narrow)
ym = numpy.transpose(ym)
print(ym.shape)

# add noise
print('adding noise...')
y = ym.copy()
for i in range(N):
	y[:,i] += numpy.random.normal(0, noise_level, size=len(x))

print('plotting ...')
#for i in range(min(N, 20)):
#	#plt.plot(x, y[:,i], '.-')
#	plt.plot(x, y[:,i], '-')
#plt.savefig('gen_widths.pdf', bbox_inches='tight')
#plt.close()
colors = ['yellow', 'pink', 'cyan', 'magenta']
colors = ['magenta', 'cyan', 'pink', 'yellow']
for i in range(min(N, 4)):
	#plt.plot(x, y[:,i], '.-')
	plt.plot(rest_wave * (1 + z[i]), 1.1 * y[:,i].max() / noise_level, 'v', color=colors[i], ms=12, mew=0.5, mec='k')
	#plt.plot(rest_wave * (1 + z[i]), 4, 'v', color=colors[i], ms=12)
	plt.plot(x, y[:,i] / noise_level, '-', color=colors[i], lw=1)
plt.ylabel('Detector signal')
plt.xlabel('Wavelength [nm]')
plt.savefig('genhorns.pdf', bbox_inches='tight')
plt.close()


#print x.shape, y.shape, z.shape
with h5py.File('data_widths_%s.hdf5' % sys.argv[1], 'w') as f:
	f.create_dataset('x', data=x, compression='gzip', shuffle=True)
	f.create_dataset('y', data=y, compression='gzip', shuffle=True)
	f.create_dataset('z', data=z, compression='gzip', shuffle=True)
	f.create_dataset('mean_narrow', data=mean_narrow, compression='gzip', shuffle=True)
	f.create_dataset('width_narrow', data=width_narrow, compression='gzip', shuffle=True)
	f.create_dataset('height_narrow', data=height_narrow, compression='gzip', shuffle=True)


