from __future__ import print_function, division
import numpy
import matplotlib.pyplot as plt
import h5py
from numpy import exp
import sys
x = numpy.linspace(400, 800, 200)

N = int(sys.argv[1])
noise_level = 0.01
numpy.random.seed(N)
y = numpy.random.normal(0, noise_level, size=(len(x),N))

with h5py.File('data_nothing_%s.hdf5' % sys.argv[1], 'w') as f:
	f.create_dataset('x', data=x, compression='gzip', shuffle=True)
	f.create_dataset('y', data=y, compression='gzip', shuffle=True)


