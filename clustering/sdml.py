from __future__ import print_function, division
"""

Geometry learning algorithms
-------------------------------

Copyright (c) 2017 Johannes Buchner

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""


import numpy as np
import numpy
from numpy import exp
import scipy.linalg

class IdentityMetric(object):
	"""
	Input is output.
	"""
	def fit(self, x):
		pass
	def transform(self, x):
		return x
	def untransform(self, y):
		return y
	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

class SimpleScaling(object):
	"""
	Whitens by subtracting the mean and scaling by the 
	standard deviation of each axis.
	"""
	def __init__(self, verbose=False):
		self.verbose = verbose

	def fit(self, X, W=None):
		self.mean = numpy.mean(X, axis=0)
		X = X - self.mean
		self.scale = numpy.std(X, axis=0)
		if self.verbose: 'Scaling metric:', self.scale
	def transform(self, x):
		return (x - self.mean) / self.scale
	
	def untransform(self, y):
		return y * self.scale + self.mean

	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

class TruncatedScaling(object):
	"""
	Whitens by subtracting the mean and scaling by the 
	standard deviation of each axis. The scaling is discretized on 
	a log axis onto integers.
	"""
	def __init__(self, verbose=False):
		self.verbose = verbose
	def fit(self, X, W=None):
		self.mean = numpy.mean(X, axis=0)
		X = X - self.mean
		#scale = numpy.max(X, axis=0) - numpy.min(X, axis=0)
		scale = numpy.std(X, axis=0)
		scalemax = scale.max() * 1.001
		scalemin = scale.min()
		# round onto discrete log scale to avoid random walk
		logscale = (-numpy.log2(scale / scalemax)).astype(int)
		self.scale = 2**(logscale.astype(float))
		#print 'Scaling metric:', self.scale, '(from', scale, ')'
		if self.verbose: 'Discretized scaling metric:\n', logscale
	
	def transform(self, x):
		return (x - self.mean) / self.scale
	
	def untransform(self, y):
		return y * self.scale + self.mean

	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

