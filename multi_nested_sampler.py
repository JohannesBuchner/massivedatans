"""
Copyright: Johannes Buchner (C) 2013

Modular, Pythonic Implementation of Nested Sampling
"""

import numpy
from numpy import exp, log, log10, pi
import progressbar

class NestedSampler(object):
	"""
	Samples points, always replacing the worst live point, forever.
	
	This implementation always removes and replaces one point (r=1),
	and does so linearly (no parallelisation).
	
	This class is implemented as an iterator.
	"""
	def __init__(self, priortransform, multi_loglikelihood, draw_constrained, 
			ndata, ndim = None, nlive_points = 200, draw_global_uniform = None):
		self.nlive_points = nlive_points
		self.priortransform = priortransform
		self.multi_loglikelihood = multi_loglikelihood
		self.draw_constrained = draw_constrained
		self.samples = []
		self.ndim = ndim
		if ndim is not None:
			self.draw_global_uniform = lambda: numpy.random.uniform(0, 1, size=ndim)
		else:
			raise ArgumentError("either pass ndim or draw_global_uniform")
			self.draw_global_uniform = draw_global_uniform
		# draw N starting points from prior
		live_pointsu = [None] * nlive_points
		live_pointsx = [None] * nlive_points
		live_pointsL = numpy.empty(nlive_points, ndata)
		for i in range(nlive_points):
			u = self.draw_global_uniform()
			x = priortransform(u)
			L = multi_loglikelihood(x)
			live_pointsu[i], live_pointsx[i], live_pointsL[i] = u, x, L
			self.samples.append([u, x, L])
		self.live_pointsu = live_pointsu
		self.live_pointsx = live_pointsx
		self.live_pointsL = live_pointsL
		self.Lmax = self.live_pointsL.max(axis=0)
		self.ndraws = nlive_points
	
	def __next__(self):
		live_pointsu = self.live_pointsu
		live_pointsx = self.live_pointsx
		live_pointsL = self.live_pointsL
		# select worst point, lowest likelihood
		
		# there is no globally worst point. 
		# A point can be worst for one but best for the other.
		# For the latter it would not make sense to replace it first.
		# --> need reactive nested sampling to add edges
		# in that graph, I can always insert, even if not the worst 
		# point
		# the edges inserted would be partial, i.e. only valid for some
		# data -- namely those where the likelihood increased
		# 
		
		i = numpy.unravel_index(live_pointsL.argmin(), live_pointsL.shape)
		ui = live_pointsu[i]
		xi = live_pointsx[i]
		Li = live_pointsL[i]
		
		# choose random 
		k = numpy.random.randint(0, self.nlive_points - 1)
		if k >= i: # don't choose the same point
			k += 1
		
		# find replacement
		uj, xj, Lj, n = self.draw_constrained(
			Lmin=Li, 
			priortransform=self.priortransform, 
			loglikelihood=self.loglikelihood, 
			previous=self.samples,
			ndim=self.ndim,
			draw_global_uniform=self.draw_global_uniform,
			startu = live_pointsu[k], 
			startx = live_pointsx[k], 
			startL = live_pointsL[k],
			starti = i,
			live_pointsu = live_pointsu,
			live_pointsx = live_pointsx,
			live_pointsL = live_pointsL,)
		
		live_pointsu[i] = uj
		live_pointsx[i] = xj
		live_pointsL[i] = Lj
		self.Lmax = max(Lj, self.Lmax)
		self.samples.append([uj, xj, Lj])
		self.ndraws += int(n)
		return ui, xi, Li
	def remainder(self):
		indices = self.live_pointsL.argsort()
		for i in indices:
			yield self.live_pointsu[i], self.live_pointsx[i], self.live_pointsL[i]
	def next(self):
		return self.__next__()
	def __iter__(self):
		while True: yield self.__next__()
		
__all__ = [NestedSampler]

