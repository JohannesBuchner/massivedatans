"""

Implementation of MultiEllipsoidal sampling via nestle

Copyright (c) 2017 Johannes Buchner

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



"""

import numpy
from numpy import exp, log, log10, pi
from nestle import bounding_ellipsoid, bounding_ellipsoids, sample_ellipsoids
from collections import defaultdict
import scipy.spatial, scipy.cluster
import matplotlib.pyplot as plt
import numpy
from numpy import exp, log, log10, pi, cos, sin
from nestle import bounding_ellipsoid, bounding_ellipsoids, sample_ellipsoids

def is_inside_unit_filter(u):
	return numpy.all(u >= 0) and numpy.all(u <= 1)


class BaseProposal(object):
	"""
	Base class for proposal function.
	
	:param scale: Scale of proposal
	:param adapt: Adaptation rule to use for scale, when new_chain is called.
	
	If adapt is False, no adaptation is done. If adapt is 'Sivia', the rule
	of Sivia & Skilling (2006) is used. If adapt is something else,
	a crude thresholding adaptation is used to gain ~50% acceptance.
	"""
	def __init__(self, adapt = False, scale = 1.):
		self.accepts = []
		self.adapt = adapt
		self.scale = scale
	"""
	Proposal function (to be overwritten)
	"""
	def propose(self, u, ndim, live_pointsu=None, is_inside_filter=None):
		return u
	"""
	Reset accept counters and adapt proposal (if activated).
	"""
	def new_chain(self, live_pointsu=None, is_inside_filter=None):
		if self.adapt and len(self.accepts) > 0:
			# adjust future scale based on acceptance rate
			m = numpy.mean(self.accepts)
			assert 0 <= m <= 1, (m, self.accepts)
			if self.adapt == 'sivia':
				if m > 0.5: self.scale *= exp(1./numpy.sum(self.accepts))
				else:       self.scale /= exp(1./(len(self.accepts) - numpy.sum(self.accepts)))
			elif self.adapt == 'sivia-neg-binom':
				# negative binomial rate estimator
				m = (sum(self.accepts) - 1) / (len(self.accepts) - 1.)
				if m > 0.5: self.scale *= exp(1./numpy.sum(self.accepts))
				else:       self.scale /= exp(1./(len(self.accepts) - numpy.sum(self.accepts)))
			elif self.adapt == 'step':
				#print 'adaptation:', m
				if m <= 0.1:
					self.scale /= 1.1
				elif m <= 0.3:
					self.scale /= 1.01
				elif m >= 0.9:
					self.scale *= 1.1
				elif m >= 0.7:
					self.scale *= 1.01
			else:
				assert False, self.adapt
			assert numpy.all(numpy.isfinite(self.scale)), self.scale
		self.accepts = []
	
	"""
	Add a point to the record.
	:param accepted: True if accepted, False if rejected.
	"""
	def accept(self, accepted):
		self.accepts.append(accepted)
	
	"""
	Print some stats on the acceptance rate
	"""
	def stats(self):
		print('Proposal %s stats: %.2f%% accepts' % (repr(self), 
			numpy.mean(self.accepts) * 100.))

class MultiScaleProposal(BaseProposal):
	"""Proposal over multiple scales, inspired by DNest. 
	Uses the formula
	
	:math:`x + n * 10^{l - s * u}`
	
	where l is the location, s is the scale and u is a uniform variate,
	and n is a normal variate.
	
	@see MultiScaleProposal
	"""
	def __init__(self, loc = -4.5, scale=1.5, adapt=False):
		# 10**(1.5 - 6 * u) (inspired by DNest)
		# a + (b - a) * u
		# a = 1.5, b = -4.5
		# a should increase for larger scales, decrease for smaller
		
		self.loc = loc
		BaseProposal.__init__(self, adapt=adapt, scale=scale)
	def __repr__(self):
		return 'MultiScaleProposal(loc=%s, scale=%s, adapt=%s)' % (self.loc, self.scale, self.adapt)
	def propose(self, u, ndim, live_pointsu=None, is_inside_filter=None):
		p = u + numpy.random.normal() * 10**(self.scale + (self.loc - self.scale) * numpy.random.uniform())
		p[p > 1] = 1
		p[p < 0] = 0
		#p = p - numpy.floor(p)
		return p


class FilteredUnitHARMProposal(BaseProposal):
	"""
	Unit HARM proposal.

	@see BaseProposal
	"""
	def __init__(self, adapt = False, scale = 1.):
		BaseProposal.__init__(self, adapt=False, scale=float(scale))
	
	def generate_direction(self, u, ndim, points):
		# generate unit direction
		x = numpy.random.normal(size=ndim)
		d = x / (x**2).sum()**0.5
		return d
	def new_chain(self, u, ndim, points, is_inside_filter):
		BaseProposal.new_chain(self)
		self.new_direction(u, ndim, points, is_inside_filter)
	def new_direction(self, u, ndim, points, is_inside_filter):
		d = self.generate_direction(u, ndim, points)
		#print('initial scale:', self.scale)
		# find end points
		forward_scale = self.scale
		# find a scale that is too large
		while True:
			assert forward_scale > 0
			p_for = u + d * forward_scale
			if is_inside_filter(p_for):
				# we are proposing too small. We should be outside
				forward_scale *= 2
				#print('too small, stepping further', forward_scale)
			else:
				break
		
		backward_scale = self.scale
		# find a scale that is too large
		while True:
			assert backward_scale > 0
			p_rev = u - d * backward_scale
			if is_inside_filter(p_rev):
				# we are proposing too small. We should be outside
				#print('too small, stepping back', backward_scale)
				backward_scale *= 2
			else:
				break
		# remember scale for next time:
		self.backward_scale = -backward_scale
		self.forward_scale = forward_scale
		self.direction = d
	
	def propose(self, u, ndim, points, is_inside_filter):
		# generate a random point between the two points.
		while True:
			#print('slice range:', (self.backward_scale, self.forward_scale))
			x = numpy.random.uniform(self.backward_scale, self.forward_scale)
			p = u + self.direction * x
			#assert self.forward_scale - self.backward_scale > 1e-100
			if x < 0:
				self.backward_scale = x
			else:
				self.forward_scale = x
			if is_inside_filter(p):
				if self.adapt:
					self.scale = self.forward_scale - self.backward_scale
					#print('adapting scale to', self.scale)
				return p
	
	def accept(self, accepted):
		# scale should not be modified
		pass
	
	def __repr__(self):
		return 'FilteredUnitHARMProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredMahalanobisHARMProposal(FilteredUnitHARMProposal):
	"""
	Mahalanobis HARM proposal.

	@see BaseProposal
	"""

	def generate_direction(self, u, ndim, points):
		# generate direction from mahalanobis metric
		metric = numpy.cov(numpy.transpose(points))
		assert metric.shape == (ndim,ndim), metric.shape
		x = numpy.random.multivariate_normal(numpy.zeros(ndim), metric)
		d = x / (x**2).sum()**0.5
		return d
	def __repr__(self):
		return 'FilteredMahalanobisHARMProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredUnitRandomSliceProposal(FilteredUnitHARMProposal):
	"""
	Unit Slice sampling proposal, random component-wise.

	@see BaseProposal
	"""
	def generate_direction(self, u, ndim, points):
		# choose a random base vector
		d = numpy.zeros(ndim)
		i = numpy.random.randint(ndim)
		d[i] = 1
		return d
	def __repr__(self):
		return 'FilteredUnitRandomSliceProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredUnitIterateSliceProposal(FilteredUnitHARMProposal):
	"""
	Unit Slice sampling proposal, iterative component-wise.

	@see BaseProposal
	"""
	def __init__(self, adapt = False, scale = 1.):
		BaseProposal.__init__(self, adapt=False, scale=float(scale))
		self.curindex = 0
	
	def generate_direction(self, u, ndim, points):
		# choose next base vector
		d = numpy.zeros(ndim)
		self.curindex = (self.curindex + 1) % ndim
		d[self.curindex] = 1
		return d
	def __repr__(self):
		return 'FilteredUnitIterateSliceProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredUnitRandomSliceProposal(FilteredUnitHARMProposal):
	"""
	Unit Slice sampling proposal, random component-wise.

	@see BaseProposal
	"""
	def generate_direction(self, u, ndim, points):
		# choose a random base vector
		d = numpy.zeros(ndim)
		i = numpy.random.randint(ndim)
		d[i] = 1
		return d
	def __repr__(self):
		return 'FilteredUnitRandomSliceProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class SliceConstrainer(object):
	"""
	Markov chain Monte Carlo proposals using the Metropolis update: 
	Do a number of steps, while adhering to boundary.
	"""
	def __init__(self, proposer = MultiScaleProposal(), nsteps = 10, nmaxsteps = 10000):
		self.proposer = proposer
		self.sampler = None
		# number of new directions
		self.nsteps = nsteps
		# number of narrowings
		self.nmaxsteps = nmaxsteps
	
	def draw_constrained(self, Lmins, priortransform, loglikelihood, ndim, 
			live_pointsu, **kwargs):
		i = numpy.random.randint(len(live_pointsu))
		ui = live_pointsu[i]
		xi = None
		naccepts = 0
		nevals = 0
		# new direction
		for i in range(self.nsteps):
			self.proposer.new_chain(ui, ndim, live_pointsu, is_inside_unit_filter)
			# narrow in until we get an accept
			for n in range(self.nmaxsteps):
				u = self.proposer.propose(ui, ndim, live_pointsu, is_inside_unit_filter)
				x = priortransform(u)
				L = loglikelihood(x)
				nevals += 1
				# MH accept rule
				# accept = L > Li or numpy.random.uniform() < exp(L - Li)
				# Likelihood-difference independent, because we do
				# exploration of the prior (full diffusion).
				# but only accept in constrained region, because that
				# is what we are exploring now.
				# accept = L >= Lmin
				####
				# For collaborative nested sampling it is sampling 
				# from the super-contour, so only one needs to work:
				accept = numpy.any(L >= Lmins)
				
				# tell proposer so it can scale
				self.proposer.accept(accept)
				if accept:
					ui, xi, Li = u, x, L
					naccepts += 1
					break
		if numpy.all(Li < Lmins):
			print()
			print('ERROR: SliceConstrainer could not find a point matching constraint!')
			print('ERROR: Proposer stats:')
			self.proposer.stats()
			assert numpy.all(Li < Lmins), (Li, Lmins, self.nmaxsteps, numpy.mean(self.proposer.accepts), len(self.proposer.accepts))
		if xi is None:
			xi = priortransform(ui)
		return ui, xi, Li, nevals

	def stats(self):
		return self.proposer.stats()

