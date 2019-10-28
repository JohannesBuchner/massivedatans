from __future__ import print_function, division
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

class MultiEllipsoidalConstrainer(object):
	def __init__(self, rebuild_every = 1000, verbose = False, enlarge=3.):
		self.iter = 0
		self.ndraws_since_rebuild = 0
		self.rebuild_every = int(rebuild_every)
		self.enlarge = enlarge
		self.verbose = verbose
		self.ells = None
		self.last_cluster_points = None
	
	def update(self, points):
		# volume is larger than standard Ellipsoid computation
		# because we have a superset of various likelihood contours
		# increase proportional to number of points
		pointvol = exp(-self.iter / self.nlive_points) * (len(points) * 1. / self.nlive_points) / self.nlive_points
		self.ells = bounding_ellipsoids(numpy.asarray(points), pointvol=pointvol)
		for ell in self.ells:
			ell.scale_to_vol(ell.vol * self.enlarge)

	def generate(self, ndim):
		ntotal = 0
		N = 10000
		while True:
			u = sample_ellipsoids(self.ells, rstate=numpy.random)
			if not (numpy.all(u > 0.) and numpy.all(u < 1.)):
				continue
			yield u, ntotal
	
	def rebuild(self, u, ndim):
		if self.last_cluster_points is not None and \
			len(self.last_cluster_points) == len(u) and \
			numpy.all(self.last_cluster_points == u):
			# do nothing if everything stayed the same
			return
		
		self.update(points=u)
		self.last_cluster_points = u
		
		self.generator = self.generate(ndim)
	
	def _draw_constrained_prepare(self, Lmins, priortransform, loglikelihood, live_pointsu, ndim, **kwargs):
		rebuild = self.ndraws_since_rebuild > self.rebuild_every or self.ells is None
		if rebuild:
			print('rebuild triggered at call')
			self.rebuild(numpy.asarray(live_pointsu), ndim)
			self.ndraws_since_rebuild = 0
		assert self.generator is not None
		return rebuild
	
	def draw_constrained(self, Lmins, priortransform, loglikelihood, live_pointsu, ndim, iter, nlive_points, **kwargs):
		ntoaccept = 0
		self.iter = iter
		self.nlive_points = nlive_points
		#print 'MLFriends trying to replace', Lmins
		rebuild = self._draw_constrained_prepare(Lmins, priortransform, loglikelihood, live_pointsu, ndim, **kwargs)
		while True:
			#print '    starting generator ...'
			for u, ntotal in self.generator:
				assert (u >= 0).all() and (u <= 1).all(), u
				x = priortransform(u)
				L = loglikelihood(x)
				ntoaccept += 1
				self.ndraws_since_rebuild += 1

				if numpy.any(L > Lmins):
					# yay, we win
					#print 'accept after %d tries' % ntoaccept
					return u, x, L, ntoaccept
				
				# if running very inefficient, optimize clustering 
				#     if we haven't done so at the start
				if not rebuild and self.ndraws_since_rebuild > self.rebuild_every:
					rebuild = True
					print('Ellipsoid rebuild triggered after %d draws' % self.ndraws_since_rebuild)
					self.rebuild(numpy.asarray(live_pointsu), ndim)
					self.ndraws_since_rebuild = 0
					break
				
