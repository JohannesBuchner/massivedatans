"""
Copyright: Johannes Buchner (C) 2013

Modular, Pythonic Implementation of Nested Sampling
"""

import numpy
from numpy import exp, log, log10, pi
import progressbar

"""
status_symbols = {0:' ', 1:u"\u2581", 2:u"\u2582", 
	3:u"\u2583", 4:u"\u2583", 
	5:u"\u2584", 6:u"\u2584", 7:u"\u2584", 8:u"\u2584", 
}
for i in range(8, 16):
	status_symbols[i+1] = u"\u2585"
for i in range(16, 32):
	status_symbols[i+1] = u"\u2586"
for i in range(32, 64):
	status_symbols[i+1] = u"\u2587"
for i in range(64, 128):
	status_symbols[i+1] = u"\u2588"
"""

status_symbols = {
	0:' ', 
	1:u"\u2581", 
	2:u"\u2582", 
	3:u"\u2583", 
	4:u"\u2584", 5:u"\u2584", 
	6:u"\u2585", 7:u"\u2585", 
	8:u"\u2586", 9:u"\u2586", 
	10:u"\u2587", 11:u"\u2587", 12:u"\u2587", 13:u"\u2587",  14:u"\u2587", 
	15:u"\u2588", 16:u"\u2588", 17:u"\u2588", 18:u"\u2588", 19:u"\u2588", 
}

class MultiNestedSampler(object):
	"""
	Samples points, always replacing the worst live point, forever.
	
	This implementation always removes and replaces one point (r=1),
	and does so linearly (no parallelisation).
	
	This class is implemented as an iterator.
	"""
	def __init__(self, priortransform, multi_loglikelihood, superset_draw_constrained, draw_constrained, 
			ndata, ndim = None, nlive_points = 200, draw_global_uniform = None,
			nsuperset_draws = 10):
		self.nlive_points = nlive_points
		self.nsuperset_draws = nsuperset_draws
		self.priortransform = priortransform
		self.multi_loglikelihood = multi_loglikelihood
		self.superset_draw_constrained = superset_draw_constrained
		self.draw_constrained = draw_constrained
		self.samples = []
		self.ndim = ndim
		if ndim is not None:
			self.draw_global_uniform = lambda: numpy.random.uniform(0, 1, size=ndim)
		else:
			raise ArgumentError("either pass ndim or draw_global_uniform")
			self.draw_global_uniform = draw_global_uniform
		# draw N starting points from prior
		pointpile = []
		live_pointsp = [None] * nlive_points
		live_pointsu = [None] * nlive_points
		live_pointsx = [None] * nlive_points
		live_pointsL = [None] * nlive_points
		print 'generating initial %d live points' % (nlive_points)
		data_mask = numpy.ones(ndata) == 1
		for i in range(nlive_points):
			u = self.draw_global_uniform()
			x = priortransform(u)
			L = multi_loglikelihood(x, data_mask=data_mask)
			live_pointsp[i] = [len(pointpile)]*ndata
			pointpile.append(u)
			live_pointsu[i] = [u]*ndata
			live_pointsx[i] = [x]*ndata
			live_pointsL[i] = L
			self.samples.append([live_pointsu[i], live_pointsx[i], live_pointsL[i]])
		print 'generated %d live points' % (len(live_pointsu))
		self.pointpile = numpy.array(pointpile)
		self.live_pointsp = numpy.array(live_pointsp)
		self.live_pointsu = numpy.array(live_pointsu)
		self.live_pointsx = numpy.array(live_pointsx)
		self.live_pointsL = numpy.array(live_pointsL)
		self.Lmax = self.live_pointsL.max(axis=0)
		assert self.Lmax.shape == (ndata,)
		self.ndraws = nlive_points
		self.shelves = [[] for _ in range(ndata)]
		self.ndata = ndata
		self.data_mask_all = numpy.ones(self.ndata) == 1
	
	def get_unique_points(self, allpoints):
		d = allpoints.reshape((-1,self.ndim))
		b = d.view(numpy.dtype((numpy.void, d.dtype.itemsize * d.shape[1])))
		_, idx = numpy.unique(b, return_index=True)
		return d[idx]
	
	def get_unique_pointsp(self, allpoints):
		idx = numpy.unique(allpoints)
		return self.pointpile[idx]
	
	def prepare(self):
		live_pointsu = self.live_pointsu
		live_pointsx = self.live_pointsx
		live_pointsL = self.live_pointsL
		Lmins = live_pointsL.min(axis=0)
		Lmini = live_pointsL.argmin(axis=0)
		# clean up shelves
		for d in range(self.ndata):
			self.shelves[d] = [(pj, uj, xj, Lj) for (pj, uj, xj, Lj) in self.shelves[d] if Lj > Lmins[d]]
		
		#print 'finding unique points...'
		#d = live_pointsu.reshape((-1,self.ndim))
		#b = d.view(numpy.dtype((numpy.void, d.dtype.itemsize * d.shape[1])))
		#_, idx = numpy.unique(b, return_index=True)
		#all_global_live_pointsu = d[idx]
		#all_global_live_pointsu = self.get_unique_points(live_pointsu)
		all_global_live_pointsu = self.get_unique_pointsp(self.live_pointsp)
		#global_live_pointsu_set = set()
		#global_live_pointsu = []
		#global_live_pointsx = []
		#global_live_pointsL = []
		#for d in range(self.ndata):
		#	global_live_pointsu_set.update([tuple(u) for u in live_pointsu[:,d]])
			#for u, x, L in zip(live_pointsu[:,d], live_pointsx[:,d], live_pointsL[:,d]):
				#if tuple(u) not in global_live_pointsu_set:
				#	global_live_pointsu_set.add(tuple(u))
				#	global_live_pointsu.append(u)
					#global_live_pointsx.append(x)
					#global_live_pointsL.append(L)
		#all_global_live_pointsu = numpy.array([list(u) for u in global_live_pointsu_set])
		all_Lmin = live_pointsL.min()
		#all_global_live_pointsL = numpy.array(global_live_pointsL)
		#all_global_live_pointsx = numpy.array(global_live_pointsx)
		return all_global_live_pointsu, all_Lmin, Lmins, Lmini
	
	def shelf_status(self):
		print 'shelf status: %s' % ''.join([status_symbols.get(len(shelf), 'X') for shelf in self.shelves])
	
	def __next__(self):
		live_pointsp = self.live_pointsp
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
		#print
		#print 'housekeeping...'
		all_global_live_pointsu, all_Lmin, Lmins, Lmini = self.prepare()
		iter = 0
		while True:
			iter += 1
			data_mask = numpy.array([len(self.shelves[d]) == 0 for d in range(self.ndata)])
			if not data_mask.any():
				# all have something in their shelves
				break
			if self.nsuperset_draws >= 0:
				sample_subset = iter > self.nsuperset_draws
			else:
				nsuperset_draws = -self.nsuperset_draws
				sample_subset = iter > nsuperset_draws
				# iter: 1 2 3 4 5 6
				# nsuperset_draws = 1
				# iter - nsuperset_draws = 0 1 2 3 4 5 6
				cut_level = max(0, 5 - (iter - nsuperset_draws))
				# cut_level = 5-0 5-1 5-2 5-3 5-4 5-5 0 0 0 
				data_mask = numpy.array([len(self.shelves[d]) <= cut_level for d in range(self.ndata)])
			
			#print 'finding current live point set ...'
			# check shelves
			if sample_subset:
				#d = live_pointsu[:,data_mask,:].reshape((-1,self.ndim))
				#b = d.view(numpy.dtype((numpy.void, d.dtype.itemsize * d.shape[1])))
				#_, idx = numpy.unique(b, return_index=True)
				#global_live_pointsu = d[idx]
				global_live_pointsu = self.get_unique_pointsp(self.live_pointsp[:,data_mask])
				#global_live_pointsu = self.get_unique_points(live_pointsu[:,data_mask,:])
				#Lmin = live_pointsu[:,data_mask].flatten()[idx].min()
				
				#global_live_pointsu_set = set()
				#global_live_pointsu = []
				##global_live_pointsx = []
				##global_live_pointsL = []
				#Lmin = numpy.inf
				#for d in range(self.ndata):
				#	if len(self.shelves[d]) == 0:
				#		for u, x, L in zip(live_pointsu[:,d], live_pointsx[:,d], live_pointsL[:,d]):
				#			if tuple(u) not in global_live_pointsu_set:
				#				global_live_pointsu_set.add(tuple(u))
				#				global_live_pointsu.append(u)
				#				#global_live_pointsx.append(x)
				#				#global_live_pointsL.append(L)
				#				Lmin = min(Lmin, L)
				#data_mask = numpy.array(data_mask)
				##global_live_pointsu = numpy.array(global_live_pointsu)
				##global_live_pointsL = numpy.array(global_live_pointsL)
				##Lmin = numpy.min(global_live_pointsL)
			else:
				data_mask = self.data_mask_all
				global_live_pointsu = all_global_live_pointsu
				#global_live_pointsx = all_global_live_pointsx
				#global_live_pointsL = all_global_live_pointsL
				Lmin = all_Lmin
			
			self.shelf_status()
			print 'live point set: %d from %d datasets, %s' % (len(global_live_pointsu), data_mask.sum(), 'focussed set constrained draw' if sample_subset else 'super-set constrained draw')
			
			if sample_subset:
				uj, xj, Lj, n = self.draw_constrained(
					Lmins=Lmins[data_mask], 
					priortransform=self.priortransform, 
					loglikelihood=lambda params: self.multi_loglikelihood(params, data_mask), 
					ndim=self.ndim,
					draw_global_uniform=self.draw_global_uniform,
					live_pointsu = global_live_pointsu,
					#live_pointsx = global_live_pointsx,
					#live_pointsL = global_live_pointsL,
					max_draws=10000,
				)
			else:
				uj, xj, Lj, n = self.superset_draw_constrained(
					Lmins=Lmins, #[data_mask], 
					priortransform=self.priortransform, 
					loglikelihood=lambda params: self.multi_loglikelihood(params, self.data_mask_all), 
					ndim=self.ndim,
					draw_global_uniform=self.draw_global_uniform,
					live_pointsu = global_live_pointsu,
					#live_pointsx = global_live_pointsx,
					#live_pointsL = global_live_pointsL,
					max_draws=1000,
				)
			
			self.ndraws += int(n)
			ppi = len(self.pointpile)
			self.pointpile = numpy.vstack((self.pointpile, [uj]))
			for j, d in enumerate(numpy.where(data_mask)[0]):
				if Lj[j] > Lmins[d]:
					n = len(self.shelves[d])
					# to insert at position n
					# there must be n elements smaller than 
					# Lj[j] in self.shelves[d] and self.live_pointsL[:,d]
					nsmaller = (live_pointsL[:,d] < Lj[j]).sum()
					if nsmaller >= n or nsmaller + sum([Li < Lj[j] for _, _, _, Li in self.shelves[d]]) >= n:
						self.shelves[d].append((ppi, uj, xj, Lj[j]))
			
			# we got a new point
			#print 'new point: > %.1f' % Lmins[data_mask].min() #, (Lj>Lmins[data_mask])*1
		
		# pop: for every data entry, advance one point
		uis = []
		xis = []
		Lis = []
		ujs = []
		xjs = []
		Ljs = []
		for d in range(self.ndata):
			Lmin = Lmins[d]
			i = Lmini[d]
			uis.append(live_pointsu[i,d])
			xis.append(live_pointsx[i,d])
			Lis.append(live_pointsL[i,d])
			pj, uj, xj, Lj = self.shelves[d].pop(0)
			ujs.append(uj)
			xjs.append(xj)
			Ljs.append(Lj)
			assert Lj > Lmin, (Lj, Lmin)
			self.live_pointsp[i,d] = pj
			live_pointsu[i,d] = uj
			live_pointsx[i,d] = xj
			live_pointsL[i,d] = Lj
		self.Lmax = live_pointsL.max(axis=0)
		#print 'Lmax:', self.Lmax
		assert self.Lmax.shape == (self.ndata,)
		self.samples.append([ujs, xjs, Ljs])
		return uis, xis, Lis
	
	def remainder(self):
		indices = [] #numpy.empty((self.ndata, self.nlive_points))
		#indices = self.live_pointsL.argsort(axis=0)
		#assert indices.shape == (self.ndata, self.nlive_points), (indices.shape, (self.ndata, self.nlive_points))
		for d in range(self.ndata):
			i = numpy.argsort(self.live_pointsL[:,d]) #[L[d] for L in self.live_pointsL])
			#indices[d] = i
			indices.append(i)
		#indices = numpy.array(indices)
		#indices2 = numpy.argsort(self.live_pointsL, axis=1)
		#assert numpy.all(indices == indices2)
		for i in range(self.nlive_points):
			#j = indices[i,:]
			#yield self.live_pointsu[i,:], self.live_pointsx[i,:], self.live_pointsL[i,:]
			u = [self.live_pointsu[indices[d][i],d] for d in range(self.ndata)]
			x = [self.live_pointsx[indices[d][i],d] for d in range(self.ndata)]
			L = [self.live_pointsL[indices[d][i],d] for d in range(self.ndata)]
			#print 'remainder:', i, L
			yield u, x, numpy.asarray(L)
			#yield self.live_pointsu[indices[:,i],:], self.live_pointsx[indices[:,i],:], self.live_pointsL[indices[:,i],:]
	
	def next(self):
		return self.__next__()
	def __iter__(self):
		while True: yield self.__next__()
		
__all__ = [MultiNestedSampler]

