from __future__ import print_function, division
"""

Sampler
----------

Copyright (c) 2017 Johannes Buchner

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



"""
import numpy
from numpy import exp, log, log10, pi
import progressbar
import igraph
from collections import defaultdict

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

def find_nsmallest(n, arr1, arr2):
	# old version
	arr = numpy.hstack((arr1, arr2))
	arr.sort()
	return arr[n]

def find_nsmallest(n, arr1, arr2):
	# new version, faster because it does not need to sort everything
	arr = numpy.concatenate((arr1, arr2))
	return numpy.partition(arr, n)[n]

class MultiNestedSampler(object):
	"""
	Samples points, always replacing the worst live point, forever.
	
	This implementation always removes and replaces one point (r=1),
	and does so linearly (no parallelisation).
	
	This class is implemented as an iterator.
	"""
	def __init__(self, priortransform, multi_loglikelihood, superset_draw_constrained, individual_draw_constrained, draw_constrained, 
			ndata, ndim, nlive_points = 200, draw_global_uniform = None,
			nsuperset_draws = 10, use_graph=False):
		self.nlive_points = nlive_points
		self.nsuperset_draws = nsuperset_draws
		self.priortransform = priortransform
		self.real_multi_loglikelihood = multi_loglikelihood
		self.multi_loglikelihood = multi_loglikelihood
		self.superset_draw_constrained = superset_draw_constrained
		self.individual_draw_constrained = individual_draw_constrained
		self.draw_constrained = draw_constrained
		#self.samples = []
		self.global_iter = 0
		self.ndim = ndim
		self.ndata = ndata
		self.superpoints = []
		# lazy building of graph
		self.use_graph = use_graph
		self.membership_graph = None
		self.last_graph = None
		self.last_graph_selection = None
		self.point_data_map = None
		# draw N starting points from prior
		pointpile = []
		pointpilex = []
		live_pointsp = [None] * nlive_points
		#live_pointsu = [None] * nlive_points
		#live_pointsx = [None] * nlive_points
		live_pointsL = [None] * nlive_points
		
		print('generating initial %d live points' % (nlive_points))
		data_mask = numpy.ones(ndata) == 1

		for i in range(nlive_points):
			u = self.draw_global_uniform()
			x = priortransform(u)
			L = multi_loglikelihood(x, data_mask=data_mask)
			p = len(pointpile)
			live_pointsp[i] = [p]*ndata
			pointpile.append(u)
			pointpilex.append(x)
			#self.global_iter += 1
			#live_pointsu[i] = [u]*ndata
			#live_pointsx[i] = [x]*ndata
			live_pointsL[i] = L
			self.superpoints.append(p)
			#self.samples.append([live_pointsu[i], live_pointsx[i], live_pointsL[i]])
		print('generated %d live points' % (nlive_points))
		self.pointpile = numpy.array(pointpile)
		self.pointpilex = numpy.array(pointpilex)
		self.live_pointsp = numpy.array(live_pointsp)
		#self.live_pointsu = numpy.array(live_pointsu)
		#self.live_pointsx = numpy.array(live_pointsx)
		self.live_pointsL = numpy.array(live_pointsL)
		self.Lmax = self.live_pointsL.max(axis=0)
		self.data_mask_all = numpy.ones(self.ndata) == 1
		self.real_data_mask_all = numpy.ones(self.ndata) == 1
		assert self.Lmax.shape == (ndata,)
		self.ndraws = nlive_points
		self.shelves = [[] for _ in range(ndata)]
		
		self.dump_iter = 1
	
	def draw_global_uniform(self):
		return numpy.random.uniform(0, 1, size=self.ndim)
	
	def get_unique_points(self, allpoints):
		d = allpoints.reshape((-1,self.ndim))
		b = d.view(numpy.dtype((numpy.void, d.dtype.itemsize * d.shape[1])))
		_, idx = numpy.unique(b, return_index=True)
		return d[idx]
	
	def get_unique_pointsp(self, allpoints):
		idx = numpy.unique(allpoints)
		return self.pointpile[idx], idx
	
	def prepare(self):
		live_pointsL = self.live_pointsL
		Lmins = live_pointsL.min(axis=0)
		Lmini = live_pointsL.argmin(axis=0)
		# clean up shelves
		for d in range(self.ndata):
			self.shelves[d] = [(pj, uj, xj, Lj) for (pj, uj, xj, Lj) in self.shelves[d] if Lj > Lmins[d]]
		all_global_live_pointsu, all_global_live_pointsp = self.get_unique_pointsp(self.live_pointsp)
		all_Lmin = live_pointsL.min()
		return all_global_live_pointsu, all_global_live_pointsp, all_Lmin, Lmins, Lmini
	
	def shelf_status(self):
		print('shelf status: %s' % ''.join([status_symbols.get(len(shelf), 'X') for shelf in self.shelves]))
	
	def cut_down(self, surviving):
		# delete some data sets
		self.live_pointsp = self.live_pointsp[:,surviving]
		self.live_pointsL = self.live_pointsL[:,surviving]
		self.shelves = [shelf for s, shelf in zip(surviving, self.shelves) if s]
		self.ndata = surviving.sum()
		self.Lmax = self.live_pointsL.max(axis=0)
		self.data_mask_all = numpy.ones(self.ndata) == 1
		self.real_data_mask_all[self.real_data_mask_all] = surviving
		def multi_loglikelihood_subset(params, mask):
			subset_mask = self.real_data_mask_all.copy()
			subset_mask[subset_mask] = mask
			return self.real_multi_loglikelihood(params, subset_mask)
			
		self.multi_loglikelihood = multi_loglikelihood_subset
		# rebuild graph because igraph does not support renaming nodes
		self.membership_graph = None
		self.point_data_map = None
		self.last_graph = None
		self.last_graph_selection = None
		#if self.point_data_map is not None:
		#	for d, s in enumerate(surviving)
		#		if s: continue
		#		for p in self.live_pointsp[:,d]:
		#			self.point_data_map[p].add(d)
			
	
	def rebuild_graph(self):
		if self.membership_graph is None:
			print('constructing graph...')
			graph = igraph.Graph(directed=False)
			# pointing from live_point to member
			for i in numpy.where(self.data_mask_all)[0]:
				graph.add_vertex("n%d" % i, id=i, vtype=0)
			for p in range(len(self.pointpile)):
				graph.add_vertex("p%d" % p, id=p, vtype=1)
			edges = []
			for i in numpy.where(self.data_mask_all)[0]:
				#graph.add_vertex("n%d" % i, id=i, vtype=0)
				edges += [("n%d" % i, "p%d" % p) for p in self.live_pointsp[:,i]]
			print('connecting graph ...')
			graph.add_edges(edges)
			print('constructing graph done.')
			self.membership_graph = graph
	
	def rebuild_map(self):
		if self.point_data_map is None:
			print('constructing map...')
			# pointing from live_point to member
			self.point_data_map = defaultdict(set)
			for i in range(self.ndata):
				for p in self.live_pointsp[:,i]:
					self.point_data_map[p].add(i)
			print('constructing map done.')
	
	
	def generate_subsets_nograph(self, data_mask, allp):
		# generate data subsets which share points.
		selected = numpy.where(data_mask)[0]
		all_selected = len(selected) == len(data_mask)
		firstmember = selected[0]
		if len(selected) == 1:
			# trivial case:
			# requested only a single slot, so return its live points
			yield data_mask, self.live_pointsp[:,firstmember]
			return
		
		if not all_selected:
			allp = numpy.unique(self.live_pointsp[:,selected].flatten())
		
		if len(allp) < 2 * self.nlive_points:
			print('generate_subsets: only %d unique live points known, so connected' % len(allp))
			# if fewer than 2*nlive unique points are known, 
			# some must be shared between data sets.
			# So no disjoint data sets
			yield data_mask, allp
			return
		
		if len(self.superpoints) > 0:
			print('generate_subsets: %d superpoints known, so connected' % len(self.superpoints))
			# there are some points shared by all data sets
			# so no disjoint data sets
			yield data_mask, allp
			return
		
		self.rebuild_map()
		to_handle = data_mask.copy()
		while to_handle.any():
			firstmember = numpy.where(to_handle)[0][0]
			to_handle[firstmember] = False
			members = [firstmember]
			# get live points of this member
			member_live_pointsp = self.live_pointsp[:,firstmember].tolist()
			# look through to_handle for entries and check if they have the points
			i = 0
			while True:
				if i >= len(member_live_pointsp) or not to_handle.any():
					break
				p = member_live_pointsp[i]
				newmembers = [m for m in self.point_data_map[p] if to_handle[m]]
				print(newmembers)
				members += newmembers
				for newp in numpy.unique(self.live_pointsp[:,newmembers]):
					if newp not in member_live_pointsp:
						member_live_pointsp.append(newp)
				to_handle[newmembers] = False
				i = i + 1
			
			# now we have our members and live points
			member_data_mask = numpy.zeros(len(data_mask), dtype=bool)
			member_data_mask[members] = True
			#print 'returning:', member_data_mask, member_live_pointsp
			yield member_data_mask, member_live_pointsp

	def generate_subsets_graph(self, data_mask, allp):
		# generate data subsets which share points.
		selected = numpy.where(data_mask)[0]
		all_selected = len(selected) == len(data_mask)
		firstmember = selected[0]
		if len(selected) == 1:
			# trivial case:
			# requested only a single slot, so return its live points
			yield data_mask, self.live_pointsp[:,firstmember]
			return
		
		if not all_selected:
			allp = numpy.unique(self.live_pointsp[:,selected].flatten())
		
		if len(allp) < 2 * self.nlive_points:
			print('generate_subsets: only %d unique live points known, so connected' % len(allp))
			# if fewer than 2*nlive unique points are known, 
			# some must be shared between data sets.
			# So no disjoint data sets
			yield data_mask, allp
			return
		
		if len(self.superpoints) > 0:
			print('generate_subsets: %d superpoints known, so connected' % len(self.superpoints))
			# there are some points shared by all data sets
			# so no disjoint data sets
			yield data_mask, allp
			return
		
		self.rebuild_graph()
		if all_selected:
			graph = self.membership_graph
		else:
			graph = self._generate_subsets_graph_create_subgraph(data_mask, allp)
		
		for sub_data_mask, sub_points in self._generate_subsets_graph_subgraphs(graph, data_mask, all_selected, allp):
			yield sub_data_mask, sub_points
	
	def _generate_subsets_graph_create_subgraph(self, data_mask, allp):
		# need to look at the subgraph with only the selected
		# dataset nodes
		members  = ['n%d' % v for v, sel in enumerate(data_mask) if sel]
		members += ['p%d' % p for p in allp]
		# if the previous graph had all these nodes (or more)
		if self.last_graph is not None and self.last_graph_selection[data_mask].all():
			# re-using previously cut-down graph
			# this may speed things up because we have to cut less
			print('generate_subsets: re-using previous graph')
			prevgraph = self.last_graph
		else:
			# not a super-set, need to start with whole graph
			prevgraph = self.membership_graph
		
		graph = prevgraph.subgraph(members)
		self.last_graph = graph
		self.last_graph_selection = data_mask
		return graph
	
	
	def _generate_subsets_graph_subgraphs(self, graph, data_mask, all_selected, allp):
		# we could test here with graph.is_connected() first
		# but if it is connected,  then it takes as long as clusters()
		# and if it not connected, we have to call clusters() anyways.
		subgraphs = graph.clusters()
		assert len(subgraphs) > 0
		
		# single-node subgraphs can occur when 
		# a live point is not used anymore
		# a real subgraph has to have a data point and its live points, 
		# so at least nlive_points+1 entries
		subgraphs = [subgraph for subgraph in subgraphs if len(subgraph) > 1]
		
		if len(subgraphs) == 1:
			yield data_mask, allp
			return
		
		# then identify disjoint subgraphs
		for subgraph in subgraphs:
			member_data_mask = numpy.zeros(len(data_mask), dtype=bool)
			member_live_pointsp = []
			for vi in subgraph:
				att = graph.vs[vi].attributes()
				#print '    ', att
				if att['vtype'] == 0:
					i = att['id']
					member_data_mask[i] = True
				else:
					p = att['id']
					member_live_pointsp.append(p)
			if member_data_mask.any():
				yield member_data_mask, member_live_pointsp
			#else:
			#	print 'skipping node-free subgraph:', [self.membership_graph.vs[vi].attributes()['name'] for vi in subgraph]
			#	print graph

	def __next__(self):
		# select worst point, lowest likelihood and replace
		live_pointsL = self.live_pointsL
		superset_membersets = None
		
		print('iteration %d' % self.global_iter)
		all_global_live_pointsu, all_global_live_pointsp, all_Lmin, Lmins, Lmini = self.prepare()
		iter = 0
		while True:
			iter += 1
			empty_mask = numpy.array([len(self.shelves[d]) == 0 for d in range(self.ndata)])
			if not empty_mask.any():
				# all have something in their shelves
				break
			
			# if superset draws enabled, do some of these first. 
			sample_subset = iter > self.nsuperset_draws
			
			if sample_subset:
				# subset draw: focus on filling empty ones
				data_mask = empty_mask
				# cut_level = 5 4 3 2 1 0 0 0 0 
				#cut_level = max(0, 5 - (iter - self.nsuperset_draws))
				#data_mask = numpy.array([len(self.shelves[d]) <= cut_level for d in range(self.ndata)])
				global_live_pointsu, global_live_pointsp = self.get_unique_pointsp(self.live_pointsp[:,data_mask])
			else:
				# super-set draw, try to fill all/any
				data_mask = self.data_mask_all
				global_live_pointsu = all_global_live_pointsu
				global_live_pointsp = all_global_live_pointsp
				Lmin = all_Lmin
			use_rebuilding_draw = sample_subset
			
			self.shelf_status()
			# if the data sets do not share any live points, 
			# it does not make sense to analyse them jointly
			# so we break them up into membersets here, stringing
			# together those that do.
			
			# if a previous superset draw did the decomposition already,
			# just reuse it
			if superset_membersets is not None and not sample_subset:
				membersets = superset_membersets
			elif self.use_graph:
				membersets = list(self.generate_subsets_graph(data_mask, global_live_pointsp))
			else:
				membersets = list(self.generate_subsets_nograph(data_mask, global_live_pointsp))
			
			if not sample_subset and superset_membersets is None:
				# store superset decomposition
				superset_membersets = membersets
			
			assert len(membersets) > 0
			if len(membersets) > 1:
				# if the data is split, regions need to be 
				# rebuilt for every group
				use_rebuilding_draw = True
			
			for ji, (joint_data_mask, joint_live_pointsp) in enumerate(membersets):
				print('live point set %d/%d: %d from %d datasets, %s' % (
					ji+1, len(membersets), len(joint_live_pointsp), 
					joint_data_mask.sum(), 
					'focussed set constrained draw' if sample_subset else 'super-set constrained draw'))
				joint_live_pointsu = self.pointpile[joint_live_pointsp]
				#print 'members:', joint_data_mask.shape, joint_live_pointsu.shape
				max_draws = 1000
				njoints = joint_data_mask.sum()
				joint_indices = numpy.where(joint_data_mask)[0]
				firstd = joint_indices[0]
				# if it is the only dataset and we need an entry here, try longer
				if njoints == 1 and len(self.shelves[firstd]) == 0:
					max_draws = 100000
				
				# if there is more than one memberset and this one is full, 
				# we do not need to do anything
				# this should be a rare occasion
				if len(membersets) > 1 and not sample_subset and all([len(self.shelves[d]) > 0 for d in joint_indices]):
					continue

				# Lmin needs to be corrected. It is the lowest L, but
				# this may not be useful for making a draw. 
				Lmins_higher = Lmins[joint_indices].copy()
				for j, d in enumerate(joint_indices):
					n = len(self.shelves[d])
					if n == 0:
						# relevant only for non-empty shelves
						continue
					# to insert at position n
					# there must be n elements smaller
					# in self.shelves[d] and self.live_pointsL[:,d]
					Lmins_higher[j] = find_nsmallest(n, live_pointsL[:,d], [Li for _, _, _, Li in self.shelves[d]])
				
				if njoints == 1:
					# only a single data set, we can keep the same region for longer
					real_firstd = numpy.where(self.real_data_mask_all)[0][firstd]
					draw_constrained = self.individual_draw_constrained(real_firstd, self.global_iter, sampler=self)
				elif use_rebuilding_draw:
					# a subset, perhaps different then last iteration
					# need to reconstruct the region from scratch
					real_joint_indices = numpy.where(self.real_data_mask_all)[0][joint_indices]
					draw_constrained = self.draw_constrained(real_joint_indices, self.real_data_mask_all, joint_live_pointsp, self.global_iter)
				else:
					# full data set, can keep longer
					draw_constrained = self.superset_draw_constrained
				
				uj, xj, Lj, n = draw_constrained(
					Lmins=Lmins_higher, 
					priortransform=self.priortransform, 
					loglikelihood=lambda params: self.multi_loglikelihood(params, joint_data_mask), 
					ndim=self.ndim,
					draw_global_uniform=self.draw_global_uniform,
					live_pointsu = joint_live_pointsu,
					max_draws=max_draws,
					iter=self.global_iter,
					nlive_points=self.nlive_points
				)
				
				# we have a new draw
				self.ndraws += int(n)
				ppi = len(self.pointpile)
				if self.membership_graph is not None:
					self.membership_graph.add_vertex("p%d" % ppi, id=ppi, vtype=1)
				self.pointpile = numpy.vstack((self.pointpile, [uj]))
				self.pointpilex = numpy.vstack((self.pointpilex, [xj]))
				nfilled = 0
				for j, d in enumerate(numpy.where(joint_data_mask)[0]):
					if Lj[j] > Lmins_higher[j]:
						self.shelves[d].append((ppi, uj, xj, Lj[j]))
						nfilled += 1
				if nfilled == self.ndata:
					# new point is a superpoint, accepted by all
					self.superpoints.append(ppi)
				print('accept after %d tries, filled %d shelves' % (n, nfilled))
			
			# we got a new point
			#print 'new point:', Lmins[data_mask], (Lj>Lmins[data_mask])*1
		
		# pop: for every data entry, advance one point
		print('advancing all...')
		self.global_iter += 1
		pj_old = self.live_pointsp[Lmini,numpy.arange(self.ndata)]
		uis = self.pointpile[pj_old]
		xis = self.pointpilex[pj_old]
		Lis = live_pointsL[Lmini, numpy.arange(self.ndata)]
		if self.membership_graph is not None:
			print('    deleting edges...')
			self.membership_graph.delete_edges([("n%d" % d, "p%d" % pj) for d, pj in enumerate(pj_old)])
		if self.point_data_map is not None:
			for d, pj in enumerate(pj_old):
				self.point_data_map[pj].remove(d)
		# point assignment changed, so can not re-use any more directly
		self.last_graph = None
		self.last_graph_selection = None
		if self.superpoints:
			print('    dropping superpoints ...')
			for pj in numpy.unique(pj_old):
				# no longer a superpoint, because it is no
				# longer shared by all data sets
				if pj in self.superpoints:
					self.superpoints.remove(pj)
		new_edges = None if self.membership_graph is None else []
		print('    replacing dead points ...')
		for d in range(self.ndata):
			i = Lmini[d]
			pj, uj, xj, Lj = self.shelves[d].pop(0)
			self.live_pointsp[i,d] = pj
			live_pointsL[i,d] = Lj
			if new_edges is not None:
				new_edges.append(("n%d" % d, "p%d" % pj))
			if self.point_data_map is not None:
				self.point_data_map[pj].add(d)
		if self.membership_graph is not None:
			print('    adding edges ...')
			self.membership_graph.add_edges(new_edges)
		self.Lmax = live_pointsL.max(axis=0)
		assert self.Lmax.shape == (self.ndata,)
		print('advancing done.')
		return numpy.asarray(uis), numpy.asarray(xis), numpy.asarray(Lis)
	
	def remainder(self, d=None):
		if d is None:
			print('sorting remainder...')
			indices = numpy.empty((self.ndata, self.nlive_points), dtype=int)
			for d in range(self.ndata):
				indices[d,:] = numpy.argsort(self.live_pointsL[:,d])
			ds = numpy.arange(self.ndata)
			print('building remainder...')
			for i in range(self.nlive_points):
				j = indices[ds,i]
				p = self.live_pointsp[j,ds]
				u = self.pointpile[p]
				x = self.pointpilex[p]
				L = self.live_pointsL[j,ds]
				#u = [self.pointpile[self.live_pointsp[indices[d][i],d]] for d in range(self.ndata)]
				#x = [self.pointpilex[self.live_pointsp[indices[d][i],d]] for d in range(self.ndata)]
				#L = numpy.asarray([self.live_pointsL[indices[d][i],d] for d in range(self.ndata)])
				yield u, x, L
			print('remainder done.')
		else:
			indices = numpy.argsort(self.live_pointsL[:,d])
			for i in indices:
				u = self.pointpile[self.live_pointsp[i,d]]
				x = self.pointpilex[self.live_pointsp[i,d]]
				L = self.live_pointsL[i,d]
				yield u, x, L
				#yield self.live_pointsu[i,d], self.live_pointsx[i,d], self.live_pointsL[i,d]
	
	next = __next__
	
	def __iter__(self):
		while True: yield self.__next__()
		
__all__ = [MultiNestedSampler]

