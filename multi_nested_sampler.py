"""
Copyright: Johannes Buchner (C) 2013

Modular, Pythonic Implementation of Nested Sampling
"""

import numpy
from numpy import exp, log, log10, pi
import progressbar
import networkx

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
		self.real_multi_loglikelihood = multi_loglikelihood
		self.multi_loglikelihood = multi_loglikelihood
		self.superset_draw_constrained = superset_draw_constrained
		self.draw_constrained = draw_constrained
		self.samples = []
		self.ndim = ndim
		self.ndata = ndata
		self.membership_graph = networkx.Graph()
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
			self.membership_graph.add_edges_from([
				((0, d), (1, len(pointpile))) for d in range(ndata)])
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
		self.data_mask_all = numpy.ones(self.ndata) == 1
		self.real_data_mask_all = numpy.ones(self.ndata) == 1
		assert self.Lmax.shape == (ndata,)
		self.ndraws = nlive_points
		self.shelves = [[] for _ in range(ndata)]
		
		self.dump_iter = 1
	
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
		print ('shelf status: %s' % ''.join([status_symbols.get(len(shelf), 'X') for shelf in self.shelves])).encode('utf-8')
	
	def cut_down(self, surviving):
		# delete some data sets
		
		self.live_pointsp = self.live_pointsp[:,surviving]
		self.live_pointsu = self.live_pointsu[:,surviving]
		self.live_pointsx = self.live_pointsx[:,surviving]
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
		
		#print 'cutting graph (not grass)'
		#print 'surviving:', surviving
		#print 'surviving IDs:', numpy.where(surviving)[0]
		#print 'dying IDs:', numpy.where(~surviving)[0]
		delete_nodes = [(0,d) for d, survives in enumerate(surviving) if not survives]
		#print 'deleting:', delete_nodes
		unconnected = [n for n, degree in 
			self.membership_graph.degree_iter() if degree==0]
		#print 'dropping:', unconnected
		self.membership_graph.remove_nodes_from(unconnected)
		# renaming in forward order, so we do not collide
		rename_nodes = [((0, oldd), (0, newd)) for newd, oldd in enumerate(numpy.where(surviving)[0]) if newd != oldd]
		#print 'renaming:', rename_nodes
		self.membership_graph.remove_nodes_from(delete_nodes)
		networkx.relabel_nodes(self.membership_graph, dict(rename_nodes), copy=False)
		#for k, v in rename_nodes:
		#	networkx.relabel_nodes(self.membership_graph, {k:v}, copy=False)
		#print self.membership_graph.nodes()
		
		# or, rebuild graph:
		"""
		graph = networkx.Graph()
		# pointing from live_point to member
		for i in numpy.where(self.data_mask_all)[0]:
			graph.add_edges_from((((0, i), (1, p)) for p in self.live_pointsp[:,i]))
		assert len(self.membership_graph.nodes()) == len(graph.nodes()), (len(self.membership_graph.nodes()), len(graph))
		assert len(self.membership_graph.edges()) == len(graph.edges()), (len(self.membership_graph.edges()), len(graph))
		#if not networkx.is_isomorphic(self.membership_graph, graph):
		for node in self.membership_graph.nodes():
			if node not in graph:
				print 'cut graph is missing node', node
		for node in graph.nodes():
			if node not in self.membership_graph:
				print 'cut graph has extra node', node
		print 'delta1', networkx.difference(self.membership_graph, graph).edges()
		print 'delta2', networkx.difference(graph, self.membership_graph).edges()
		print 'collecting edges 1'
		edges_a = set(graph.edges())
		print 'collecting edges 2'
		edges_b = set(self.membership_graph.edges())
		print 'comparing edges'
		for (a, b) in edges_a:
			if (a, b) not in edges_b and (b, a) not in edges_b:
				print 'cut graph is missing edge', (a, b)
		for (a, b) in edges_b:
			if (a, b) not in edges_a and (b, a) not in edges_a:
				print 'cut graph has extra edge', (a, b)
		print 'checking if isomorphic:'
		assert networkx.is_isomorphic(self.membership_graph, graph)
		self.membership_graph = graph
		"""
		
	def generate_subsets(self, data_mask):
		#if self.dump_iter % 50 == 0:
		#	numpy.savez('dump_%d.npz' % self.dump_iter, 
		#		live_pointsp=self.live_pointsp, data_mask=data_mask)
		#self.dump_iter += 1
		# generate data subsets which share points.
		firstmember = numpy.where(data_mask)[0][0]
		if len(self.live_pointsp[:,firstmember]) == len(numpy.unique(self.live_pointsp[:,data_mask].flatten())):
			# trivial case: all live points are the same across data sets
			yield data_mask, self.live_pointsp[:,firstmember]
			return
		
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
				sharing = (self.live_pointsp[:,to_handle] == p).any(axis=0)
				#assert len(sharing) == to_handle.sum()
				newmembers = numpy.where(to_handle)[0][sharing]
				assert numpy.all(newmembers == numpy.arange(len(to_handle))[to_handle][sharing])

				#print 'new members:', newmembers
				members += newmembers.tolist()
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

	def generate_subsets(self, data_mask):
		# generate data subsets which share points.
		live_pointsp = self.live_pointsp
		
		firstmember = numpy.where(data_mask)[0][0]
		allp = numpy.unique(live_pointsp[:,data_mask].flatten())
		if len(live_pointsp[:,firstmember]) == len(allp):
			# trivial case: all live points are the same across data sets
			yield data_mask, live_pointsp[:,firstmember]
			return
		
		subgraphs = list(networkx.connected_component_subgraphs(
			self.membership_graph, copy=False))
		if len(subgraphs) == 1:
			yield data_mask, allp
			return
	
		# then identify disjoint subgraphs
		for subgraph in subgraphs:
			member_data_mask = numpy.zeros(len(data_mask), dtype=bool)
			member_live_pointsp = []
			for nodetype, i in subgraph:
				if nodetype == 0:
					member_data_mask[i] = True
				else:
					member_live_pointsp.append(i)
			if not member_data_mask.any():
				continue
			yield member_data_mask, member_live_pointsp
	
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
				# check shelves
				global_live_pointsu = self.get_unique_pointsp(self.live_pointsp[:,data_mask])
			else:
				nsuperset_draws = -self.nsuperset_draws
				sample_subset = iter > nsuperset_draws
				# iter: 1 2 3 4 5 6
				# nsuperset_draws = 1
				# iter - nsuperset_draws = 0 1 2 3 4 5 6
				cut_level = max(0, 5 - (iter - nsuperset_draws))
				# cut_level = 5-0 5-1 5-2 5-3 5-4 5-5 0 0 0 
				data_mask = numpy.array([len(self.shelves[d]) <= cut_level for d in range(self.ndata)])
			
				# check shelves
				data_mask = self.data_mask_all
				global_live_pointsu = all_global_live_pointsu
				Lmin = all_Lmin
			
			self.shelf_status()
			membersets = list(self.generate_subsets(data_mask))
			for ji, (joint_data_mask, joint_live_pointsp) in enumerate(membersets):
				print 'live point set %d/%d: %d from %d datasets, %s' % (ji, len(membersets), len(joint_live_pointsp), joint_data_mask.sum(), 'focussed set constrained draw' if sample_subset else 'super-set constrained draw')
				joint_live_pointsu = self.pointpile[joint_live_pointsp]
				#print 'members:', joint_data_mask.shape, joint_live_pointsu.shape
				max_draws = 1000
				# if it is the only dataset and we need an entry here, try longer
				if joint_data_mask.sum() == 1 and len(self.shelves[numpy.where(joint_data_mask)[0][0]]) == 0:
					max_draws = 100000
				uj, xj, Lj, n = (self.draw_constrained if sample_subset else self.superset_draw_constrained)(
					Lmins=Lmins[joint_data_mask], 
					priortransform=self.priortransform, 
					loglikelihood=lambda params: self.multi_loglikelihood(params, joint_data_mask), 
					ndim=self.ndim,
					draw_global_uniform=self.draw_global_uniform,
					live_pointsu = joint_live_pointsu,
					max_draws=max_draws,
				)
				
				# we have a new draw
				self.ndraws += int(n)
				ppi = len(self.pointpile)
				self.pointpile = numpy.vstack((self.pointpile, [uj]))
				for j, d in enumerate(numpy.where(joint_data_mask)[0]):
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
			pj_old = self.live_pointsp[i,d]
			self.membership_graph.remove_edge((0,d), (1, pj_old))
			pj, uj, xj, Lj = self.shelves[d].pop(0)
			ujs.append(uj)
			xjs.append(xj)
			Ljs.append(Lj)
			assert Lj > Lmin, (Lj, Lmin)
			self.live_pointsp[i,d] = pj
			self.membership_graph.add_edge((0,d), (1, pj))
			live_pointsu[i,d] = uj
			live_pointsx[i,d] = xj
			live_pointsL[i,d] = Lj
		self.Lmax = live_pointsL.max(axis=0)
		#print 'Lmax:', self.Lmax
		assert self.Lmax.shape == (self.ndata,)
		self.samples.append([ujs, xjs, Ljs])
		return numpy.asarray(uis), numpy.asarray(xis), numpy.asarray(Lis)
	
	def remainder(self, d=None):
		if d is None:
			indices = [] 
			for d in range(self.ndata):
				i = numpy.argsort(self.live_pointsL[:,d]) #[L[d] for L in self.live_pointsL])
				#indices[d] = i
				indices.append(i)
			for i in range(self.nlive_points):
				u = [self.live_pointsu[indices[d][i],d] for d in range(self.ndata)]
				x = [self.live_pointsx[indices[d][i],d] for d in range(self.ndata)]
				L = [self.live_pointsL[indices[d][i],d] for d in range(self.ndata)]
				yield u, x, numpy.asarray(L)
		else:
			indices = numpy.argsort(self.live_pointsL[:,d])
			for i in indices:
				yield self.live_pointsu[i,d], self.live_pointsx[i,d], self.live_pointsL[i,d]
	def next(self):
		return self.__next__()
	def __iter__(self):
		while True: yield self.__next__()
		
__all__ = [MultiNestedSampler]

