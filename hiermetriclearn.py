import numpy
import scipy.spatial, scipy.cluster
import matplotlib.pyplot as plt
from clustering.neighbors import find_rdistance, is_within_distance_of, count_within_distance_of, any_within_distance_of
from clustering.jarvispatrick import jarvis_patrick_clustering, jarvis_patrick_clustering_iterative
from clustering.sdml import IdentityMetric, SimpleScaling, TruncatedScaling
from collections import defaultdict
from clustering.radfriendsregion import ClusterResult, RadFriendsRegion

class MetricLearningFriendsConstrainer(object):
	"""
	0) Store unit metric.
	1) Splits live points into clusters using Jarvis-Patrick K=1 clustering
	2) Project new clusters onto old clusters for identification tree.
	   If new cluster encompasses more than one old cluster: 
	3) Overlay all clusters (shift by cluster mean) and compute new metric (covariance)
	4) Using original points and new metric, compute RadFriends bootstrapped distance and store
	5) In each RadFriends cluster, find points.
        6) If still mono-mode: no problem
	   If discovered new clusters in (1): store filtering function and cluster assignment
	   If no new clusters: no problem
	
	When point is replaced:
	1) Check if point is in a cluster that is dying out: 
	   when point is last in current or previously stored clustering
	
	For sampling:
	1) Draw a new point from a metric-shaped ball from random point
	2) Filter with previous filtering functions if exist
	3) Evaluate likelihood
	
	For filtering:
	1) Given a point, check if within metric-shaped ball of a existing point
	2) Filter with previous filtering functions if exist
	
	"""
	def __init__(self, metriclearner, rebuild_every = 50, metric_rebuild_every = 50, verbose = False,
			keep_phantom_points=False, optimize_phantom_points=False,
			force_shrink=False):
		self.iter = 0
		self.region = None
		self.rebuild_every = int(rebuild_every)
		self.metric_rebuild_every = int(metric_rebuild_every)
		self.previous_filters = []
		self.verbose = verbose
		self.keep_phantom_points = keep_phantom_points
		self.optimize_phantom_points = optimize_phantom_points
		self.force_shrink = force_shrink
		self.phantom_points = []
		self.phantom_points_Ls = []
		self.metriclearner = metriclearner
		self.metric = IdentityMetric()
		self.clusters = None
		self.direct_draws_efficient = True
		self.last_cluster_points = None
		#self.metricregionhistory = []
	
	def cluster(self, u, ndim, keepMetric=False):
		"""
		1) Splits live points into clusters using Jarvis-Patrick K=1 clustering
		2) Project new clusters onto old clusters for identification tree.
		   If new cluster encompasses more than one old cluster: 
		3) Overlay all clusters (shift by cluster mean) and compute new metric (covariance)
		4) Using original points and new metric, compute RadFriends bootstrapped distance and store
		5) In each RadFriends cluster, find points.
		6) If still mono-mode: no problem
		   If discovered new clusters in (1): store filtering function and cluster assignment
		   If no new clusters: no problem
		"""
		w = self.metric.transform(u)
		prev_region = self.region
		if keepMetric:
			self.region = RadFriendsRegion(members=w)
			if self.force_shrink and self.region.maxdistance > prev_region.maxdistance:
				self.region = RadFriendsRegion(members=w, maxdistance=prev_region.maxdistance)
			return
		
		metric_updated = False
		clustermetric = self.metric
		print 'computing distances for clustering...'
		wdists = scipy.spatial.distance.cdist(w, w, metric='euclidean')
		clusters = [numpy.arange(len(w))]
		# Overlay all clusters (shift by cluster mean) 
		print 'Metric update ...'
		shifted_cluster_members = []
		for members in clusters:
			cluster_mean = numpy.mean(u[members,:], axis=0)
			shifted_cluster_members += (u[members,:] - cluster_mean).tolist()
		
		# Using original points and new metric, compute RadFriends bootstrapped distance and store
		if self.metriclearner == 'none':
			metric = self.metric # stay with identity matrix
			metric_updated = False
		elif self.metriclearner == 'simplescaling':
			metric = SimpleScaling()
			metric.fit(shifted_cluster_members)
			metric_updated = True
		elif self.metriclearner == 'truncatedscaling':
			metric = TruncatedScaling()
			metric.fit(shifted_cluster_members)
			metric_updated = self.metric == IdentityMetric() or not numpy.all(self.metric.scale == metric.scale)
		else:
			assert False, self.metriclearner
		
		self.metric = metric
		
		oldclusters = self.clusters
		self.clusters = clusters
		
		wnew = self.metric.transform(u)
		#shifted_cluster_members = []
		#for members in clusters:
		#	cluster_mean = numpy.mean(wnew[members,:], axis=0)
		#	shifted_cluster_members += (wnew[members,:] - cluster_mean).tolist()
		#shifted_cluster_members = numpy.asarray(shifted_cluster_members)
		#shifted_region = RadFriendsRegion(members=shifted_cluster_members)
		print 'Region update ...'
		
		self.region = RadFriendsRegion(members=wnew) #, maxdistance=shifted_region.maxdistance)
		if not metric_updated and self.force_shrink and prev_region is not None:
			if self.region.maxdistance > prev_region.maxdistance:
				self.region = RadFriendsRegion(members=w, maxdistance=prev_region.maxdistance)
		
		if oldclusters is None or len(clusters) != len(oldclusters):
		#if True:
			# store filter function
			self.previous_filters.append((self.metric, self.region, ClusterResult(metric=clustermetric, clusters=self.clusters, points=w)))
		
		#rfclusters = self.region.get_clusters()
		#print 'Clustering: JP has %d clusters, radfriends has %d cluster:' % (len(clusters), len(rfclusters))
		#var = self.iter, self.metric, u, self.region.maxdistance
		#assert self.is_inside(numpy.array([0.123456]*ndim)), var
		#assert self.is_inside(numpy.array([0.654321]*ndim)), var
		print 'done.'
	
	def are_inside_cluster(self, points):
		w = self.metric.transform(points)
		return self.region.are_inside(w)
	
	def is_inside(self, point):
		if not ((point >= 0).all() and (point <= 1).all()):
			return False
		w = self.metric.transform(point)
		return self.region.is_inside(w)

	def generate(self, ndim):
		ntotal = 0
		"""
		for w, n in self.region.generate():
			u = self.metric.untransform(w)
			ntotal += n
			#if numpy.all(u >= 0) and numpy.all(u <= 1):
			if all([0 <= ui <= 1 for ui in u]):
				yield u, ntotal
				ntotal = 0
			else:
				print 'rejected [box constraint]'
			
		"""
		N = 10000
		while True:
			#if numpy.random.uniform() < 0.01:
			if True:
				# draw from unit cube
				# this can be efficient if volume still large
				ntotal = ntotal + N
				us = numpy.random.uniform(size=(N, ndim))
				ws = self.metric.transform(us)
				nnear = self.region.are_inside(ws)
				#print '  %d of %d accepted' % (nnear.sum(), N)
				for u in us[nnear,:]:
					#print 'unit cube draw success:', ntotal
					yield u, ntotal
					ntotal = 0
			if ndim < 40:
				# draw from radfriends directly
				for ws, n in self.region.generate(N):
					us = self.metric.untransform(ws)
					assert us.shape[1] == ndim, us.shape
					ntotal = ntotal + n
					mask = numpy.logical_and(us < 1, us > 0).all(axis=1)
					assert mask.shape == (len(us),), (mask.shape, us.shape)
					if mask.any():
						#print 'radfriends draw in unit cube:', mask.sum(), ntotal
						for u in us[mask,:]:
							assert u.shape == (us[0].shape), (u.shape, us.shape, mask.shape)
							yield u, ntotal
							ntotal = 0
					#if all([0 <= ui <= 1 for ui in u]):
					#	yield u, ntotal
					#	ntotal = 0
	
	def rebuild(self, u, ndim, keepMetric=False):
		if self.last_cluster_points is not None and \
			len(self.last_cluster_points) == len(u) and \
			numpy.all(self.last_cluster_points == u):
			# nothing
			return
		
		#for prev_u, prev_region, prev_metric, prev_clusters in self.metricregionhistory[::-1]:
		#	if prev_u.shape == u.shape and numpy.all(prev_u == u):
		#		self.region = prev_region
		#		self.metric = prev_metric
		#		self.clusters = prev_clusters
		#		self.generator = self.generate(ndim)
		#		return
		
		self.cluster(u=u, ndim=ndim, keepMetric=keepMetric)
		self.last_cluster_points = u
		#if keepMetric is False:
		#	# store into history only fresh ones with new metric
		#	self.metricregionhistory.append((u, self.region, self.metric, self.clusters))
		#	self.metricregionhistory = self.metricregionhistory[-10:]
		
		print 'maxdistance:', self.region.maxdistance
		self.generator = self.generate(ndim)
	
	def is_last_of_its_cluster(self, u, uothers):
		# check if only point of current clustering left
		w = self.metric.transform(u)
		wothers = self.metric.transform(uothers)
		othersregion = RadFriendsRegion(members=wothers, maxdistance=self.region.maxdistance)
		if not othersregion.is_inside(w):
			return True
		
		# check previous clusterings
		for metric, region, clusters in self.previous_filters:
			if clusters.get_n_clusters() < 2:
				# only one cluster, so can not die out
				continue
			
			# check in which cluster this point was
			i = clusters.get_cluster_id(u)
			j = clusters.get_cluster_ids(uothers)
			#print 'cluster_sets:', i, set(j)
			if i not in j:
				# this is the last point of that cluster
				return True
		return False
	
	def _draw_constrained_prepare(self, Lmins, priortransform, loglikelihood, live_pointsu, ndim, **kwargs):
		rebuild = self.iter % self.rebuild_every == 0 or self.region is None
		keepMetric = not (self.iter % self.metric_rebuild_every == 0)
		if rebuild:
			self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=keepMetric)
		self.iter += 1
		assert self.generator is not None
		ntoaccept = 0
		ntotalsum = 0
		if self.keep_phantom_points:
			# check if the currently dying point is the last of a cluster
			starti = kwargs['starti']
			ucurrent = live_pointsu[starti]
			#wcurrent = self.metric.transform(ucurrent)
			uothers = [ui for i, ui in enumerate(live_pointsu) if i != starti]
			#wothers = self.metric.transform(uothers)
			phantom_points_added = False
			if self.is_last_of_its_cluster(ucurrent, uothers):
				if self.optimize_phantom_points:
					print 'optimizing phantom point', ucurrent
					import scipy.optimize
					def f(u):
						w = self.metric.transform(u)
						if not self.region.is_inside(w):
							return 1e100
						x = priortransform(u)
						L = loglikelihood(x)
						if self.verbose: print 'OPT %.2f ' % L, u
						return -L
					r = scipy.optimize.fmin(f, ucurrent, ftol=0.5, full_output=True)
					ubest = r[0]
					Lbest = -r[1]
					ntoaccept += r[3]
					print 'optimization gave', r
					wbest = self.metric.transform(ubest)
					if not self.is_last_of_its_cluster(ubest, uothers):
						print 'that optimum is inside the other points, so no need to store'
					else:
						print 'remembering phantom point', ubest, Lbest
						self.phantom_points.append(ubest)
						self.phantom_points_Ls.append(Lbest)
						phantom_points_added = True
				else:
					print 'remembering phantom point', ucurrent
					self.phantom_points.append(ucurrent)
					phantom_points_added = True
			
			if phantom_points_added:
				self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=keepMetric)
				rebuild = True
			
			
			if self.optimize_phantom_points and len(self.phantom_points) > 0:
				# purge phantom points that are below Lmin
				keep = [i for i, Lp in enumerate(self.phantom_points_Ls) if Lp > Lmin]
				self.phantom_points = [self.phantom_points[i] for i in keep]
				if len(keep) != len(self.phantom_points_Ls):
					print 'purging some old phantom points. new:', self.phantom_points, Lmin
					self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=keepMetric)
					rebuild = True
					
				self.phantom_points_Ls = [self.phantom_points_Ls[i] for i in keep]
		return ntoaccept, ntotalsum, rebuild
	
	def get_Lmax(self):
		if len(self.phantom_points_Ls) == 0:
			return None
		return max(self.phantom_points_Ls)

	def draw_constrained(self, Lmins, priortransform, loglikelihood, live_pointsu, ndim, **kwargs):
		ntoaccept, ntotalsum, rebuild = self._draw_constrained_prepare(Lmins, priortransform, loglikelihood, live_pointsu, ndim, **kwargs)
		rebuild_metric = rebuild
		while True:
			for u, ntotal in self.generator:
				assert (u >= 0).all() and (u <= 1).all(), u
				ntotalsum += ntotal
				x = priortransform(u)
				L = loglikelihood(x)
				ntoaccept += 1

				#print 'ntotal:', ntotal
				if ntotal > 100000:
					self.direct_draws_efficient = False
				
				if numpy.any(L > Lmins):
					# yay, we win
					#print 'accept after %d tries' % ntoaccept
					return u, x, L, ntoaccept
				
				# if running very inefficient, optimize clustering 
				#     if we haven't done so at the start
				if not rebuild and ntoaccept > 100:
					rebuild = True
					print 'low efficiency is triggering RadFriends rebuild'
					self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=True)
					break
				if not rebuild_metric and ntoaccept > 1000:
					rebuild_metric = True
					print 'low efficiency is triggering metric rebuild'
					self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=False)
					break
