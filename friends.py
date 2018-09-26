from __future__ import print_function, division
import numpy
import scipy.spatial, scipy.cluster
import matplotlib.pyplot as plt
from nested_sampling.clustering import clusterdetect
from nested_sampling.clustering.neighbors import find_maxdistance, find_rdistance, initial_rdistance_guess, nearest_rdistance_guess

class FriendsConstrainer(object):
	"""
	Rejection sampling pre-filtering method based on neighborhood to live points.

	"Distant" means in this implementation that the distance to a cluster member
	is large. 
	The maximum distance to a cluster is computed by considering each
	cluster member and its k nearest neighbors in turn, and 
	computing the maximum distance.
	
	:param rebuild_every: After how many iterations should the clustering
		distance be re-computed?
	    
	:param radial:	
		if radial = True, then the normal euclidean distance is used.
		otherwise, the absolute coordinate difference in each dimension is used.
	
	:param metric:
		metric to use. Use 'chebyshev' for SupFriends, in which case then 
		the supremum norm is used. Use 'euclidean' for RadFriends, via 
		the euclidean norm.
	
	:param jackknife:
		if True, instead of leaving out a group of live points in
		the distance estimate, only one is left out in turn (jackknife resampling
		instead of bootstrap resampling).
	
	:param force_shrink:
		if True, the distance can only decrease between sampling steps.
	
	"""
	def __init__(self, rebuild_every = 50, radial = True, metric = 'euclidean', jackknife = False,
			force_shrink = False,
			hinter = None, verbose = False, 
			keep_phantom_points=False, optimize_phantom_points=False):
		self.maxima = []
		self.iter = 0
		self.region = None
		self.rebuild_every = rebuild_every
		self.radial = radial
		self.metric = metric
		self.file = None
		self.jackknife = jackknife
		self.force_shrink = force_shrink
		self.hinter = hinter
		self.verbose = verbose
		if keep_phantom_points:
			assert self.force_shrink, 'keep_phantom_points needs force_shrink=True'
		self.keep_phantom_points = keep_phantom_points
		self.optimize_phantom_points = optimize_phantom_points
		self.phantom_points = []
		self.phantom_points_Ls = []
		self.last_cluster_points = None
	
	def cluster(self, u, ndim, keepRadius=False):
		"""
		
		"""
		if self.verbose: print('building region ...')
		if len(u) > 10:
			if keepRadius and self.region is not None and 'maxdistance' in self.region:
				maxdistance = self.region['maxdistance']
			else:
				if self.radial:
					if self.jackknife:
						#maxdistance = initial_rdistance_guess(u, k=1, metric=self.metric)
						maxdistance = nearest_rdistance_guess(u, metric=self.metric)
					else:
						maxdistance = find_rdistance(u, nbootstraps=20, metric=self.metric, verbose=self.verbose)
				else:
					maxdistance = find_maxdistance(u)
			if self.force_shrink and self.region is not None and 'maxdistance' in self.region:
				maxdistance = min(maxdistance, self.region['maxdistance'])
			if self.keep_phantom_points and len(self.phantom_points) > 0:
				# add phantoms to u now
				print('including phantom points in cluster members', self.phantom_points)
				u = numpy.vstack((u, self.phantom_points))
			ulow  = numpy.max([u.min(axis=0) - maxdistance, numpy.zeros(ndim)], axis=0)
			uhigh = numpy.min([u.max(axis=0) + maxdistance, numpy.ones(ndim)], axis=0)
		else:
			maxdistance = None
			ulow = numpy.zeros(ndim)
			uhigh = numpy.ones(ndim)
		if self.verbose: print('setting sampling region:', (ulow, uhigh), maxdistance)
		self.region = dict(members=u, maxdistance=maxdistance, ulow=ulow, uhigh=uhigh)
		self.generator = None
		
	def is_inside(self, u):
		"""
		Check if this new point is near or inside one of our clusters
		"""
		ndim = len(u)
		ulow = self.region['ulow']
		uhigh = self.region['uhigh']
		if not ((ulow <= u).all() and (uhigh >= u).all()):
			# does not even lie in our primitive rectangle
			# do not even need to compute the distances
			return False
		
		members = self.region['members']
		maxdistance = self.region['maxdistance']
		
		# if not initialized: no prefiltering
		if maxdistance is None:
			return True
		
		# compute distance to each member in each dimension
		if self.radial:
			dists = scipy.spatial.distance.cdist(members, [u], metric=self.metric)
			assert dists.shape == (len(members), 1)
			dist_criterion = dists < maxdistance
		else:
			dists = numpy.abs(u - members)
			assert dists.shape == (len(members), ndim), (dists.shape, ndim, len(members))
			# nearer than maxdistance in all dimensions
			dist_criterion = numpy.all(dists < maxdistance, axis=1)
			assert dist_criterion.shape == (len(members),), (dist_criterion.shape, len(members))
		# is it true for at least one?
		closeby = dist_criterion.any()
		if closeby:
			return True
		return False
	
	def are_inside_rect(self, u):
		"""
		Check if the new points are near or inside one of our clusters
		"""
		ulow = self.region['ulow']
		uhigh = self.region['uhigh']
		mask = numpy.logical_and(((ulow <= u).all(axis=1), (uhigh >= u).all(axis=1)))
	def are_inside_cluster(self, u, ndim):
		members = self.region['members']
		maxdistance = self.region['maxdistance']
		
		# if not initialized: no prefiltering
		if maxdistance is None:
			return numpy.ones(len(u), dtype=bool)
		
		# compute distance to each member in each dimension
		if self.radial:
			dists = scipy.spatial.distance.cdist(members, u, metric=self.metric)
			assert dists.shape == (len(members), len(u))
			dist_criterion = dists < maxdistance
		else:
			raise NotImplementedError()
		# is it true for at least one?
		closeby = dist_criterion.any(axis=0)
		return closeby
	
	def generate(self, ndim):
		it = True
		verbose = False and self.verbose
		ntotal = 0
		# largest maxdistance where generating from full space makes sense
		full_maxdistance = 0.5 * (0.01)**(1./ndim)
		while True:
			maxdistance = self.region['maxdistance']
			if maxdistance is None:
				# do a prefiltering rejection sampling first
				u = numpy.random.uniform(self.region['ulow'], self.region['uhigh'], size=ndim)
				yield u, ntotal
				ntotal = 0
				continue
			members = self.region['members']
			it = numpy.random.uniform() < 0.01
			# depending on the region size compared to 
			# the total space, one of the two methods will
			# be more efficient
			if it or not self.radial or maxdistance > full_maxdistance:
				it = True
				# for large regions
				# do a prefiltering rejection sampling first
				us = numpy.random.uniform(self.region['ulow'], self.region['uhigh'], size=(100, ndim))
				ntotal += 100
				mask = self.are_inside_cluster(self.transform_points(us), ndim)
				if not mask.any():
					continue
				us = us[mask]
				#indices = numpy.arange(len(mask))[mask]
				#for i in indices:
				#	u = us[indices[i],:]
				for u in us:
					yield u, ntotal
					ntotal = 0
			else:
				# for small regions
				# draw from points
				us = members[numpy.random.randint(0, len(members), 100),:]
				ntotal += 100
				if verbose: print('chosen point', us)
				if self.metric == 'euclidean':
					# draw direction around it
					direction = numpy.random.normal(0, 1, size=(100, ndim))
					direction = direction / ((direction**2).sum(axis=1)**0.5).reshape((-1,1))
					if verbose: print('chosen direction', direction)
					# choose radius: volume gets larger towards the outside
					# so give the correct weight with dimensionality
					radius = maxdistance * numpy.random.uniform(0, 1, size=(100,1))**(1./ndim)
					us = us + direction * radius
				else:
					assert self.metric == 'chebyshev'
					us = us + numpy.random.uniform(-maxdistance, maxdistance, size=(100, ndim))
				if verbose: print('using point', u)
				inside = numpy.logical_and((us >= 0).all(axis=1), (us <= 1).all(axis=1))
				if not inside.any():
					if verbose: print('outside boundaries', us, direction, maxdistance)
					continue
				us = us[inside]
				# count the number of points this is close to
				dists = scipy.spatial.distance.cdist(members, us, metric=self.metric)
				assert dists.shape == (len(members), len(us))
				nnear = (dists < maxdistance).sum(axis=0)
				if verbose: print('near', nnear)
				#ntotal += 1
				# accept with probability 1./nnear
				coin = numpy.random.uniform(size=len(us))
				
				accept = coin < 1. / nnear
				if not accept.any():
					if verbose: print('probabilistic rejection due to overlaps')
					continue
				us = us[accept]
				for u in us:
					yield u, ntotal
					ntotal = 0
			
	def transform_new_points(self, us):
		return us
	def transform_points(self, us):
		return us
	def transform_point(self, u):
		return u
	
	def rebuild(self, u, ndim, keepRadius=False):
		if self.last_cluster_points is None or \
			len(self.last_cluster_points) != len(u) or \
			numpy.any(self.last_cluster_points != u):
			self.cluster(u=self.transform_new_points(u), ndim=ndim, keepRadius=keepRadius)
			self.last_cluster_points = u
		
			# reset generator
			self.generator = self.generate(ndim=ndim)
	def debug(self, ndim):
		if self.file is None:
			#self.file = open("friends_debug.txt", "a")
			import tempfile
			filename = tempfile.mktemp(dir='',
				prefix='friends%s-%s_' % (
				'1' if self.jackknife else '',
				self.metric))
			self.file = open(filename, 'w')
			self.file.write("{} {} {}\n".format(self.iter, self.region['maxdistance'], len(self.region['members'])))
		self.file.write("{} {} {} {}\n".format(self.iter, self.region['maxdistance'], len(self.region['members']), ndim))
	def debugplot(self, u = None):
		print('creating plot...')
		n = len(self.region['members'][0]) / 2
		plt.figure(figsize=(6, n/2*4+1))
		m = self.region['members']
		d = self.region['maxdistance']
		for i in range(n):
			plt.subplot(numpy.ceil(n / 2.), 2, 1+i)
			j = i * 2
			k = i * 2 + 1
			plt.plot(m[:,j], m[:,k], 'x', color='b', ms=1)
			plt.gca().add_artist(plt.Circle((m[0,j], m[0,k]), d, color='g', alpha=0.3))
			if u is not None:
				plt.plot(u[j], u[k], 's', color='r')
				plt.gca().add_artist(plt.Circle((u[j], u[k]), d, color='r', alpha=0.3))
		prefix='friends%s-%s_' % ('1' if self.jackknife else '', self.metric)
		plt.savefig(prefix + 'cluster.pdf')
		plt.close()
		print('creating plot... done')
	
	def draw_constrained(self, Lmins, priortransform, loglikelihood, live_pointsu, ndim, max_draws=None, **kwargs):
		# previous is [[u, x, L], ...]
		self.iter += 1
		rebuild = self.iter % self.rebuild_every == 1
		if rebuild or self.region is None:
			self.rebuild(numpy.asarray(live_pointsu), ndim, keepRadius=False)
		if self.generator is None:
			self.generator = self.generate(ndim=ndim)
		ntoaccept = 0
		ntotalsum = 0
		while True:
			for u, ntotal in self.generator:
				assert (u >= 0).all() and (u <= 1).all(), u
				ntotalsum += ntotal
				
				if self.hinter is not None:
					hints = self.hinter(u)
					if len(hints) == 0:
						# no way
						continue
					if len(hints) > 1:
						# choose a random solution, by size
						raise NotImplementedError("multiple solutions not implemented")
						hints = hints[numpy.random.randInt(len(hints))]
					else:
						hints = hints[0]
				
					for i, lo, hi in hints:
						u[i] = numpy.random.uniform(lo, hi)
					if not is_inside(self.transform_point(u)):
						# not sure if this is a good idea
						# it means we dont completely trust
						# the hinting function
						continue
				
				x = priortransform(u)
				L = loglikelihood(x)
				ntoaccept += 1
				
				if numpy.any(L > Lmins) or (max_draws is not None and ntotalsum > max_draws):
					# yay, we win
					if ntotalsum > 10000: 
						if self.verbose: 
							print('sampled %d points, evaluated %d ' % (ntotalsum, ntoaccept))
							#self.debugplot(u)
					return u, x, L, ntoaccept
				
				# if running very inefficient, optimize clustering 
				#     if we haven't done so at the start
				if not rebuild and ntoaccept > 1000:
					#self.debugplot(u)
					break
			rebuild = True
			self.rebuild(numpy.asarray(live_pointsu), ndim, keepRadius=False)

if __name__ == '__main__':
	friends = FriendsConstrainer(radial = True)
	
	u = numpy.random.uniform(0.45, 0.55, size=1000).reshape((-1, 2))
	ndim = 2
	friends.cluster(u, ndim=ndim)
	Lmin = -1
	rv = scipy.stats.norm(0.515, 0.03)
	def priortransform(x): return x
	def loglikelihood(x): return rv.logpdf(x).sum()
	previous = []
	colors = ['r', 'g', 'orange']
	plt.figure("dists", figsize=(7,4))
	plt.figure("plane", figsize=(5,5))
	plt.plot(u[:,0], u[:,1], 'x')
	Lmins = [-5, 2, 2.5] #, 2.58]
	for j, (Lmin, color) in enumerate(zip(numpy.array(Lmins)*ndim, colors)):
		values = []
		for i in range(200):
			friends.iter = 4 # avoid rebuild
			u, x, L, ntoaccept = friends.draw_constrained(Lmin, priortransform, loglikelihood, previous, ndim)
			plt.figure("plane")
			plt.plot(u[0], u[1], '+', color=color)
			values.append(u)
		values = numpy.array(values)
		plt.figure("dists")
		for k in range(ndim):
			plt.subplot(1, ndim, k + 1)
			plt.title('Lmin={}, dim={}'.format(Lmin, k))
			plt.hist(values[:,k], cumulative=True, normed=True, 
				color=color, bins=1000, histtype='step')
	plt.figure("plane")
	plt.savefig('friends_sampling_test.pdf', bbox_inches='tight')
	plt.close()
	plt.figure("dists")
	plt.savefig('friends_sampling_test_dists.pdf', bbox_inches='tight')
	plt.close()
	
	# another test: given a group of samples, assert that only neighbors are evaluated
	
	r = numpy.random.uniform(0.2, 0.25, size=400)
	phi = numpy.random.uniform(0, 1, size=400)**10 * 2*numpy.pi
	u = numpy.transpose([0.5 + r*numpy.cos(phi), 0.5 + r*numpy.sin(phi)])
	friends.cluster(u, ndim=2)
	plt.figure(figsize=(10,5))
	plt.subplot(1, 2, 1)
	plt.plot(u[:,0], u[:,1], 'x')
	suggested = []
	def loglikelihood(x):
		r = ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)**0.5
		#assert r < 0.5
		#assert r > 0.1
		suggested.append(r)
		if r > 0.2 and r < 0.25:
			plt.plot(x[0], x[1], 'o', color='green')
			return 100
		plt.plot(x[0], x[1], 'o', color='red')
		return -100
	
	ndim = 2
	taken = []
	for i in range(100):
		friends.iter = 4 # avoid rebuild
		u, x, L, ntoaccept = friends.draw_constrained(Lmin, priortransform, loglikelihood, previous, ndim)
		r = ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)**0.5
		taken.append(r)
		print('suggested:', u)
	plt.subplot(1, 2, 2)
	plt.hist(taken, cumulative=True, normed=True, 
			color='g', bins=1000, histtype='step')
	plt.hist(suggested, cumulative=True, normed=True, 
			color='r', bins=1000, histtype='step')
	#x = numpy.linspace(0, 1, 400)
	#y = x**ndim - (x - min(suggested) / max(suggested))**ndim
	#y /= max(y)
	#plt.plot(x * (max(suggested) - min(suggested)) + min(suggested), y, '--', color='grey')
	
	plt.savefig('friends_sampling_test_sampling.pdf', bbox_inches='tight')
	plt.close()
	
	
	

