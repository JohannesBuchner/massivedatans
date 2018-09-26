from __future__ import print_function, division
import numpy
from hiermetriclearn import MetricLearningFriendsConstrainer
from elldrawer import MultiEllipsoidalConstrainer

# use this for MLFriends (RadFriends, but with standardized Euclidean metric)
def generate_fresh_constrainer_mlfriends():
	return MetricLearningFriendsConstrainer(
		metriclearner = 'truncatedscaling', force_shrink=True,
		rebuild_every=1000, metric_rebuild_every=20, 
		verbose=False)

# use this for Ellipsoidal Sampling, like MultiNest
def generate_fresh_constrainer_multiellipsoidal():
	return MultiEllipsoidalConstrainer(rebuild_every=1000, enlarge=3.)

generate_fresh_constrainer = generate_fresh_constrainer_multiellipsoidal

class CachedConstrainer(object):
	"""
	This keeps metric learners if they are used (in the last three iterations).
	Otherwise, constructs a fresh one.
	"""
	def __init__(self, sampler=None):
		self.iter = -1
		self.prev_prev_prev_generation = {}
		self.prev_prev_generation = {}
		self.prev_generation = {}
		self.curr_generation = {}
		self.last_mask = []
		self.last_points = []
		self.last_realmask = None
		self.sampler = sampler
	
	def get(self, mask, realmask, points, it):
		while self.iter < it:
			# new generation
			self.prev_prev_prev_generation = self.prev_prev_generation
			self.prev_prev_generation = self.prev_generation
			self.prev_generation = self.curr_generation
			self.curr_generation = {}
			self.last_mask = []
			self.last_realmask = None
			self.last_points = []
			self.iter += 1
		
		# if we only dropped a single (or a few) data sets
		# compared to the call just before, lets reuse the same
		# this happens in the focussed draw with 1000s of data sets
		# where a single data set can accept a point; 
		# not worth to recompute the region.
		if self.last_realmask is not None and len(mask) < len(self.last_mask) and \
			len(mask) > 0.80 * len(self.last_mask) and \
			len(points) <= len(self.last_points) and \
			len(points) > 0.90 * len(self.last_points) and \
			numpy.mean(self.last_realmask == realmask) > 0.80 and \
			numpy.in1d(points, self.last_points).all():
			print('re-using previous, similar region (%.1f%% data set overlap, %.1f%% points overlap)' % (numpy.mean(self.last_realmask == realmask) * 100., len(points) * 100. / len(self.last_points), ))
			k = tuple(self.last_mask.tolist())
			return self.curr_generation[k].draw_constrained
		print('not re-using region', (len(mask), len(self.last_mask), len(points), len(self.last_points), (len(mask) < len(self.last_mask), len(mask) > 0.80 * len(self.last_mask), len(points) > 0.90 * len(self.last_points), numpy.mean(self.last_realmask == realmask) ) ))
		
		# normal operation:
		k = tuple(mask.tolist())
		self.last_realmask = realmask
		self.last_mask = mask
		self.last_points = points
		
		# try to recycle
		if k in self.curr_generation:
			pass
		elif k in self.prev_generation:
			print('re-using previous1 region')
			self.curr_generation[k] = self.prev_generation[k]
		elif k in self.prev_prev_generation:
			print('re-using previous2 region')
			self.curr_generation[k] = self.prev_prev_generation[k]
		elif k in self.prev_prev_prev_generation:
			print('re-using previous3 region')
			self.curr_generation[k] = self.prev_prev_prev_generation[k]
		else:
			# nothing found, so start from scratch
			self.curr_generation[k] = generate_fresh_constrainer()
			#self.curr_generation[k] = MetricLearningFriendsConstrainer(
			#	metriclearner = 'truncatedscaling', force_shrink=True,
			#	rebuild_every=1000, metric_rebuild_every=20, 
			#	verbose=False)
			self.curr_generation[k].sampler = self.sampler
		
		return self.curr_generation[k].draw_constrained

def generate_individual_constrainer(rebuild_every=1000, metric_rebuild_every=20):
	individual_constrainers = {}
	individual_constrainers_lastiter = {}
	def individual_draw_constrained(i, it, sampler):
		if i not in individual_constrainers:
			#individual_constrainers[i] = MetricLearningFriendsConstrainer(
			#	metriclearner = 'truncatedscaling', force_shrink=True,
			#	rebuild_every=rebuild_every, metric_rebuild_every=metric_rebuild_every, 
			#	verbose=False)
			individual_constrainers[i] = generate_fresh_constrainer()
			individual_constrainers[i].sampler = sampler
			individual_constrainers_lastiter[i] = it
		if it > individual_constrainers_lastiter[i] + 5:
			# force rebuild
			individual_constrainers[i].region = None
		individual_constrainers_lastiter[i] = it
		return individual_constrainers[i].draw_constrained
	return individual_constrainers, individual_constrainers_lastiter, individual_draw_constrained

def generate_superset_constrainer():
	return generate_fresh_constrainer()
	#return MetricLearningFriendsConstrainer(metriclearner = 'truncatedscaling', 
	#	rebuild_every=1000, metric_rebuild_every=20, verbose=False, force_shrink=True)


