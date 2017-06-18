"""
Copyright: Johannes Buchner (C) 2013

Modular, Pythonic Implementation of Nested Sampling
"""

import numpy
from numpy import exp, log, log10, pi
import progressbar
from adaptive_progress import AdaptiveETA
from numpy import logaddexp

def integrate_remainder(sampler, logwidth, logVolremaining, logZ, H, globalLmax):
	# logwidth remains the same now for each sample
	remainder = list(sampler.remainder())
	logV = logwidth
	L0 = remainder[-1][2]
	logLs = [Li - L0 for ui, xi, Li in remainder]
	Ls = numpy.exp([Li - L0 for ui, xi, Li in remainder])
	LsMax = Ls.copy()
	LsMax[-1] = numpy.exp(globalLmax - L0)
	Lmax = LsMax[1:].sum(axis=0) + LsMax[-1]
	#Lmax = Ls[1:].sum(axis=0) + Ls[-1]
	Lmin = Ls[:-1].sum(axis=0) + Ls[0]
	# take extreme values for all Lmax, Lmin:
	#Lmax = numpy.exp(globalLmax - L0) * len(Ls)
	#Lmin = numpy.min(Ls) * len(Ls)
	logLmid = log(Ls.sum(axis=0)) + L0
	logZmid = logaddexp(logZ, logV + logLmid)
	logZup  = logaddexp(logZ, logV + log(Lmax) + L0)
	logZlo  = logaddexp(logZ, logV + log(Lmin) + L0)
	#print 'upper:', logZup, 'lower:', logZlo, 'middle:', logZmid
	logZerr = logZup - logZlo
	#logZerr = numpy.max([logZup - logZmid, logZmid - logZlo], axis=0)
	#return logV + logLmid, logZerr

	# try bootstrapping for error estimation
	"""
	bs_logZmids = []
	for _ in range(20):
		i = numpy.random.randint(0, len(Ls), len(Ls))
		i.sort()
		i = numpy.arange(len(Ls))
		bs_Ls = LsMax[i,:]
		Lmax = bs_Ls[1:].sum(axis=0) + bs_Ls[-1]
		bs_Ls = Ls[i,:]
		Lmin = bs_Ls[:-1].sum(axis=0) + bs_Ls[0]
		bs_logZmids.append(logaddexp(logZ, logV + log(Lmax.sum(axis=0)) + L0))
		bs_logZmids.append(logaddexp(logZ, logV + log(Lmin.sum(axis=0)) + L0))
	bs_logZerr = numpy.max(bs_logZmids, axis=0) - numpy.min(bs_logZmids, axis=0)
	#print numpy.shape(bs_logZmids), bs_logZerr.shape, len(Ls), Ls.shape
	#print 'logZ errors: %.2f %.2f %.2f' % (bs_logZerr[0], logZerr[0], (H[0] / sampler.nlive_points)**0.5)
	"""
	for i in range(len(remainder)):
		ui, xi, Li = remainder[i]
		wi = logwidth + Li
		logZnew = logaddexp(logZ, wi)
		H = exp(wi - logZnew) * Li + exp(logZ - logZnew) * (H + logZ) - logZnew
		logZ = logZnew
	
	return logV + logLmid, logZerr, logZmid, logZerr + (H / sampler.nlive_points)**0.5, logZerr + (H / sampler.nlive_points)**0.5 #, bs_logZerr + (H / sampler.nlive_points)**0.5


"""
Performs the Nested Sampling integration by calling the *sampler* multiple times
until the *tolerance* is reached, or the maximum number of likelihood evaluations
is exceeded.

:param sampler: Sampler
:param tolerance: uncertainty in log Z to compute to
:param max_samples: maximum number of likelihood evaluations (None for no limit)

@return dictionary containing the keys

  logZ, logZerr: log evidence and uncertainty, 
  samples: all obtained samples,
  weights: posterior samples: 
  	list of prior coordinates, transformed coordinates, likelihood value 
  	and weight
  information: information H
  niterations: number of nested sampling iterations
"""
def multi_nested_integrator(multi_sampler, tolerance = 0.01, max_samples=None, min_samples = 0, need_robust_remainder_error=True):
	sampler = multi_sampler
	logVolremaining = 0
	logwidth = log(1 - exp(-1. / sampler.nlive_points))
	weights = [] #[-1e300, 1]]
	
	widgets = [progressbar.Counter('%f'),
		progressbar.Bar(), progressbar.Percentage(), AdaptiveETA()]
	pbar = progressbar.ProgressBar(widgets = widgets)
	
	i = 0
	ndata = multi_sampler.ndata
	running = numpy.ones(ndata, dtype=bool)
	last_logwidth = numpy.zeros(ndata)
	last_logVolremaining = numpy.zeros(ndata)
	last_remainderZ = numpy.zeros(ndata)
	last_remainderZerr = numpy.zeros(ndata)
	logZerr = numpy.zeros(ndata)
	ui, xi, Li = sampler.next()
	wi = logwidth + Li
	logZ = wi
	H = Li - logZ
	remainder_tails = [[]] * ndata
	pbar.currval = i
	pbar.maxval = sampler.nlive_points
	pbar.start()
	while True:
		i = i + 1
		logwidth = log(1 - exp(-1. / sampler.nlive_points)) + logVolremaining
		last_logwidth[running] = logwidth
		last_logVolremaining[running] = logwidth
		logVolremaining -= 1. / sampler.nlive_points
		
		# fill up, otherwise set weight to zero
		Lifull = numpy.zeros(ndata)
		Lifull[:] = -numpy.inf
		Lifull[running] = Li
		uifull = numpy.zeros((ndata, ui.shape[1]))
		uifull[running,:] = ui
		xifull = numpy.zeros((ndata, ui.shape[1]))
		xifull[running,:] = xi
		weights.append([uifull, xifull, Lifull, numpy.where(running, logwidth, -numpy.inf), running])
		
		logZerr[running] = (H[running] / sampler.nlive_points)**0.5
		
		#maxContribution = sampler.Lmax + logVolremaining
		#minContribution = Li + logVolremaining
		#midContribution = logaddexp(maxContribution, minContribution)
		#logZup  = logaddexp(maxContribution, logZ)
		#logZmid = logaddexp(midContribution, logZ)
		pbar.update(i)
		
		# expected number of iterations:
		i_final = -sampler.nlive_points * (-sampler.Lmax + log(exp(numpy.max([tolerance - logZerr[running], logZerr[running] / 100.], axis=0) + logZ[running]) - exp(logZ[running])))
		i_final = numpy.where(i_final < i+1, i+1, numpy.where(i_final > i+100000, i+100000, i_final))
		pbar.maxval = i_final.max()
		#logmaxContribution = logZup - logZ
		
		if i > min_samples and i % 10 == 1: # and all(remainderZ - log(100) < logZ):
		#if i > min_samples:
			remainderZ, remainderZerr, totalZ, totalZerr, totalZerr_bootstrapped = integrate_remainder(sampler, logwidth, logVolremaining, logZ[running], H[running], sampler.Lmax)
			# tolerance
			#remainderZerr[remainderZerr == 0] = 100
			last_remainderZ[running] = remainderZ
			last_remainderZerr[running] = remainderZerr
			#total_error = logZerr[running] + remainderZerr
			#terminating = numpy.logical_and(totalZerr < tolerance, remainderZerr < 0.01)
			terminating = totalZerr < tolerance
			#terminating = numpy.random.uniform(size=running.sum()) < 0.1
			widgets[0] = '|%d/%d samples+%d/%d|lnZ = %.2f +- %.3f + %.3f|L=%.2f^%.2f ' % (
				i + 1, pbar.maxval, sampler.nlive_points, sampler.ndraws, logaddexp(logZ[running][0], remainderZ[0]), max(logZerr[running]), max(remainderZerr), Li[0], sampler.Lmax[0])
			if terminating.any():
				print 'terminating some:', terminating
				for j, k in enumerate(numpy.where(running)[0]):
					if terminating[j]:
						remainder_tails[k] = [[ui, xi, Li, logwidth] for ui, xi, Li in sampler.remainder(j)]
				sampler.cut_down(~terminating)
				running[running] = ~terminating
			if not running.any():
				break
			#print logZ[running][0], remainderZ[0], logZerr[running], remainderZerr, Li, logaddexp(logZ[running][0], remainderZ[0])
			print widgets[0]
		#else:
		#	widgets[0] = '|%d/%d samples+%d/%d|lnZ = %.2f +- %.3f|L=%.2f^%.2f ' % (
		#		i + 1, pbar.maxval, sampler.nlive_points, sampler.ndraws, logZ[running][0], max(logZerr[running]), Li[0], sampler.Lmax)
		ui, xi, Li = sampler.next()
		wi = logwidth + Li
		logZnew = logaddexp(logZ[running], wi)
		H[running] = exp(wi - logZnew) * Li + exp(logZ[running] - logZnew) * (H[running] + logZ[running]) - logZnew
		logZ[running] = logZnew
	
	# not needed for integral, but for posterior samples, otherwise there
	# is a hole in the most likely parameter ranges.
	#remainderZ, remainderZerr = integrate_remainder(sampler, last_logwidth, last_logVolremaining, logZ)
	
	#weights += [[ui, xi, Li, last_logwidth, running] for ui, xi, Li in sampler.remainder()]
	all_tails = numpy.ones(ndata, dtype=bool)
	for i in range(sampler.nlive_points):
		u, x, L, logwidth = zip(*[tail[i] for tail in remainder_tails])
		weights.append([u, x, L, logwidth, all_tails])
	logZerr = logZerr + last_remainderZerr
	logZ = logaddexp(logZ, last_remainderZ)
	
	return dict(logZ=logZ, logZerr=logZerr, 
		weights=weights, information=H,
		niterations=i)

__all__ = [multi_nested_integrator]

