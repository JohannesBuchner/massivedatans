"""
Copyright: Johannes Buchner (C) 2013

Modular, Pythonic Implementation of Nested Sampling
"""

import numpy
from numpy import exp, log, log10, pi
import progressbar
from adaptive_progress import AdaptiveETA
from numpy import logaddexp

def integrate_remainder(sampler, logwidth, logVolremaining, logZ):
	# logwidth remains the same now for each sample
	remainder = list(sampler.remainder())
	logV = logwidth
	L0 = remainder[-1][2]
	Ls = numpy.exp([Li - L0 for ui, xi, Li in remainder])
	Lmax = Ls[1:].sum(axis=0) + Ls[-1]
	Lmin = Ls[:-1].sum(axis=0) + Ls[0]
	logLmid = log(Ls.sum(axis=0)) + L0
	logZmid = logaddexp(logZ, logV + logLmid)
	logZup  = logaddexp(logZ, logV + log(Lmax) + L0)
	logZlo  = logaddexp(logZ, logV + log(Lmin) + L0)
	#print 'upper:', logZup, 'lower:', logZlo, 'middle:', logZmid
	logZerr = numpy.max([logZup - logZmid, logZmid - logZlo], axis=0)
	return logV + logLmid, logZerr

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
def multi_nested_integrator(multi_sampler, tolerance = 0.01, max_samples=None, min_samples = 0):
	sampler = multi_sampler
	logVolremaining = 0
	logwidth = log(1 - exp(-1. / sampler.nlive_points))
	weights = [] #[-1e300, 1]]
	
	widgets = [progressbar.Counter('%f'),
		progressbar.Bar(), progressbar.Percentage(), AdaptiveETA()]
	pbar = progressbar.ProgressBar(widgets = widgets)
	
	i = 0
	ui, xi, Li = sampler.next()
	wi = logwidth + Li
	logZ = wi
	H = Li - logZ
	pbar.currval = i
	pbar.maxval = sampler.nlive_points
	pbar.start()
	while True:
		i = i + 1
		logwidth = log(1 - exp(-1. / sampler.nlive_points)) + logVolremaining
		logVolremaining -= 1. / sampler.nlive_points
		
		weights.append([ui, xi, Li, logwidth])
		
		logZerr = (H / sampler.nlive_points)**0.5
		
		#maxContribution = sampler.Lmax + logVolremaining
		#minContribution = Li + logVolremaining
		#midContribution = logaddexp(maxContribution, minContribution)
		#logZup  = logaddexp(maxContribution, logZ)
		#logZmid = logaddexp(midContribution, logZ)
		pbar.update(i)
		
		# expected number of iterations:
		i_final = -sampler.nlive_points * (-sampler.Lmax + log(exp(numpy.max([tolerance - logZerr, logZerr / 100.], axis=0) + logZ) - exp(logZ)))
		i_final = numpy.where(i_final < i+1, i+1, numpy.where(i_final > i+100000, i+100000, i_final))
		pbar.maxval = i_final.max()
		#logmaxContribution = logZup - logZ
		
		if len(weights) > max(min_samples, sampler.nlive_points): # and all(remainderZ - log(100) < logZ):
			remainderZ, remainderZerr = integrate_remainder(sampler, logwidth, logVolremaining, logZ)
			# tolerance
			total_error = logZerr + remainderZerr
			#total_error = logZerr + logmaxContribution
			if max_samples is not None and int(max_samples) < int(sampler.ndraws):
				pbar.finish()
				print 'maximum number of samples reached'
				break
			if all(total_error < tolerance):
				pbar.finish()
				print 'tolerance reached:', total_error, logZerr, remainderZerr
				break
			# we want to make maxContribution as small as possible
			#   but if it becomes 10% of logZerr, that is enough
			if numpy.logical_or(total_error < tolerance, remainderZerr < logZerr / 10.).all():
				pbar.finish()
				print 'tolerance will not improve further: remainder error (%.3f) is much smaller than systematic errors (%.3f)' % (max(logZerr), max(remainderZerr))
				break
			widgets[0] = '|%d/%d samples+%d/%d|lnZ = %.2f(%.2f) +- %.3f + %.3f|L=%.2e ' % (
				i + 1, pbar.maxval, sampler.nlive_points, sampler.ndraws, logaddexp(logZ[0], remainderZ[0]), remainderZ[0], max(logZerr), max(remainderZerr), Li[0])
		else:
			widgets[0] = '|%d/%d samples+%d/%d|lnZ = %.2f +- %.3f|L=%.2e ' % (
				i + 1, pbar.maxval, sampler.nlive_points, sampler.ndraws, logZ[0], max(logZerr), Li[0])
		print widgets[0]
		ui, xi, Li = sampler.next()
		wi = logwidth + Li
		logZnew = logaddexp(logZ, wi)
		H = exp(wi - logZnew) * Li + exp(logZ - logZnew) * (H + logZ) - logZnew
		logZ = logZnew
	
	# not needed for integral, but for posterior samples, otherwise there
	# is a hole in the most likely parameter ranges.
	remainderZ, remainderZerr = integrate_remainder(sampler, logwidth, logVolremaining, logZ)
	weights += [[ui, xi, Li, logwidth] for ui, xi, Li in sampler.remainder()]
	logZerr += remainderZerr
	logZ = logaddexp(logZ, remainderZ)
	
	return dict(logZ=logZ, logZerr=logZerr, 
		samples=sampler.samples, weights=weights, information=H,
		niterations=i)

__all__ = [multi_nested_integrator]

