"""
Copyright: Johannes Buchner (C) 2013

Modular, Pythonic Implementation of Nested Sampling
"""

import numpy
from numpy import exp, log, log10, pi
import progressbar
from adaptive_progress import AdaptiveETA
from numpy import logaddexp
from nested_sampling.nested_integrator import conservative_estimator, integrate_remainder

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
def multi_nested_integrator(multi_sampler, tolerance = 0.01, max_samples=None):
	logVolremaining = 0
	logwidth = log(1 - exp(-1. / multi_sampler.nlive_points))
	weights = [] #[-1e300, 1]]
	
	widgets = [progressbar.Counter('%f'),
		progressbar.Bar(), progressbar.Percentage(), AdaptiveETA()]
	pbar = progressbar.ProgressBar(widgets = widgets)
	
	i = 0
	ui, xi, Li = multi_sampler.next()
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
		i_final = -sampler.nlive_points * (-sampler.Lmax + log(exp(max(tolerance - logZerr, logZerr / 100.) + logZ) - exp(logZ)))
		pbar.maxval = min(max(i+1, i_final), i+100000)
		#logmaxContribution = logZup - logZ
		remainderZ, remainderZerr = integrate_remainder(sampler, logwidth, logVolremaining, logZ)
		
		if len(weights) > sampler.nlive_points:
			# tolerance
			total_error = logZerr + remainderZerr
			#total_error = logZerr + logmaxContribution
			if max_samples is not None and int(max_samples) < int(sampler.ndraws):
				pbar.finish()
				print 'maximum number of samples reached'
				break
			if total_error < tolerance:
				pbar.finish()
				print 'tolerance reached:', total_error, logZerr, remainderZerr
				break
			# we want to make maxContribution as small as possible
			#   but if it becomes 10% of logZerr, that is enough
			if remainderZerr < logZerr / 10.:
				pbar.finish()
				print 'tolerance will not improve: remainder error (%.3f) is much smaller than systematic errors (%.3f)' % (logZerr, remainderZerr)
				break
		
		widgets[0] = '|%d/%d samples+%d/%d|lnZ = %.2f +- %.3f + %.3f|L=%.2e @ %s' % (
			i + 1, pbar.maxval, sampler.nlive_points, sampler.ndraws, logaddexp(logZ, remainderZ), logZerr, remainderZerr, Li,
			numpy.array_str(xi, max_line_width=1000, precision=4))
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

__all__ = [nested_integrator]

