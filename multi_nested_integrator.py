from __future__ import print_function, division
"""

Integrator
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
from adaptive_progress import AdaptiveETA
from numpy import logaddexp
import sys

def integrate_remainder(sampler, logwidth, logVolremaining, logZ, H, globalLmax):
	# logwidth remains the same now for each sample
	remainder = list(sampler.remainder())
	logV = logwidth
	L0 = remainder[-1][2]
	L0 = globalLmax
	logLs = [Li - L0 for ui, xi, Li in remainder]
	Ls = numpy.exp(logLs)
	LsMax = Ls.copy()
	LsMax[-1] = numpy.exp(globalLmax - L0)
	Lmax = LsMax[1:].sum(axis=0) + LsMax[-1]
	#Lmax = Ls[1:].sum(axis=0) + Ls[-1]
	Lmin = Ls[:-1].sum(axis=0) + Ls[0]
	logLmid = log(Ls.sum(axis=0)) + L0
	logZmid = logaddexp(logZ, logV + logLmid)
	logZup  = logaddexp(logZ, logV + log(Lmax) + L0)
	logZlo  = logaddexp(logZ, logV + log(Lmin) + L0)
	logZerr = logZup - logZlo
	assert numpy.isfinite(H).all()
	assert numpy.isfinite(logZerr).all(), logZerr

	for i in range(len(remainder)):
		ui, xi, Li = remainder[i]
		wi = logwidth + Li
		logZnew = logaddexp(logZ, wi)
		#Hprev = H
		H = exp(wi - logZnew) * Li + exp(logZ - logZnew) * (H + logZ) - logZnew
		H[H < 0] = 0
		#assert (H>0).all(), (H, Hprev, wi, Li, logZ, logZnew)
		logZ = logZnew
	
	#assert numpy.isfinite(logZerr + (H / sampler.nlive_points)**0.5), (H, sampler.nlive_points, logZerr)
	
	return logV + logLmid, logZerr, logZmid, logZerr + (H / sampler.nlive_points)**0.5, logZerr + (H / sampler.nlive_points)**0.5

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
	
	widgets = ["|...|",
		progressbar.Bar(), progressbar.Percentage(), AdaptiveETA()]
	pbar = progressbar.ProgressBar(widgets = widgets, maxval=sampler.nlive_points)
	
	i = 0
	ndata = multi_sampler.ndata
	running = numpy.ones(ndata, dtype=bool)
	last_logwidth = numpy.zeros(ndata)
	last_logVolremaining = numpy.zeros(ndata)
	last_remainderZ = numpy.zeros(ndata)
	last_remainderZerr = numpy.zeros(ndata)
	logZerr = numpy.zeros(ndata)
	ui, xi, Li = next(sampler)
	wi = logwidth + Li
	logZ = wi
	H = Li - logZ
	remainder_tails = [[]] * ndata
	pbar.currval = i
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
		
		sys.stdout.flush()
		pbar.update(i)
		
		# expected number of iterations:
		i_final = -sampler.nlive_points * (-sampler.Lmax + log(exp(numpy.max([tolerance - logZerr[running], logZerr[running] / 100.], axis=0) + logZ[running]) - exp(logZ[running])))
		i_final = numpy.where(i_final < i+1, i+1, numpy.where(i_final > i+100000, i+100000, i_final))
		max_value = max(i+1, i_final.max())
		if hasattr(pbar, 'max_value'):
			pbar.max_value = max_value
		elif hasattr(pbar, 'maxval'):
			pbar.maxval = max_value
		
		if i > min_samples and i % 50 == 1 or (max_samples and i > max_samples):
			remainderZ, remainderZerr, totalZ, totalZerr, totalZerr_bootstrapped = integrate_remainder(sampler, logwidth, logVolremaining, logZ[running], H[running], sampler.Lmax)
			print('checking for termination:', remainderZ, remainderZerr, totalZ, totalZerr)
			# tolerance
			last_remainderZ[running] = remainderZ
			last_remainderZerr[running] = remainderZerr
			terminating = totalZerr < tolerance
			if max_samples and i > max_samples:
				terminating[:] = True
			widgets[0] = '|%d/%d samples+%d/%d|lnZ = %.2f +- %.3f + %.3f|L=%.2f^%.2f ' % (
				i + 1, max_value, sampler.nlive_points, sampler.ndraws, logaddexp(logZ[running][0], remainderZ[0]), max(logZerr[running]), max(remainderZerr), Li[0], sampler.Lmax[0])
			if terminating.any():
				print('terminating %d, namely:' % terminating.sum(), list(numpy.where(terminating)[0]))
				for j, k in enumerate(numpy.where(running)[0]):
					if terminating[j]:
						remainder_tails[k] = [[ui, xi, Li, logwidth] for ui, xi, Li in sampler.remainder(j)]
				sampler.cut_down(~terminating)
				running[running] = ~terminating
			if not running.any():
				break
			print(widgets[0])
		ui, xi, Li = next(sampler)
		wi = logwidth + Li
		logZnew = logaddexp(logZ[running], wi)
		H[running] = exp(wi - logZnew) * Li + exp(logZ[running] - logZnew) * (H[running] + logZ[running]) - logZnew
		logZ[running] = logZnew
	
	# add tail
	# not needed for integral, but for posterior samples, otherwise there
	# is a hole in the most likely parameter ranges.
	all_tails = numpy.ones(ndata, dtype=bool)
	for i in range(sampler.nlive_points):
		u, x, L, logwidth = list(zip(*[tail[i] for tail in remainder_tails]))
		weights.append([u, x, L, logwidth, all_tails])
	logZerr = logZerr + last_remainderZerr
	logZ = logaddexp(logZ, last_remainderZ)
	
	return dict(logZ=logZ, logZerr=logZerr, 
		weights=weights, information=H,
		niterations=i)

__all__ = [multi_nested_integrator]

