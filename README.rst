=========================================================================
Big Data vs. complex physical models - a scalable inference algorithm
=========================================================================

A algorithm for fitting models against many data sets, giving parameter probability distributions.
The key is that model evaluations are efficiently re-used between data sets,
making the algorithm scale sub-linearly.

See paper for details: https://arxiv.org/abs/1707.04476

How to run
============

You need to install

* python-igraph
* numpy, scipy
* h5py
* progressbar
* gcc

Then run::

	$ # build
	$ make
	$ # simulate data set
	$ python gensimple_horns.py 10000
	$ # analyse
	$ OMP_NUM_THREADS=4 python sample.py data_widths_10000.hdf5 100
	$ # simulate no-signal data set
	$ python gennothing.py 10000 # simulate no-signal data set
	$ # analyse
	$ OMP_NUM_THREADS=4 python sample.py data_nothing_10000.hdf5 10000

See paper draft for details.

Improving Performance
=======================

See TODO.

Implementation notes and Code organisation
============================================

* sample.py sets up everything
* Set your problem definition (parameters, model, likelihood) in sample.py
* Integrator: multi_nested_integrator.py . Calls sampler repeatedly.
* Joint Sampler: multi_nested_sampler.py . This deals with managing the graph and the queues and which live points to use for a new draw. Calls draw_constrained
* The queues (paper) are called shelves in the code.
* RadFriends: hiermetriclearn.py: Suggests new samples from live points and filters with likelihood function to return a higher point.
* clustering/: Fast C implementations for checking if a point is in the neighbourhood and computing safe distances.





