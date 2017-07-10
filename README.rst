=========================================================================
Big Data vs. complex physical models - a scalable inference algorithm
=========================================================================

Work in progress.

How to run::

	$ python gensimple_horns.py 10000 # simulate data set
	$ OMP_NUM_THREADS=4 python sample.py data_widths_10000.hdf5 100
	$ python gennothing.py 10000 # simulate no-signal data set
	$ OMP_NUM_THREADS=4 python sample.py data_nothing_10000.hdf5 10000

See paper draft for details.

