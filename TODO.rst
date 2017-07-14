============
TODO
============

Non-performance
-----------------

* Change prints to logging

Performance (single-threaded)
------------------------------

Here we discuss wall-clock time, not number of model evaluations.
If the model is slow enough, there is no issue.

Currently, the execution speed is limited by two functions:

1. Building the RadFriends region draw_constrained -> maxdistance

maxdistance could be optimized by calling it less often. This is
what sample.CachedConstrainer tries to do. The checks there could be more 
generous. -> Done, but maybe more optimisation possible?

One could also increase the rebuild_every parameters

One could modify MetricLearningFriendsConstrainer to rebuild not every n calls, 
but every n likelihood evaluations. This would improve performance when drawing
is already quite efficient. See nestle, which does this.
-> Done!

2. Building the graph to find independent data sets, multi_nested_sampler.generate_subsets_graph

igraph could be replaced with graphtool, which supports parallelisation.

One could further explore when to use 
generate_subsets_graph vs generate_subsets_nograph (controlled by use_graph)


Performance (parallelisation)
------------------------------

* The subsets could be sampled in parallel.

* The entire framework could be set up in a MapReduce/MPI way, with the 
  MetricLearningFriendsConstrainer proposing (multiple) points,
  passing to multiple machines for evaluating the model,
  then using MapReduce to evaluate the likelihood over the Big Data set,
  and returning this to MetricLearningFriendsConstrainer.
  See MultiNest, which already parallelises the likelihood evaluations.





