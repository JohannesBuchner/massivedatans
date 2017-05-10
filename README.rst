The idea is as follows:

We are analysing ndata data sets simultaneously. Ideally, the data sets 
are somewhat similar, so that model evaluations are relevant across data sets.

Two walls form a corner.
The first wall are the live points (array with nlive x ndata)
The second wall is fragged (ndata queues).
The corner of the wall is the lowest likelihood live points.

If none of the queues are not empty, we can push the second wall forward.
This pushes out the corner and replaces the point in the first wall.

The queues have to be filled with live points.



on new draws add to queue position n iff
Li greater than the n-th smallest L of live points and current queue entries


Strategies for deciding whether to include the live points of a dataset for
drawing a new point (and thereby, deciding whether its L should be evaluated):

A) Use all for 10 iterations
   Use datasets with empty shelves with 0 entries and repeat until done
   
   Results: OK, there are some speedups from 4->10 (15%) 10->100 (2x). 4 is much slower than 1.

B) Put maximum on draws of ndraws_max = 1000.
   Use all for one try.
   Then use all shelves with < 5 entries.
   Then use all shelves with < 3 entries.
   Then use all shelves with 0 entries, with ndraws_max = inf and repeat until done

C) Use all for one try.
   Then use all empty shelves, with ndraws_max = inf and repeat until done
   
   Much slower than A. Same speedups.

D) Keep count of the successful accidental adds. Make bidirectional connections between the datasets
   These expire if not renewed in the last 10 iterations where both where included.
   Use empty shelves
   Then use all shelves with 0 entries, and those pulled in by connections
   Then use all shelves with 0 entries, and those pulled in by connections with < 3 entries
   Then use all shelves with 0 entries, with ndraws_max = inf and repeat until done

ad D:
   When a draw is made from a set, all that learned something are connected 
   pairwise in a co-learning graph (list of edges).
   Phase 1: superset_draws, with everyone (nsuperset_draws=10 times)
   Phase 2: Add shelves with zero entries to a set, then add the ones that co-learned recently (last 100 iterations where both were involved).
   Phase 3: Only use those with zero entries.
   
   On successful draws, create success-mask. In each iteration, store a list of these masks.
   
   When looking for co-learners of data set i (mask empty-mask), go through previous 100 iterations.
   Create a copy of empty-mask, colearners-mask.
   For each success-mask in the previous 100 iterations:
       AND the success-mask and the empty-mask.
       If any are true, colearners-mask = colearners-mask OR success-mask
   The final colearners-mask is to be used.
   
   
   
   
   Also need to implement variable termination.
   sample() should take a mask, for which new samples are needed.
   integrator should store terminating cases, and ask sampler only for the remaining.
   return dict might need to be re-assembled, with weights=0 as fillers for extra iterations.






