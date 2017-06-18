import numpy
import sys
import time
import networkx
import igraph

def generate_subsets_reference(data_mask, live_pointsp, graph, _):
	# generate data subsets which share points.
	firstmember = numpy.where(data_mask)[0][0]
	if len(live_pointsp[:,firstmember]) == len(numpy.unique(live_pointsp[:,data_mask].flatten())):
		# trivial case: all live points are the same across data sets
		yield data_mask, live_pointsp[:,firstmember]
		return
	
	to_handle = data_mask.copy()
	while to_handle.any():
		firstmember = numpy.where(to_handle)[0][0]
		to_handle[firstmember] = False
		members = [firstmember]
		# get live points of this member
		member_live_pointsp = live_pointsp[:,firstmember].tolist()
		# look through to_handle for entries and check if they have the points
		i = 0
		while True:
			if i >= len(member_live_pointsp) or not to_handle.any():
				break
			p = member_live_pointsp[i]
			sharing = (live_pointsp[:,to_handle] == p).any(axis=0)
			#assert len(sharing) == to_handle.sum()
			newmembers = numpy.where(to_handle)[0][sharing]
			#assert numpy.all(newmembers == numpy.arange(len(to_handle))[to_handle][sharing])

			#print 'new members:', newmembers
			members += newmembers.tolist()
			for newp in numpy.unique(live_pointsp[:,newmembers]):
				if newp not in member_live_pointsp:
					member_live_pointsp.append(newp)
			to_handle[newmembers] = False
			i = i + 1
		
		# now we have our members and live points
		member_data_mask = numpy.zeros(len(data_mask), dtype=bool)
		member_data_mask[members] = True
		#print 'returning:', member_data_mask, member_live_pointsp
		yield member_data_mask, member_live_pointsp

def generate_subsets_graph_simple(data_mask, live_pointsp, graph, _):
	# generate data subsets which share points.
	firstmember = numpy.where(data_mask)[0][0]
	# then identify disjoint subgraphs
	for subgraph in networkx.connected_component_subgraphs(graph, copy=False):
		member_data_mask = numpy.zeros(len(data_mask), dtype=bool)
		member_live_pointsp = []
		for nodetype, i in subgraph.nodes():
			if nodetype == 0:
				member_data_mask[i] = True
			else:
				member_live_pointsp.append(i)
		yield member_data_mask, member_live_pointsp
	
def generate_subsets_graph(data_mask, live_pointsp, graph, _):
	# generate data subsets which share points.
	firstmember = numpy.where(data_mask)[0][0]
	allp = numpy.unique(live_pointsp[:,data_mask].flatten())
	if len(live_pointsp[:,firstmember]) == len(allp):
		# trivial case: all live points are the same across data sets
		yield data_mask, live_pointsp[:,firstmember]
		return
	
	subgraphs = list(networkx.connected_component_subgraphs(graph, copy=False))
	if len(subgraphs) == 1:
		yield data_mask, allp
		return
	
	# then identify disjoint subgraphs
	for subgraph in subgraphs:
		print 'networkx subgraph:', subgraph.nodes()
		member_data_mask = numpy.zeros(len(data_mask), dtype=bool)
		member_live_pointsp = []
		for nodetype, i in subgraph.nodes():
			if nodetype == 0:
				member_data_mask[i] = True
			else:
				member_live_pointsp.append(i)
		yield member_data_mask, member_live_pointsp

def generate_subsets_igraph(data_mask, live_pointsp, _graph, graph):
	# generate data subsets which share points.
	firstmember = numpy.where(data_mask)[0][0]
	allp = numpy.unique(live_pointsp[:,data_mask].flatten())
	if len(live_pointsp[:,firstmember]) == len(allp):
		# trivial case: all live points are the same across data sets
		yield data_mask, live_pointsp[:,firstmember]
		return
	
	subgraphs = graph.clusters()
	if len(subgraphs) == 1:
		yield data_mask, allp
		return
	# then identify disjoint subgraphs
	for subgraph in subgraphs:
		#print 'igraph subgraph:', subgraph
		member_data_mask = numpy.zeros(len(data_mask), dtype=bool)
		member_live_pointsp = []
		for vi in subgraph:
			att = graph.vs[vi].attributes()
			#print '    ', att
			if att['vtype'] == 0:
				i = att['nodeid']
				member_data_mask[i] = True
			else:
				p = att['pointid']
				member_live_pointsp.append(p)
		if member_data_mask.any():
			yield member_data_mask, member_live_pointsp
	
data_sets = []

t0 = time.time()
for filename in sys.argv[1:]:
	data = numpy.load(filename)
	data_mask, live_pointsp = data['data_mask'], data['live_pointsp']
	# create graph
	graph = networkx.Graph()
	graph2 = igraph.Graph()
	# pointing from live_point to member
	
	for p in numpy.unique(live_pointsp):
		graph2.add_vertex("p%d" % p, pointid=p, vtype=1)
	for i in numpy.where(data_mask)[0]:
		graph.add_edges_from((((0, i), (1, p)) for p in live_pointsp[:,i]))
		graph2.add_vertex("n%d" % i, nodeid=i, vtype=0)
		graph2.add_edges([("n%d" % i, "p%d" % p) for p in live_pointsp[:,i]])
	
	data_sets.append((data_mask, live_pointsp, graph, graph2))
t1 = time.time()
print 'loading took %fs' % (t1 - t0)

prev_output = []
for implementation in [generate_subsets_reference, generate_subsets_graph_simple, generate_subsets_graph, generate_subsets_igraph]:
	print 'running', implementation
	output = []
	t0 = time.time()
	for a, b, graph, graph2 in data_sets:
		out = list(implementation(a, b, graph, graph2))
		output.append(out)
	t1 = time.time()
	print '   took %fs' % (t1 - t0)
	#for a, b in  zip(output, 
	if prev_output != []:
		print 'checking for correctness...'
		for memberlist1, memberlist2 in zip(output, prev_output):
			memberlist1 = sorted(memberlist1, key=lambda ml: (len(ml), ml[0][0]))
			memberlist2 = sorted(memberlist2, key=lambda ml: (len(ml), ml[0][0]))
			for (md, ml), (md2, ml2) in zip(memberlist1, memberlist2):
				#print len(md), md.sum(), len(md2), md2.sum(), len(ml), len(ml2)
				assert numpy.all(md == md2), (numpy.where(md), numpy.where(md2))
				assert sorted(ml) == sorted(ml2)
			assert len(memberlist1) == len(memberlist2), (len(memberlist1), len(memberlist2))
	prev_output = output


	
	
