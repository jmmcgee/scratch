#!/usr/local/bin/python
import utils
import numpy as np
from numpy import linalg as alg
import networkx as nx
#from networkx.utils import (powerlaw_sequence, create_degree_sequence)

arr = np.array([[1., 1., 1., 1., 1.], [2., 2., 2.]])
arr = np.identity(5)

# should create scale-free digraph 
num = 100
alpha=0.25
beta=0.70
gamma=0.05
print(alpha+beta+gamma)
print(alpha+beta+gamma == 1.0)
delta_in=0.4
delta_out=0.6
seed=325327587528/17
#sequence = create_degree_sequence(num, powerlaw_sequence, exponent=exp)
#graph = nx.configuration_model(sequence, seed=seed)
graph = nx.scale_free_graph(n=num,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta_in=delta_in,
        delta_out=delta_out,
        seed=seed)

# remove parallel edges and self-loops
loops = graph.selfloop_edges()
graph.remove_edges_from(loops)
#graph = nx.Graph(graph)

print("graph order: " + str(graph.order()))
print("graph size: " + str(graph.size()))
utils.rank_deg_plot(graph)

# remove nodes with zero in-edges
#zero_edges = filter(lambda out: out[1] == 1, graph.out_degree_iter())

zero_in_nodes = [v for (v,deg) in graph.in_degree_iter() if deg == 0]
print("zero_in_nodes: " + str(len(zero_in_nodes)))
graph.remove_nodes_from(zero_in_nodes)

zero_in_nodes = [v for (v,deg) in graph.in_degree_iter() if deg == 0]
print("zero_in_nodes: " + str(len(zero_in_nodes)))

# get largest connected component
# unfortunately, the iterator over the components is not guaranteed to be sorted by size
#components = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
#lcc = graph.subgraph(components[0])

print("graph order: " + str(graph.order()))
print("graph size: " + str(graph.size()))
utils.rank_deg_plot(graph)

from networkx_viewer import Viewer
app = Viewer(graph)
#app.mainloop()
