#!/usr/local/bin/python
import random
import itertools

import numpy as np
from numpy import linalg
from scipy.sparse import csgraph
import networkx as nx
import networkx.algorithms.cuts as cuts

import utils

# numpy settings
np.set_printoptions(precision=4,linewidth=120)

# create random graph
A = utils.random_graph(n=6, density=0.7, simple=True, directed=False)
G = nx.DiGraph(A)
directed = nx.is_directed(G)
print("Adjacency:\n"+str(A))

# compute degree matrix and laplacian
D = np.zeros(A.shape)
for j in range(len(A)):
    D[j,j] = sum(A[:,j])
L = csgraph.laplacian(A)#, normed=True)
print("Degree:\n"+str(D))
print("Laplacian:\n"+str(L))

# eigenvalues, eigenvectors
eigvalsA, eigvecsA = utils.eigs(A, sort=True, reverse=True)
eigvalsD, eigvecsD = utils.eigs(D, sort=False)
eigvalsL, eigvecsL = utils.eigs(L, sort=True, reverse=False)
print("eigvals(A):\n" +str(eigvalsA))
print("eigvecs(A):\n" +str(eigvecsA))
print("eigvals(D):\n" +str(eigvalsD))
print("eigvecs(D):\n" +str(eigvecsD))
print("eigvals(L):\n" +str(eigvalsL))
print("eigvecs(L):\n" +str(eigvecsL))

be = utils.boundary_expansion(G, max_subset_proportion=0.5)
ne = utils.node_expansion(G, max_subset_proportion=0.5)
scc = sorted(nx.strongly_connected_components(G), key = len, reverse=True)
#cc = sorted(nx.connected_components(G), key = len, reverse=True) if not directed else None
#alg_conn = nx.linalg.algebraic_connectivity(G) if not directed else None

spectral_gap = eigvalsA[0] - eigvalsA[1]
print("spectral gap (eig1M - eig2M) of A: ", spectral_gap)
print("spectral gap (eig1m) of L: ", eigvalsL[len(scc)])

# compute number of strongly connected components
if(nx.is_directed(G)):
    print("strongly_connected_components: ", len(scc))
    print("boundary_expansion: ", be)
    print("node_expansion: ", ne)
else:
    cc = sorted(nx.connected_components(G), key = len, reverse=True)
    print("connected_components: ", len(cc))
    print("strongly_connected_components: ", len(scc))
    print("boundary_expansion: ", be)
    print("node_expansion: ", ne)
    print("algebraic_connectivity: ", nx.linalg.algebraic_connectivity(G))
