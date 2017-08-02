#!/usr/local/bin/python
import random
import itertools

import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.csgraph import csgraph_to_dense as to_dense
import networkx as nx
import networkx.algorithms.cuts as cuts

import utils

# numpy settings
np.set_printoptions(precision=4,linewidth=120)

# create random graph
A = utils.random_graph(n=5, density=0.7, simple=True, directed=False)
#G = nx.Graph(A)
G = nx.Graph(A)
directed = nx.is_directed(G)
print("Adjacency:\n"+str(A))

# compute degree matrix
inD = np.zeros(A.shape)
outD = np.zeros(A.shape)
for i in range(len(A)):
    inD[i,i] = sum(A[:,i])
    outD[i,i] = sum(A[i,:])
D = inD
print("Degree:\n"+str(D))

# compute laplacian
if(nx.is_directed(G)):
    L = nx.linalg.directed_laplacian_matrix(G)
else:
    #L = csgraph.laplacian(A)#, normed=True)
    L = nx.linalg.laplacian_matrix(G)
L = to_dense(L)
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

#Aspectrum = np.array(sorted(nx.linalg.adjacency_spectrum(G), reverse=True))
#Lspectrum = np.array(sorted(nx.linalg.laplacian_spectrum(G), reverse=False))
#print("spetrum of A: "+str(Aspectrum))
#print("spetrum of L: "+str(Lspectrum))

be = utils.boundary_expansion(G, max_subset_proportion=0.5)
ne = utils.node_expansion(G, max_subset_proportion=0.5)
scc = sorted(nx.strongly_connected_components(G), key = len, reverse=True) if directed else None
cc = sorted(nx.connected_components(G), key = len, reverse=True) if not directed else None
cc = scc if cc == None else cc
alg_conn = nx.linalg.algebraic_connectivity(G) if not directed else None

spectral_gap = eigvalsA[0] - eigvalsA[1]
print("spectral gap? (eig1M - eig2M) of A: ", spectral_gap)
print("spectral gap? (eig1m) of L: ", eigvalsL[len(cc)])

# compute number of strongly connected components
if(nx.is_directed(G)):
    print("strongly_connected_components: ", len(scc))
    print("boundary_expansion: ", be)
    print("node_expansion: ", ne)
else:
    cc = sorted(nx.connected_components(G), key = len, reverse=True)
    print("connected_components: ", len(cc))
    print("boundary_expansion: ", be)
    print("node_expansion: ", ne)
    print("algebraic_connectivity: ", alg_conn)
