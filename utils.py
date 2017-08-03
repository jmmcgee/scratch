#!/usr/local/bin/python
import random
import itertools

import numpy as np
from numpy import linalg
import networkx as nx
import networkx.algorithms.cuts as cuts
import matplotlib.pyplot as plt


"""
Random graph from given degree sequence.
Draw degree rank plot and graph with matplotlib.
"""
__author__ = """Aric Hagberg <aric.hagberg@gmail.com>"""
import networkx as nx


def rank_deg_plot(G=None):
    if G == None:
        G = nx.gnp_random_graph(100,0.02)
    #G = 

    degree_sequence=sorted([deg for (v,deg) in nx.degree(G)], reverse=True) # degree sequence
    #print "Degree sequence", degree_sequence
    dmax=max(degree_sequence)

    #plt.loglog(degree_sequence,'b-',marker='o')
    plt.plot(degree_sequence,'b-',marker='o')
    plt.title("In-Degree rank plot")
    plt.ylabel("in-degree")
    plt.xlabel("rank")

    # draw graph in inset
    plt.axes([0.45,0.45,0.45,0.45])
    if( nx.is_directed(G) ):
        Gcc=sorted(nx.strongly_connected_component_subgraphs(G), key = len, reverse=True)[0]
    else:
        Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
    pos=nx.spring_layout(Gcc)
    plt.axis('off')
    nx.draw_networkx_nodes(Gcc,pos,node_size=20)
    nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

    plt.savefig("degree_histogram.png")
    plt.show()

def random_graph(n=16, density=0.5, simple=False, directed=False):
    A = np.zeros((n,n))

    # randomly add edges
    num_edges = 0
    max_edges = int((n*n - n*int(simple))*density)
    assert(density <= 1.)
    assert(max_edges <= n*n - n*int(simple))
    while num_edges < max_edges:
        r = random.randint(0,n-1)
        s = random.randint(0,n-1)
        while (simple and r == s) or A[r,s] != 0:
            r = random.randint(0,n-1)
            s = random.randint(0,n-1)
        A[r,s] = 1
        num_edges += 1
        if not directed and r != s:
            A[s,r] = A[r,s]
            num_edges +=1
    return A
    #G = nx.DiGraph(a)

def eigs(M, sort=False, reverse=True):
    vals, vecs = linalg.eig(M)
    if sort:
        both = sorted(zip(vals, vecs), key=lambda(val, vec): val,  reverse=reverse)
        vecs = np.array([vec for (val, vec) in both])
        vals = np.array([val for (val, vec) in both])
    return (vals, vecs)

def boundary_expansion(G, max_subset_proportion=0.5, max_subset_size=None):
    # expansion should be initially larger than the largest possible expansion
    boundary_expansion = len(G.nodes) + len(G.edges)

    # set size of max subset to iterator over to find minimum expansion
    if max_subset_size == None:
        max_subset_size=int(len(G.nodes) * max_subset_proportion)

    # iterate over all subsets of size 1 to max_subset_size
    count = 0
    for size in range(max_subset_size):
        for S in itertools.combinations(G.nodes, size):
            if not S: continue
            count += 1
            be = cuts.boundary_expansion(G,S)
            if be < boundary_expansion:
                boundary_expansion = be
    return boundary_expansion

def node_expansion(G, max_subset_proportion=0.5, max_subset_size=None):
    # expansion should be initially larger than the largest possible expansion
    node_expansion = len(G.nodes) + len(G.edges)

    # set size of max subset to iterator over to find minimum expansion
    if max_subset_size == None:
        max_subset_size=int(len(G.nodes) * max_subset_proportion)

    # iterate over all subsets of size 1 to max_subset_size
    count = 0
    for size in range(max_subset_size):
        for S in itertools.combinations(G.nodes, size):
            if not S: continue
            count += 1
            ne = cuts.node_expansion(G,S)
            if ne < node_expansion:
                node_expansion = ne
    return node_expansion

