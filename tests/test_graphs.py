"""Unit tests for KNN & distance graph utilities."""

import numpy as np
import networkx as nx

import module  # assumes module.py is at repo root


# ---------- KNN GRAPH ---------- #

def test_build_knn_graph_bidirectional():
    data = np.array([0.0, 1.0, 2.0, 10.0])
    G = module.build_knn_graph(data, k=1)

    # Nodes 0–1 and 1–2 should be mutual nearest neighbours
    assert G.has_edge(0, 1)
    assert G.has_edge(1, 2)
    # Node 3 is far away – should be isolated
    assert 3 not in G or G.degree(3) == 0


# ---------- DISTANCE GRAPH ---------- #

def test_build_distance_graph_threshold():
    data = np.array([0.0, 1.5, 3.1])
    d = 2.0
    G = module.build_distance_graph(data, d)
    assert G.has_edge(0, 1)
    assert not G.has_edge(0, 2)


# ---------- CONNECTED COMPONENTS ---------- #

def test_compute_number_of_components():
    # Two disconnected triangles => 2 components
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert module.compute_number_of_components(G) == 2


# ---------- CLIQUE NUMBER ---------- #

def test_clique_number():
    G = nx.complete_graph(5)
    assert module.clique_number(G) == 5


# ---------- TRIANGLE COUNT ---------- #

def test_triangle_count():
    # K_4 has 4 triangles
    G = nx.complete_graph(4)
    assert module.triangle_count(G) == 4
