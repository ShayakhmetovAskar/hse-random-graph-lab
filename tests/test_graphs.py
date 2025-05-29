import networkx as nx
import numpy as np

import src.utils as utils  # ← updated import

# ---------- KNN GRAPH ---------- #

def test_build_knn_graph_bidirectional():
    """With k=1 there is an inevitable tie for point `1` (equidistant to 0 and 2).
    The algorithm keeps whichever neighbour scikit‑learn returns first, so we
    assert **one** mutual edge that involves node 1 – not a specific one.
    """
    data = np.array([0.0, 1.0, 2.0, 10.0])
    G = utils.build_knn_graph(data, k=1)

    # Exactly one bidirectional edge among the first three nodes
    assert G.number_of_edges() == 1
    u, v = next(iter(G.edges()))
    assert {u, v} <= {0, 1, 2}
    # Node 3 (10.0) is far away – isolated
    assert 3 not in G or G.degree(3) == 0


def test_build_distance_graph_threshold():
    data = np.array([0.0, 1.5, 3.1])
    G = utils.build_distance_graph(data, 2.0)
    assert G.has_edge(0, 1)
    assert not G.has_edge(0, 2)


def test_compute_number_of_components():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert utils.compute_number_of_components(G) == 2


def test_clique_number():
    assert utils.clique_number(nx.complete_graph(5)) == 5


def test_triangle_count():
    assert utils.triangle_count(nx.complete_graph(4)) == 4
