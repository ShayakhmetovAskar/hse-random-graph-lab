from typing import List, Set
import numpy as np
from scipy.stats import skewnorm, t
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_skewnormal(n, alpha):
    return skewnorm.rvs(a=alpha, size=n)


def generate_student_t(n, nu):
    return t.rvs(df=nu, size=n)


# Построение KNN-графа
def build_knn_graph(data, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(data.reshape(-1, 1))  # +1 потому что включает саму точку
    distances, indices = nn.kneighbors(data.reshape(-1, 1))
    # Удаляем первый элемент (саму точку)
    indices = indices[:, 1:]

    # Строим словарь соседства
    neighbors = {i: set(indices[i]) for i in range(len(data))}

    # Создаем граф только с двусторонними отношениями
    G = nx.Graph()
    for i in range(len(data)):
        for j in neighbors[i]:
            if i in neighbors[j]:  # взаимное соседство
                G.add_edge(i, j)
    return G


# Построение дистанционного графа
def build_distance_graph(data, d):
    G = nx.Graph()
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if abs(data[i] - data[j]) <= d:
                G.add_edge(i, j)
    return G


# Вычисление числа компонент связности
def compute_number_of_components(G):
    return nx.number_connected_components(G)

# Вычисление кликового числа 
def clique_number(G):
    return max(len(c) for c in nx.find_cliques(G))
