import networkx as nx
import numpy as np
from scipy.stats import lognorm, skewnorm, t, weibull_min
from sklearn.neighbors import NearestNeighbors


def generate_skewnormal(n, alpha):
    return skewnorm.rvs(a=alpha, size=n)


def generate_student_t(n, nu):
    return t.rvs(df=nu, size=n)

def generate_lognormal_zero_sigma(size=100, sigma=np.log(5)):
    return lognorm.rvs(s=sigma,scale=1.0,size=size)

def generate_weibull_half_lambda(size=100, lambda_=1.0,k = 0.5):
    return weibull_min.rvs(c=k,scale=lambda_,size=size)

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

#Вычисление числа треугольников
def triangle_count(G: nx.Graph) -> int:
    return sum(nx.triangles(G).values()) // 3