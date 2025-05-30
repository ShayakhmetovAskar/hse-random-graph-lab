import networkx as nx
import numpy as np
from scipy.stats import lognorm, skewnorm, t, weibull_min
from sklearn.neighbors import NearestNeighbors

SIGMA_CONST = np.log(5)


def generate_skewnormal(n, alpha):
    return skewnorm.rvs(a=alpha, size=n)


def generate_student_t(n, nu):
    return t.rvs(df=nu, size=n)


def generate_lognormal_zero_sigma(size=100, sigma=SIGMA_CONST):
    return lognorm.rvs(s=sigma, scale=1.0, size=size)


def generate_weibull_half_lambda(size=100, lambda_=1.0, k=0.5):
    return weibull_min.rvs(c=k, scale=lambda_, size=size)


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
            if i in neighbors[j]:  # Проверяем взаимное соседство
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


def compute_number_of_components(G):
    """Вычисление числа компонент"""
    return nx.number_connected_components(G)


def clique_number(G):
    return max(len(c) for c in nx.find_cliques(G))


def safe_compute(func, G, default=0):
    """Безопасное вычисление характеристики с обработкой ошибок"""
    try:
        return func(G)
    except Exception:
        return default


def max_degree(G):
    """Максимальная степень вершины"""
    if len(G.nodes()) == 0:
        return 0
    return max(dict(G.degree()).values())


def min_degree(G):
    """Минимальная степень вершины"""
    if len(G.nodes()) == 0:
        return 0
    return min(dict(G.degree()).values())


def num_components(G):
    """Количество компонент связности"""
    return nx.number_connected_components(G)


def articulation_points(G):
    """Количество точек сочленения"""
    return len(list(nx.articulation_points(G)))


def triangle_count(G):
    """Количество треугольников"""
    return sum(nx.triangles(G).values()) // 3


def average_degree(G):
    """Средняя степень вершины"""
    if len(G.nodes()) == 0:
        return 0
    return np.mean(list(dict(G.degree()).values()))


def density(G):
    """Плотность графа"""
    return nx.density(G)


def clustering_coefficient(G):
    """Средний коэффициент кластеризации"""
    return nx.average_clustering(G)


def diameter_largest_component(G):
    """Диаметр наибольшей компоненты связности"""
    if G.number_of_nodes() == 0:
        return 0

    components = list(nx.connected_components(G))
    if not components:
        return 0

    largest_cc = max(components, key=len)
    if len(largest_cc) <= 1:
        return 0

    subgraph = G.subgraph(largest_cc)
    try:
        return nx.diameter(subgraph)
    except Exception:
        return 0


def edge_connectivity(G):
    """Рёберная связность"""
    if G.number_of_nodes() < 2:
        return 0
    try:
        return nx.edge_connectivity(G)
    except Exception:
        return 0

def get_generator(distribution_name, params):
    """Возвращает функцию-генератор для заданного распределения"""
    generators = {
        "skewnorm": lambda size: skewnorm.rvs(a=params.get("alpha", 0), size=size, random_state=None),
        "student_t": lambda size: t.rvs(df=params.get("nu", 1), size=size, random_state=None),
    }

    if distribution_name not in generators:
        raise ValueError(f"Unknown distribution: {distribution_name}. Available: {list(generators.keys())}")

    return generators[distribution_name]
