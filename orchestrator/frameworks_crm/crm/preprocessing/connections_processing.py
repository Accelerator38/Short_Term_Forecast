from abc import ABCMeta, abstractmethod
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


class WellsConnectionsGraph():# граф для хранения соединений между скважинами
    """
    Class for storing connections between wells. Different initializations are supported.
    """

    def __init__(self, coordinates, connectionsInit):
        """

        Parameters
        ----------
        coordinates pd.Dataframe with columns named X and Y
        connectionsInit ConnectionsInit edges initializer
        """
        self.verts = coordinates
        self.edges = connectionsInit(coordinates)

    def adgacency_matrix(self): # функция матрицы смежности
        n_verts = len(self.verts)
        adj = np.zeros((n_verts, n_verts), dtype=np.bool)
        for n1, n2 in self.edges:
            adj[n1, n2], adj[n2, n1] = True, True
        return adj


class ConnectionsInit(metaclass=ABCMeta): #класс различных инициализаций
    """
    Base class for different initializations of WellsConnectionsGraph.
    """

    @abstractmethod
    def __call__(self, coordinates):
        pass


class ConnectionsInitByDist(ConnectionsInit):# инициализация с полносвязным графом, после фильтрация
    """
    Initializations with creates fully connected graph, which is followed by filtration of spaced nodes.
    """

    def __init__(self, dist_thres=1e3):
        self.dist_thres = dist_thres

    def __call__(self, coordinates, safe_dist_mtrx=False):
        pairwise_dist = pairwise_distances(coordinates.as_matrix())
        if safe_dist_mtrx:
            self.pairwise_dist = pairwise_dist

        adj_matrix = pairwise_dist < self.dist_thres
        adj_matrix = np.triu(adj_matrix, k=1)
        return [index for index, x in np.ndenumerate(adj_matrix) if x]


class ConnectionsInitByAngleAndDist(ConnectionsInitByDist): # тоже самое, что и предыдущий класс, но с учётом параллельных краёв
    """
    Initializations with creates fully connected graph, which is followed by filtration of spaced nodes and
    nearly parallel edges.
    """

    def __init__(self, dist_thres=1e3, angle_thres=10):
        super().__init__(dist_thres)
        self.angle_thres = angle_thres * np.pi / 180

    def _get_angle(self, a, b, c):
        return np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

    def _in_edges(self, n1, n2, edges):
        return (n1, n2) in edges or (n2, n1) in edges

    def _filter_edges(self, e, edges):
        n1, n2 = e
        verts_num = self.pairwise_dist.shape[0]
        for n3 in [n for n in range(verts_num) if self._in_edges(n, n1, edges) and self._in_edges(n, n2, edges)]:
            n1_n2 = self.pairwise_dist[n1, n2]
            n1_n3 = self.pairwise_dist[n1, n3]
            n2_n3 = self.pairwise_dist[n2, n3]
            max_e = np.max([n1_n2, n1_n3, n2_n3])
            if n1_n2 == max_e and self._get_angle(n1_n2, n1_n3, n2_n3) > np.pi - 2 * self.angle_thres:
                return False
        return True

    def __call__(self, coordinates):
        edges = super().__call__(coordinates, safe_dist_mtrx=True)
        edges = [e for e in edges if self._filter_edges(e, edges)]
        del self.pairwise_dist
        return edges
