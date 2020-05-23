from random import sample
from algorithms.collaborative_filtering.neighborhood import NeighborhoodCF
from data_structures import DynamicArray


class Clustering(NeighborhoodCF):
    def __init__(
        self, neighbors=[], n_neighbors=5, treshold=0.5, clusters=[],
            centroids=[], cluster_map=[]):
        self.th = treshold
        self.centroids = self._init_model(centroids, self._init_centroids)
        self.clusters = self._init_model(clusters, self._init_clusters)
        self.cluster_map = self._init_model(
            cluster_map, self._init_cluster_map)
        super().__init__(neighbors, n_neighbors)

    def _init_centroids(self, elements):
        if len(elements) == 0:
            return []
        return sample(elements, 1)

    def _init_clusters(self, elements):
        clusters = []
        for element in elements:
            sims = [self.similarity_between(
                element, centroid) for centroid in self.centroids]
            max_sim = max(sims)
            if max_sim < self.th:
                self.centroids.append(element)
                clusters.append([element])
            else:
                centroid_index = sims.index(max_sim)
                clusters[centroid_index].append(element)
        return clusters

    def _init_cluster_map(self, elements):
        cluster_map = dict()
        for element in elements:
            for index, cluster in enumerate(self.clusters):
                if element in cluster:
                    cluster_map[element] = index
                    break
        return cluster_map

    def _init_neighborhood(self):
        neighbors = DynamicArray(
            default_value=lambda: DynamicArray(default_value=lambda: list()))
        for cluster in self.clusters:
            cluster_neighborhood = self._init_neighborhood(cluster)
            neighbors.append(cluster_neighborhood)
        return neighbors

    def neighborhood_of(self, identifier):
        try:
            cluster_index = self.cluster_map[identifier]
            return self.neighbors[cluster_index][identifier]
        except KeyError:
            return []

    def increment(self, identifier):
        sims = [self.similarity_between(
            identifier, centroid) for centroid in self.centroids]
        try:
            max_sim = max(sims)
        except ValueError:
            max_sim = 0
        if max_sim < self.th:
            self.centroids.append(identifier)
            self.clusters.append([identifier])
            self.cluster_map[identifier] = len(self.clusters) - 1
        else:
            centroid_index = sims.index(max_sim)
            self.clusters[centroid_index].append(identifier)
            self.cluster_map[identifier] = centroid_index
            cluster = self.clusters[centroid_index]
            self.neighbors[centroid_index] = super()._init_neighborhood(
                cluster)
