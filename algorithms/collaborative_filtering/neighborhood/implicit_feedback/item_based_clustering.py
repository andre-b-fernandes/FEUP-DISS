from algorithms.collaborative_filtering.neighborhood import Clustering
from .item_based_cf import ItemBasedImplicitCF


class ItemBasedClustering(ItemBasedImplicitCF, Clustering):
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[], neighborhood=[], n_neighbors=5,
            treshold=0.5, clusters=[], centroids=[], cluster_map=[]):
        super().__init__(matrix, intersections, l1, inv_index, similarities)
        self.th = treshold
        self.n_neighbors = n_neighbors
        self.centroids = self._init_model(centroids, self._init_centroids)
        self.clusters = self._init_model(clusters, self._init_clusters)
        self.cluster_map = self._init_model(
            cluster_map, self._init_cluster_map)
        self.neighbors = self._init_model(
            neighborhood, self._init_neighborhood)

    def _init_centroids(self):
        return super()._init_centroids(self.items)

    def _init_clusters(self):
        return super()._init_clusters(self.items)

    def _init_cluster_map(self):
        return super()._init_cluster_map(self.items)

    def new_rating(self, rating):
        _, item_id = rating
        super().new_rating(rating)
        self.increment(item_id)
