from .clustering import Clustering


class UserClustering(Clustering):
    def __init__(
        self, neighbors=[], n_neighbors=5, treshold=0.5,
            clusters=[], centroids=[], cluster_map=[]):
        super().__init__(
            neighbors, n_neighbors, treshold,
            clusters, centroids, cluster_map)

    def _init_centroids(self):
        return super()._init_centroids(self.users)

    def _init_clusters(self):
        return super()._init_clusters(self.users)

    def _init_cluster_map(self):
        return super()._init_cluster_map(self.users)
