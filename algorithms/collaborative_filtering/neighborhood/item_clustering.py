from .clustering import Clustering


class ItemClustering(Clustering):
    """
    Description
        A class which clusters items according to their
        similarities and calculates neighborhoods inside these
        clusters.
    """
    def __init__(
        self, neighbors=[], n_neighbors=5, treshold=0.5,
            clusters=[], centroids=[], cluster_map=[]):
        """
        Description
            ItemClustering's constructor.

        Arguments
            :param neighbors: The neighborhood model.
            :type neighbors: list
            :param n_neighbors: Number of neighbors to compute for each item.
            :type n_neighbors: int
            :param treshold: A minimum similarity which pairs need to have for
                clusters.
            :type treshold: float
            :param clusters: The cluster model.
            :type clusters: list
            :param centroids: The centroids model.
            :type centroids: list
            :param cluster_map: The inverted index of elements to their cluster
            :type cluster_map: dictionary
        """
        super().__init__(
            neighbors, n_neighbors, treshold,
            clusters, centroids, cluster_map)

    def _init_centroids(self):
        """
        Description
            A function which initiates the centroids for the
            items.
        """
        return super()._init_centroids(self.items)

    def _init_clusters(self):
        """
        Description
            A function which initiates the clusters for the
            items.
        """
        return super()._init_clusters(self.items)

    def _init_cluster_map(self):
        """
        Description
            A function which initiates the inverted index of
            items to clusters.
        """
        return super()._init_cluster_map(self.items)
