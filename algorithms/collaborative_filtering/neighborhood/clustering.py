from .neighborhood import NeighborhoodCF


class Clustering(NeighborhoodCF):
    def __init__(
            self, matrix, n_neighbors,
            treshold, clusters, centroids, cluster_map):
        super().__init__(matrix, n_neighbors)
        self.clusters = clusters
        self.centroids = centroids
        self.cluster_map = cluster_map
        self.th = treshold

    def _init_neighborhood(self):
        neighbors = []
        for cluster in self.clusters:
            cluster_neighborhood = self._init_neighborhood(cluster)
            neighbors.append(cluster_neighborhood)
        return neighbors

    def neighborhood_of(self, identifier):
        cluster_index = self.cluster_map[identifier]
        return self.neighbors[cluster_index][identifier]

    def increment(self, identifier):
        sims = [self.similarity_between(
            identifier, centroid) for centroid in self.centroids]
        max_sim = max(sims)
        if max_sim < self.th:
            self.centroids.append(identifier)
            self.clusters.append([identifier])
        else:
            centroid_index = sims.index(max_sim)
            self.clusters[centroid_index].append(identifier)
            self.cluster_map[identifier] = centroid_index
            cluster = self.clusters[centroid_index]
            self.neighbors[centroid_index] = super()._init_neighborhood(
                cluster)


class ClusteringItem(Clustering):
    def _init_centroids(self):
        for item in self.items:
            self.increment(item)


class ClusteringUser(Clustering):
    def _init_centroids(self):
        for user in self.users:
            self.increment(user)
