from algorithms.collaborative_filtering.neighborhood import NeighborhoodCF


class UserNeighborhood(NeighborhoodCF):
    def __init__(self, neighbors=[], n_neighbors=5):
        super().__init__(neighbors, n_neighbors)

    def _init_neighborhood(self):
        return super()._init_neighborhood(self.users)
