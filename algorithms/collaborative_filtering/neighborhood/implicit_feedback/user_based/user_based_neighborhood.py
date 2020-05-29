from .user_based_cf import UserBasedImplicitCF
from algorithms.collaborative_filtering.neighborhood import UserNeighborhood


class UserBasedNeighborhood(UserBasedImplicitCF, UserNeighborhood):
    def __init__(self, matrix=[], similarities=[], co_rated=[],
                 neighbors=[], n_neighbors=5):
        super().__init__(matrix, similarities, co_rated)
        UserNeighborhood.__init__(self, neighbors, n_neighbors)

    def new_rating(self, rating):
        super().new_rating(rating)
        self.neighbors = self._init_neighborhood()
