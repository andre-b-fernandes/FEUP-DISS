from .user_based_cf import UserBasedExplicitCF
from algorithms.collaborative_filtering.neighborhood import UserNeighborhood


class UserBasedNeighborhood(UserBasedExplicitCF, UserNeighborhood):
    def __init__(
            self, matrix=[], similarities=[], avg_ratings=dict(), co_rated=[],
            neighbors=[], n_neighbors=5):
        super().__init__(matrix, similarities, avg_ratings, co_rated)
        super(UserNeighborhood, self).__init__(neighbors, n_neighbors)

    def new_rating(self, rating):
        super().new_rating(rating)
        self.neighbors = self._init_neighborhood()
