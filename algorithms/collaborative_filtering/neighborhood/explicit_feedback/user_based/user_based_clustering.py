from .user_based_cf import UserBasedExplicitCF
from algorithms.collaborative_filtering.neighborhood import UserClustering


class UserBasedClustering(UserBasedExplicitCF, UserClustering):
    def __init__(
        self, matrix=[], similarities=[], co_rated=[],
        neighbors=[], n_neighbors=5, treshold=0.5,
            clusters=[], centroids=[], cluster_map=[]):
        super().__init__(matrix, similarities, co_rated)
        super(UserClustering, self).__init__(
            neighbors, n_neighbors, treshold,
            clusters, centroids, cluster_map)

    def new_rating(self, rating):
        user_id, _, _ = rating
        super().new_rating(rating)
        self.increment(user_id)
