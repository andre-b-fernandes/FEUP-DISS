from .user_based_cf import UserBasedImplicitCF
from algorithms.collaborative_filtering.neighborhood import UserClustering


class UserBasedClustering(UserBasedImplicitCF, UserClustering):
    def __init__(
        self, matrix=[], similarities=[], co_rated=[],
        neighbors=[], n_neighbors=5, treshold=0.5,
            clusters=[], centroids=[], cluster_map=[]):
        super().__init__(matrix, similarities, co_rated)
        UserClustering.__init__(
            self, neighbors, n_neighbors, treshold,
            clusters, centroids, cluster_map)

    def new_rating(self, rating):
        user_id, _ = rating
        super().new_rating(rating)
        self.increment(user_id)
