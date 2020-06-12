from .user_based_cf import UserBasedImplicitCF
from algorithms.collaborative_filtering.neighborhood import UserClustering


class UserBasedClustering(UserBasedImplicitCF, UserClustering):
    """
    Description
        A class which implements the user-based neighborhood clustering
        algorithm for implicit feedback.
        Extends UserBasedImplicitCF and UserClustering.
    """
    def __init__(
        self, matrix=[], similarities=[], co_rated=[],
        neighbors=[], n_neighbors=5, treshold=0.5,
            clusters=[], centroids=[], cluster_map=[]):
        """
        Description
            UserBasedClustering's constructor.

        Arguments
            :param matrix: The ratings matrix.
            :type matrix: list
            :param similarities: The similarity matrix.
            :type similarities: SymmetricMatrix
            :param avg_ratings: Users' average ratings.
            :type avg_ratings: DynamicArray
            :param co_rated: The co-rated items matrix.
            :type co_rated: SymmetricMatrix
            :param neighbors: The neighborhood model.
            :type neighbors: list
            :param n_neighbors: Number of neighbors to compute for each user.
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
        super().__init__(matrix, similarities, co_rated)
        UserClustering.__init__(
            self, neighbors, n_neighbors, treshold,
            clusters, centroids, cluster_map)

    def new_rating(self, rating):
        """
        Description
            The function which processes a new iteration. Expects a tuple
            (user, item)

        Arguments
            :param rating: The rating tuple.
            :type rating: tuple
        """
        user_id, _ = rating
        super().new_rating(rating)
        self.increment(user_id)
