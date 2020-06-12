from .user_based_cf import UserBasedExplicitCF
from algorithms.collaborative_filtering.neighborhood import UserNeighborhood


class UserBasedNeighborhood(UserBasedExplicitCF, UserNeighborhood):
    """
    Description
        A class which implements the classic user-based neighborhood
        algorithm. Extends UserBasedExplicitCF and UserClustering.
    """
    def __init__(
            self, matrix=[], similarities=[], avg_ratings=dict(), co_rated=[],
            neighbors=[], n_neighbors=5):
        """
        Description
            UserBasedNeighborhood's constructor.

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
            :param n_neighbors: Number of neighbors to compute.
            :type n_neighbors: int
        """
        super().__init__(matrix, similarities, avg_ratings, co_rated)
        super(UserNeighborhood, self).__init__(neighbors, n_neighbors)

    def new_rating(self, rating):
        """
        Description
            The function which processes a new iteration. Expects a tuple
            (user, item)

        Arguments
            :param rating: The rating tuple.
            :type rating: tuple
        """
        super().new_rating(rating)
        self.neighbors = self._init_neighborhood()
