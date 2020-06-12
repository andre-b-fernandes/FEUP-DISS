from .user_based_cf import UserBasedImplicitCF
from algorithms.collaborative_filtering.neighborhood import UserNeighborhood


class UserBasedNeighborhood(UserBasedImplicitCF, UserNeighborhood):
    """
    Description
        The class which implements the classic item-based neighborhood
        algorithm which extends UserBasedImplicitCF and UserNeighborhood.
    """
    def __init__(self, matrix=[], similarities=[], co_rated=[],
                 neighbors=[], n_neighbors=5):
        """
        Description
            UserBasedNeighborhood's constructor.

        Arguments
            :param matrix: The ratings matrix.
            :type matrix: list
            :param similarities: The user similarity matrix.
            :type similarities: SymmetricMatrix
            :param co_rated: Co-rated items matrix.
            :type co_rated: SymmetricMatrix
            :param neighbors: The neighborhood model.
            :type neighbors: list
            :param n_neighbors: Number of neighbors to compute.
            :type n_neighbors: int
        """
        super().__init__(matrix, similarities, co_rated)
        UserNeighborhood.__init__(self, neighbors, n_neighbors)

    def new_rating(self, rating):
        """
        Description
            The function which processes a new iteration. Expects a tuple
            (user, item).

        Arguments
            :param rating: The rating tuple.
            :type rating: tuple
        """
        super().new_rating(rating)
        self.neighbors = self._init_neighborhood()
