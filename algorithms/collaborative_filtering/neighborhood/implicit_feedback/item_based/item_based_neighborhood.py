from algorithms.collaborative_filtering.neighborhood import ItemNeighborhood
from .item_based_cf import ItemBasedImplicitCF


class ItemBasedNeighborhood(ItemBasedImplicitCF, ItemNeighborhood):
    """
    Description
        The class which implements the classic item-based neighborhood
        algorithm which extends ItemBasedImplicitCF and ItemNeighborhood.
    """
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[], neighborhood=[], n_neighbors=5):
        """
        Description
            ItemBasedNeighborhood's constructor.

        Arguments
            :param matrix: The ratings matrix.
            :type matrix: list
            :param intersections: A matrix of item intersections.
            :type intersections: SymmetricMatrix
            :param l1: An array of items' l1 norms.
            :type l1: DynamicArray
            :param inv_index: An inverted index of users to items.
            :type inv_index: defaultdict(set)
            :param similarities: A similarity matrix.
            :type similarities: SymmetricMatrix
            :param neighbors: The neighborhood model.
            :type neighbors: list
            :param n_neighbors: Number of neighbors to compute.
            :type n_neighbors: int
        """
        super().__init__(matrix, intersections, l1, inv_index, similarities)
        ItemNeighborhood.__init__(self, neighborhood, n_neighbors)

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
