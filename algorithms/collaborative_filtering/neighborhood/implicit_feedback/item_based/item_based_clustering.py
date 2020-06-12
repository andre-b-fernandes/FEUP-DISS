from algorithms.collaborative_filtering.neighborhood import ItemClustering
from .item_based_cf import ItemBasedImplicitCF


class ItemBasedClustering(ItemBasedImplicitCF, ItemClustering):
    """
    Description
        A class which implements the item-based neighborhood
        clustering algorithm extending ItemBasedImplicitCF and
        ItemClustering.
    """
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[], neighborhood=[], n_neighbors=5,
            treshold=0.5, clusters=[], centroids=[], cluster_map=[]):
        """
        Description
            ItemBasedClustering's constructor.

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
            :param neighborhood: The neighborhood model.
            :type neighborhood: list
            :param n_neighbors: Number of neighbors to compute for each item.
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
        super().__init__(matrix, intersections, l1, inv_index, similarities)
        ItemClustering.__init__(
            self, neighborhood, n_neighbors, treshold, clusters, centroids,
            cluster_map)

    def new_rating(self, rating):
        """
        Description
            The function which processes a new iteration. Expects a tuple
            (user, item).

        Arguments
            :param rating: The rating tuple.
            :type rating: tuple
        """
        _, item_id = rating
        super().new_rating(rating)
        self.increment(item_id)
