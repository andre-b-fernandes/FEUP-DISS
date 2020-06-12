from .neighborhood import NeighborhoodCF


class ItemNeighborhood(NeighborhoodCF):
    """
    Description
        A class which aims at calculating neighborhood
        models of items using standard knn and extends
        NeighborhoodCF.
    """
    def __init__(self, neighbors, n_neighbors):
        """
        Description
            ItemNeighborhood's constructor.

        Arguments
            :param neighbors: The neighborhood model.
            :type neighbors: list
            :param n_neighbors: Number of neighbors to compute.
            :type n_neighbors: int
        """
        super().__init__(neighbors, n_neighbors)

    def _init_neighborhood(self):
        """
        Description
            A function which computes and returns
            a neighborhood for the item set, returning
            a DynamicArray.
        """
        return super()._init_neighborhood(self.items)
