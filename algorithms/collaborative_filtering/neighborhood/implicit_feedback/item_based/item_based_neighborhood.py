from algorithms.collaborative_filtering.neighborhood import ItemNeighborhood
from .item_based_cf import ItemBasedImplicitCF


class ItemBasedNeighborhood(ItemBasedImplicitCF, ItemNeighborhood):
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[], neighborhood=[], n_neighbors=5):
        super().__init__(matrix, intersections, l1, inv_index, similarities)
        super(ItemNeighborhood, self).__init__(neighborhood, n_neighbors)

    def new_rating(self, rating):
        super().new_rating(rating)
        self.neighbors = self._init_neighborhood()
