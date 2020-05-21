from algorithms.collaborative_filtering.neighborhood import (
    NeighborhoodCF)
from .item_based_cf import ItemBasedImplicitCF


class ItemBasedNeighborhood(ItemBasedImplicitCF, NeighborhoodCF):
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[], neighborhood=[], n_neighbors=5):
        super().__init__(matrix, intersections, l1, inv_index, similarities)
        NeighborhoodCF().__init__(matrix, n_neighbors)

    def _init_neighborhood(self):
        return NeighborhoodCF()._init_neighborhood(self.items)

    def new_rating(self, rating):
        super().new_rating(rating)
        self.neighbors = self._init_neighborhood()
