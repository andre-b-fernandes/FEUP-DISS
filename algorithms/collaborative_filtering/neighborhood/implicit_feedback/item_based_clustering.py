from algorithms.collaborative_filtering.neighborhood import (ClusteringItem)
from .item_based_cf import ItemBasedImplicitCF


class ItemBasedClustering(ItemBasedImplicitCF, ClusteringItem):
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[], neighborhood=[], n_neighbors=5):
        super().__init__(matrix, intersections, l1, inv_index, similarities)
        ClusteringItem().__init__(matrix, n_neighbors)

    def new_rating(self, rating):
        _, item_id = rating
        super().new_rating(rating)
        self.increment(item_id)
