from algorithms.collaborative_filtering.neighborhood import ItemClustering
from .item_based_cf import ItemBasedImplicitCF


class ItemBasedClustering(ItemBasedImplicitCF, ItemClustering):
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[], neighborhood=[], n_neighbors=5,
            treshold=0.5, clusters=[], centroids=[], cluster_map=[]):
        super().__init__(matrix, intersections, l1, inv_index, similarities)
        ItemClustering.__init__(
            self, neighborhood, n_neighbors, treshold, clusters, centroids,
            cluster_map)

    def new_rating(self, rating):
        _, item_id = rating
        super().new_rating(rating)
        self.increment(item_id)
