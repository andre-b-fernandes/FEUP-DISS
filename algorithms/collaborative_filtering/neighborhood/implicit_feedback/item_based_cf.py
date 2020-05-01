from algorithms.collaborative_filtering import CollaborativeFiltering
from data_structures import SymmetricMatrix, DynamicArray
from collections import defaultdict
from utils import cosine_similarity, mode
from math import sqrt

ITEM_INTERSECTION_KEY = "ITEM_INTERSECTIONS"
ITEMS_L1_NORMS_KEY = "ITEMS_L1_NORMS"
INVERTED_INDEX_KEY = "INVERTED_INDEX"
ITEM_SIMILARITIES_KEY = "ITEM_SIMILARITIES"
ITEM_NEIGHBORHOOD_KEY = "ITEM_NEIGHBORHOOD"


class ItemBasedImplicitCF(CollaborativeFiltering):
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[], neighborhood=[], n_neighbors=5):
        super().__init__(matrix)
        self._init_model(inv_index, INVERTED_INDEX_KEY, self._init_inv_index)
        self._init_model(
            intersections, ITEM_INTERSECTION_KEY, self._init_intersections)
        self._init_model(l1, ITEMS_L1_NORMS_KEY, self._init_l1)
        self._init_model(
            similarities, ITEM_SIMILARITIES_KEY, self._init_similarities)
        self._init_model(
            neighborhood, ITEM_NEIGHBORHOOD_KEY, self._init_neighborhood)
        self.n_neighbors = n_neighbors

    def _init_similarities(self):
        self.model[ITEM_SIMILARITIES_KEY] = SymmetricMatrix(
            len(self.items), value=lambda: 0)
        for item in self.items:
            for another_item in range(item + 1):
                self._init_similarity(item, another_item)

    def _init_similarity(self, item, another_item):
        self.model[
                    ITEM_SIMILARITIES_KEY][
                        (item, another_item)] = cosine_similarity(
                        self.model[ITEM_INTERSECTION_KEY][
                            (item, another_item)],
                        sqrt(self.model[ITEMS_L1_NORMS_KEY][item]),
                        sqrt(self.model[ITEMS_L1_NORMS_KEY][another_item])
                    )

    def _init_neighborhood(self):
        self.model[ITEM_NEIGHBORHOOD_KEY] = defaultdict(list)
        for item in self.items:
            candidates = self.items.difference({item})
            ordered = sorted(
                candidates,
                key=lambda element: self.model[ITEM_SIMILARITIES_KEY][(
                    item, element)], reverse=True)
            self.model[ITEM_NEIGHBORHOOD_KEY][item] = ordered[
                0:self.n_neighbors]

    def _init_intersections(self):
        self.model[ITEM_INTERSECTION_KEY] = SymmetricMatrix(
            len(self.items), lambda: 0)
        for items in self.model[INVERTED_INDEX_KEY].values():
            for item in items:
                for another_item in range(item + 1):
                    self.model[
                        ITEM_INTERSECTION_KEY][(item, another_item)] += 1

    def _init_l1(self):
        self.model[ITEMS_L1_NORMS_KEY] = DynamicArray(
            [0 for _ in self.items], default_value=lambda: 0)
        for items in self.model[INVERTED_INDEX_KEY].values():
            for item in items:
                self.model[ITEMS_L1_NORMS_KEY][item] += 1

    def _init_inv_index(self):
        self.model[INVERTED_INDEX_KEY] = defaultdict(set)
        for user in self.users:
            for item in self.items:
                if self.matrix[user][item] is not None:
                    self.model[INVERTED_INDEX_KEY][user].add(item)

    def new_rating(self, rating):
        user_id, item_id = rating
        self.matrix[user_id][item_id] = 1
        self.users.add(user_id)
        self.items.add(item_id)
        self.model[INVERTED_INDEX_KEY][user_id].add(item_id)
        self.model[ITEMS_L1_NORMS_KEY][item_id] += 1
        for another_item_id in self.model[INVERTED_INDEX_KEY][user_id]:
            self.model[ITEM_INTERSECTION_KEY][(item_id, another_item_id)] += 1
        for another_item_id in self.items:
            self._init_similarity(item_id, another_item_id)
        self._init_neighborhood()

    def predict(self, user_id, item_id):
        item_neighborhood = self._neighborhood_of(item_id)
        user_ratings = [self.matrix[user_id][i] for i in item_neighborhood]
        m = mode(user_ratings)
        if m is None:
            return 0
        else:
            return m

    def recommend(self, user_id, n_rec, repeated=False):
        user_items = self.model[INVERTED_INDEX_KEY][user_id]
        candidates = {
            ident for item in user_items for ident in self._neighborhood_of(
                item)}
        if not repeated:
            candidates = candidates.difference(user_items)
        final = list(candidates)
        return sorted(
            final, key=lambda item_id: self.predict(user_id, item_id))[:-n_rec]

    def _neighborhood_of(self, item_id):
        return self.model[ITEM_NEIGHBORHOOD_KEY][item_id]
