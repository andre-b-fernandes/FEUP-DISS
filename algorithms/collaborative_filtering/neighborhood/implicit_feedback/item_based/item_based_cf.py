from data_structures import SymmetricMatrix, DynamicArray
from algorithms.collaborative_filtering import CollaborativeFiltering
from collections import defaultdict
from utils import cosine_similarity
from random import shuffle


class ItemBasedImplicitCF(CollaborativeFiltering):
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[]):
        super().__init__(matrix)
        self.inv_index = self._init_model(
            inv_index, self._init_inv_index)
        self.intersections = self._init_model(
            intersections, self._init_intersections)
        self.l1_norms = self._init_model(
            l1, self._init_l1)
        self.similarities = self._init_model(
            similarities, self._init_similarities)

    def _init_similarities(self):
        sims = SymmetricMatrix(
            len(self.items), value=lambda: 0)
        for item in self.items:
            for another_item in range(item + 1):
                sims[(item, another_item)] = self._init_similarity(
                    item, another_item)
        return sims

    def _init_similarity(self, item, another_item):
        return cosine_similarity(
            self.intersections_between(item, another_item),
            self.l1_norm_of(item),
            self.l1_norm_of(another_item)
        )

    def _init_intersections(self):
        intersections = SymmetricMatrix(len(self.items), lambda: 0)
        for items in self.inv_index.values():
            for item in items:
                others = set(range(item + 1)).intersection(items)
                for another_item in others:
                    intersections[(item, another_item)] += 1
        return intersections

    def _init_l1(self):
        l1_norms = DynamicArray(
            [0 for _ in self.items], default_value=lambda: 0)
        for items in self.inv_index.values():
            for item in items:
                l1_norms[item] += 1
        return l1_norms

    def _init_inv_index(self):
        inv_index = defaultdict(set)
        for user in self.users:
            for item in self.items:
                if self.matrix[user][item] is not None:
                    inv_index[user].add(item)
        return inv_index

    def _update_intersections(self, user_id, item_id):
        for another_item_id in self.inv_index_of(user_id):
            self.intersections[(item_id, another_item_id)] += 1

    def _update_similarities(self, item_id):
        for another_item_id in self.items:
            self.similarities[(
                item_id, another_item_id)] = self._init_similarity(
                    item_id, another_item_id)

    def new_rating(self, rating):
        user_id, item_id = rating
        self.matrix[user_id][item_id] = 1
        self.users.add(user_id)
        self.items.add(item_id)
        if item_id not in self.inv_index_of(user_id):
            self.inv_index[user_id].add(item_id)
            self.l1_norms[item_id] += 1
            self._update_intersections(user_id, item_id)
        self._update_similarities(item_id)

    def recommend(self, user_id, n_rec, repeated=False):
        candidates = {
            ident for item in self.items for ident in self.neighborhood_of(
                item)}
        user_items = self.inv_index_of(user_id)
        if not repeated:
            candidates = candidates.difference(user_items)
        final = list(candidates)
        shuffle(final)
        return final[0:n_rec]

    def process_stream(self, stream):
        for rating in stream:
            self.new_rating(rating)

    def intersections_between(self, item, another_item):
        return self.intersections[(item, another_item)]

    def l1_norm_of(self, item):
        return self.l1_norms[item]

    def inv_index_of(self, user_id):
        return self.inv_index[user_id]
