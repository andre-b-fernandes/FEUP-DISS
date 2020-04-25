from algorithms.collaborative_filtering.neighborhood import (
    NeighborhoodCF, NEIGHBORS_KEY, SIMILARITIES_KEY) 
from data_structures import SymmetricMatrix, DynamicArray
from collections import defaultdict
from utils import cosine_similarity
from random import shuffle
from copy import deepcopy
from threading import Thread

ITEM_INTERSECTION_KEY = "ITEM_INTERSECTIONS"
ITEMS_L1_NORMS_KEY = "ITEMS_L1_NORMS"
INVERTED_INDEX_KEY = "INVERTED_INDEX"


class ItemBasedImplicitCF(NeighborhoodCF):
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[], neighborhood=[], n_neighbors=5):
        super().__init__(matrix, neighborhood, n_neighbors)
        self._init_model(inv_index, INVERTED_INDEX_KEY, self._init_inv_index)
        self._init_model(
            intersections, ITEM_INTERSECTION_KEY, self._init_intersections)
        self._init_model(l1, ITEMS_L1_NORMS_KEY, self._init_l1)
        self._init_model(
            similarities, SIMILARITIES_KEY, self._init_similarities)
        self._init_model(
            neighborhood, NEIGHBORS_KEY, self._init_neighborhood)

    def _init_similarities(self):
        self.model[SIMILARITIES_KEY] = SymmetricMatrix(
            len(self.items), value=lambda: 0)
        for item in self.items:
            for another_item in range(item + 1):
                self._init_similarity(item, another_item)

    def _init_similarity(self, item, another_item):
        self.model[
                    SIMILARITIES_KEY][
                        (item, another_item)] = cosine_similarity(
                        self.model[ITEM_INTERSECTION_KEY][
                            (item, another_item)],
                        self.model[ITEMS_L1_NORMS_KEY][item],
                        self.model[ITEMS_L1_NORMS_KEY][another_item]
                    )

    def _init_neighborhood(self):
        super()._init_neighborhood(self.items)

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
        if item_id not in self.model[INVERTED_INDEX_KEY][user_id]:
            self.model[INVERTED_INDEX_KEY][user_id].add(item_id)
            self.model[ITEMS_L1_NORMS_KEY][item_id] += 1
            for another_item_id in self.model[INVERTED_INDEX_KEY][user_id]:
                self.model[ITEM_INTERSECTION_KEY][(
                    item_id, another_item_id)] += 1
        for another_item_id in self.items:
            self._init_similarity(item_id, another_item_id)
        self._init_neighborhood()

    def recommend(self, user_id, n_rec, repeated=False):
        candidates = {
            ident for item in self.items for ident in self.neighborhood_of(
                item)}
        user_items = self.model[INVERTED_INDEX_KEY][user_id]
        if not repeated:
            candidates = candidates.difference(user_items)
        final = list(candidates)
        shuffle(final)
        return final[0:n_rec]

    def process_stream(self, stream):
        for rating in stream:
            self.new_rating(rating)

    def parallel_process_stream(self, stream, n_cores):
        models = [deepcopy(self) for _ in range(n_cores)]
        parts = self._split_stream(stream, n_cores)
        self._update_models(models, parts)
        self._merge_models(models)

    def _split_stream(self, stream, n_cores):
        size = len(stream)
        if size < n_cores:
            raise ValueError(
                "Number of cores superior to number of elements in stream")
        int_division = int(size / n_cores)
        remainder = size % n_cores
        parts = [stream[core * int_division:(
            core + 1) * int_division] for core in range(n_cores)]
        parts += stream[(
            n_cores - 1) * int_division:n_cores * int_division + remainder]
        return parts

    def _update_models(self, models, parts):
        threads = [Thread(
            target=model.process_stream, args=(
                part, )) for part, model in zip(parts, models)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _merge_models(self, models):
        for model in models:
            self._merge_items(model)
            self._merge_users(model)
            self._merge_intersections(model)
            self._merge_l1_norms(model)
            self._merge_inverted_index(model)
        self._init_similarities()
        self._init_neighborhood()

    def _merge_intersections(self, model):
        for item_id in self.items:
            for another_item_id in range(item_id + 1):
                self.model[ITEM_INTERSECTION_KEY][(
                    item_id, another_item_id)] += model.intersections_between(
                        item_id, another_item_id)

    def _merge_l1_norms(self, model):
        for item_id in self.items:
            self.model[ITEMS_L1_NORMS_KEY][item_id] += model.l1_norm_of(
                item_id)

    def _merge_items(self, model):
        self.items = self.items.union(model.items)

    def _merge_users(self, model):
        self.users = self.users.union(model.users)

    def _merge_inverted_index(self, model):
        for user_id in self.users:
            self.model[INVERTED_INDEX_KEY][user_id] = self.inv_index_of(
                user_id).union(model.inv_index_of(user_id))

    def intersections_matrix(self):
        return self.model[ITEM_INTERSECTION_KEY]

    def intersections_between(self, item, another_item):
        return self.model[ITEM_INTERSECTION_KEY][(item, another_item)]

    def l1_norm_of(self, item):
        return self.model[ITEMS_L1_NORMS_KEY][item]

    def inv_index_of(self, user_id):
        return self.model[INVERTED_INDEX_KEY][user_id]
