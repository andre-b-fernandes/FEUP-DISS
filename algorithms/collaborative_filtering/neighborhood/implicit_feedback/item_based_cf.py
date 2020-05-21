from data_structures import SymmetricMatrix, DynamicArray
from collections import defaultdict
from utils import cosine_similarity
from random import shuffle
from copy import deepcopy
from threading import Thread


class ItemBasedImplicitCF:
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[]):
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
            self._merge_matrix(model)
        self.similarities = self._init_similarities()
        self.neighbors = self._init_neighborhood()

    def _merge_matrix(self, model):
        for user_id in self.users:
            for item_id in self.items:
                self.matrix[user_id][item_id] = model.matrix[user_id][item_id]

    def _merge_intersections(self, model):
        for item_id in self.items:
            for another_item_id in range(item_id + 1):
                self.intersections[(
                    item_id, another_item_id)] += model.intersections_between(
                        item_id, another_item_id)

    def _merge_l1_norms(self, model):
        for item_id in self.items:
            self.l1_norms[item_id] += model.l1_norm_of(item_id)

    def _merge_items(self, model):
        self.items = self.items.union(model.items)

    def _merge_users(self, model):
        self.users = self.users.union(model.users)

    def _merge_inverted_index(self, model):
        for user_id in self.users:
            self.inv_index[user_id] = self.inv_index_of(user_id).union(
                model.inv_index_of(user_id))

    def intersections_between(self, item, another_item):
        return self.intersections[(item, another_item)]

    def l1_norm_of(self, item):
        return self.l1_norms[item]

    def inv_index_of(self, user_id):
        return self.inv_index[user_id]
