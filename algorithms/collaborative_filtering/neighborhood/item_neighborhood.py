from .neighborhood import NeighborhoodCF
from copy import deepcopy
from threading import Thread


class ItemNeighborhood(NeighborhoodCF):
    def __init__(self, neighbors, n_neighbors):
        super().__init__(neighbors, n_neighbors)

    def _init_neighborhood(self):
        return super()._init_neighborhood(self.items)

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
        self.neighbors = self._init_neighborhood(self.items)

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
