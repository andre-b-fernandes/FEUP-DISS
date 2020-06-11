from algorithms.collaborative_filtering.neighborhood.implicit_feedback import \
    LSHMinHash
from numpy.random import permutation


class ItemLSHMinHash(LSHMinHash):
    def __init__(self, matrix=[], signature_matrix=[], buckets=[], n_perms=6,
                 n_bands=2):
        super().__init__(matrix, signature_matrix, buckets, n_perms, n_bands)

    def _permutation(self, element):
        return permutation(element)

    def _elements(self):
        return self.items

    def get_vector(self, element, pos):
        return element.col(pos)

    def new_rating(self, rating):
        super().new_rating(rating)
        _, item_id = rating
        self._update_signature_matrix(item_id)
        self._update_buckets(item_id)

    def recommend(self, user_id, n_rec, repeated=False):
        row = self.matrix[user_id]
        row_filtered = [
            item_id for item_id in self.items if row[item_id] == 1]
        signatures = [self.get_vector(self.signature_matrix, item_id)
                      for item_id in row_filtered]
        rec = set()
        candidates = {item_id: 0 for item_id in self.items}
        for sign in signatures:
            bands = self._group_by_bands(sign)
            for band in bands:
                items = self.buckets[band]
                for item in items:
                    candidates[item] += 1
                rec = rec.union(items)
        if not repeated:
            rec = rec.difference(set(row_filtered))
        return sorted(rec, key=lambda item_id: candidates[item_id])[-n_rec:]
