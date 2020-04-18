from algorithms.collaborative_filtering import CollaborativeFiltering
from numpy.random import permutation
from data_structures import DynamicArray
from collections import defaultdict

SIGNATURE_MATRIX_KEY = "signature_matrix"
BUCKETS_KEY = "buckets"


class ItemLSH(CollaborativeFiltering):

    def __init__(self, matrix=[], signature_matrix=[], buckets=[], n_perms=6,
                 n_bands=2):
        super().__init__(matrix)
        self.n_permutations = n_perms
        self.n_bands = n_bands
        self._init_model(signature_matrix, SIGNATURE_MATRIX_KEY,
                         self._init_signature_matrix)
        self._init_model(buckets, BUCKETS_KEY, self._init_buckets)

    def _init_signature_matrix(self):
        signatures = self._calculate_signatures(self.matrix)
        self.model[SIGNATURE_MATRIX_KEY] = DynamicArray(
            signatures, lambda: DynamicArray())

    def _calculate_signatures(self, matrix):
        signatures = []
        for i in range(self.n_permutations):
            permutated_matrix = DynamicArray(permutation(matrix))
            sign = self._generate_signature(permutated_matrix)
            signatures.append(sign)
        return signatures

    def _generate_signature(self, perm_matrix):
        signature = []
        for col in range(len(self.items)):
            column = perm_matrix.col(col)
            identifier = self._min_hash(column)
            signature.append(identifier)
        return DynamicArray(signature)

    def _min_hash(self, column):
        return next((i for i, x in enumerate(column) if x == 1), None)

    def _init_buckets(self):
        self.model[BUCKETS_KEY] = defaultdict(set)
        for i in range(len(self.items)):
            column = self.model[SIGNATURE_MATRIX_KEY].col(i)
            candidates = self._group_by_bands(column)
            for candidate in candidates:
                self._add_to_bucket(candidate, i)

    def _update_buckets(self, item_id):
        column = self.model[SIGNATURE_MATRIX_KEY].col(item_id)
        bands = self._group_by_bands(column)
        for band in bands:
            self._add_to_bucket(band, item_id)

    def _add_to_bucket(self, h, element):
        self.model[BUCKETS_KEY][h].add(element)

    def _group_by_bands(self, column):
        return [tuple(column[c:c + self.n_bands])
                for c in range(0, len(column), self.n_bands)]

    def _update_signature_matrix(self, item_id):
        column = self.matrix.col(item_id)
        sign = [self._min_hash(permutation(column))
                for _ in range(self.n_permutations)]
        self.model[SIGNATURE_MATRIX_KEY].set_col(item_id, sign)

    def new_rating(self, rating):
        user_id, item_id = rating
        self.items.add(item_id)
        self.users.add(user_id)
        self.matrix[user_id][item_id] = 1
        self._update_signature_matrix(item_id)
        self._update_buckets(item_id)

    def recommend(self, user_id, n_rec, repeated=False):
        row = self.matrix[user_id]
        row_filtered = [
            item_id for item_id in self.items if row[item_id] == 1]
        signatures = [self.model[SIGNATURE_MATRIX_KEY].col(item_id)
                      for item_id in row_filtered]
        rec = set()
        candidates = {item_id: 0 for item_id in self.items}
        for sign in signatures:
            bands = self._group_by_bands(sign)
            for band in bands:
                items = self.model[BUCKETS_KEY][band]
                for item in items:
                    candidates[item] += 1
                rec = rec.union(items)
        if not repeated:
            rec = rec.difference(set(row_filtered))
        return sorted(rec, key=lambda item_id: candidates[item_id])[-n_rec:]        

    def signature_matrix(self):
        return self.model[SIGNATURE_MATRIX_KEY]

    def buckets(self):
        return self.model[BUCKETS_KEY]
