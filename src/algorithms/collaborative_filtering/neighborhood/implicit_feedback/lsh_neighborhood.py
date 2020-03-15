from src.algorithms.collaborative_filtering.model import CollaborativeFiltering
from numpy.random import permutation
from pandas import DataFrame

SIGNATURE_MATRIX_KEY = "signature_matrix"
BUCKETS_KEY = "buckets"


class LSHBased(CollaborativeFiltering):

    def __init__(self, matrix, n_perms=5, n_buckets=2):
        super().__init__(matrix)
        self.n_permutations = n_perms
        self.n_buckets = n_buckets

    def _init_signature_matrix(self):
        signatures = []
        for i in range(self.n_permutations):
            permutated_matrix = DataFrame(permutation(self.matrix))
            sign = self._generates_signature(permutated_matrix)
            signatures.append(sign)

        self.model[SIGNATURE_MATRIX_KEY] = DataFrame(signatures)

    def _generates_signature(self, perm_matrix):
        signature = []
        for col in range(len(self.matrix[0])):
            column = perm_matrix[col]  # Assuming the data structure complies.
            identifier = next(
                (i for i, x in enumerate(column) if x is not None),
                None)
            signature.append(identifier)
        return signature

    def _signature_element(self, column):
        pass

    def _init_buckets(self):
        self.model[BUCKETS_KEY] = dict()
        for i in self.model[SIGNATURE_MATRIX_KEY].columns:
            column = self.model[SIGNATURE_MATRIX_KEY][i]
            candidates = [tuple(column[c:c + self.n_buckets])
                          for c in range(0, len(column), self.n_buckets)]
            for candidate in candidates:
                self.model[BUCKETS_KEY][candidate] = i

    # Assuming users in the rows and items in the columns
    def new_stream(self, user_id, item_id):
        self.matrix[user_id][item_id] = 1

        pass

    def recommend(self, user_id):
        pass
