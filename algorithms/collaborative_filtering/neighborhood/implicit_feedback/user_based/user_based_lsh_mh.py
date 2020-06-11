from algorithms.collaborative_filtering.neighborhood.implicit_feedback import \
    LSHMinHash
from numpy.random import permutation
from random import shuffle


class UserLSHMinHash(LSHMinHash):
    def __init__(self, matrix=[], signature_matrix=[], buckets=[], n_perms=6,
                 n_bands=2):
        super().__init__(matrix, signature_matrix, buckets, n_perms, n_bands)

    def _permutation(self, element):
        return [permutation(elem) for elem in element]

    def _elements(self):
        return self.users

    def get_vector(self, element, pos):
        return element[pos]

    def new_rating(self, rating):
        super().new_rating(rating)
        user_id, _ = rating
        self._update_signature_matrix(user_id)
        self._update_buckets(user_id)

    def recommend(self, user_id, n_rec, repeated=False):
        row = self.matrix[user_id]
        row_filtered = [
            item_id for item_id in self.items if row[item_id] == 1]
        signature = self.signature_matrix[user_id]
        bands = self._group_by_bands(signature)
        candidates = {user_id: 0 for user_id in self.users}
        rec = set()
        for band in bands:
            users = self.buckets[band]
            for user in users:
                candidates[user] += 1
            rec = rec.union(users)
        top_users = sorted(
            rec, key=lambda user_id: candidates[user_id])[-n_rec:]
        items = {
            item for user in top_users for item in self.items if self.matrix[
                user][item] == 1}
        if not repeated:
            items = items.difference(set(row_filtered))
        items = list(items)
        shuffle(items)
        return items[0:n_rec]
