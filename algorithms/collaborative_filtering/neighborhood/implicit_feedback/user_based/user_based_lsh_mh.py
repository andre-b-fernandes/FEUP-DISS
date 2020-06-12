from algorithms.collaborative_filtering.neighborhood.implicit_feedback import \
    LSHMinHash
from numpy.random import permutation
from random import shuffle


class UserLSHMinHash(LSHMinHash):
    """
    Description
        A class which implements the user-based locality-sensitive min hashing
        algorithm which extends LSHMinHash.
    """
    def __init__(self, matrix=[], signature_matrix=[], buckets=[], n_perms=6,
                 n_bands=2):
        """
        Description
            UserLSHMinHash's constructor.

        Arguments
            :param matrix: A ratings matrix.
            :type matrix: list
            :param signature_matrix: The signature matrix which contains
                elements' signatures in the columns.
            :type signature_matrix: DynamicArray
            :param buckets: The buckets where elements are hashed into.
            :type buckets: defaultdict(set)
            :param n_perms: Number of permutations for hashing.
            :type n_perms: int
            :param n_bands: Number of bands which are used for bucketing.
            :type n_bands: int
        """
        super().__init__(matrix, signature_matrix, buckets, n_perms, n_bands)

    def _permutation(self, element):
        """
        Description
            A function which defines how to permutate a matrix.

        Arguments
            :param matrix: The matrix to be permutated.
            :type matrix: list
        """
        return [permutation(elem) for elem in element]

    def _elements(self):
        """
        Description
            A function which defines the elements to be
            hashed.
        """
        return self.users

    def get_vector(self, element, pos):
        """
        Description
            A function which returns a position of a matrix.

        Arguments
            :param matrix: The matrix to be accessed.
            :type matrix: list
            :param pos: The index/position.
            :type pos: int
        """
        return element[pos]

    def new_rating(self, rating):
        """
        Description
            The function which processes a new iteration. Expects a tuple
            (user, item)

        Arguments
            :param rating: The rating tuple.
            :type rating: tuple
        """
        super().new_rating(rating)
        user_id, _ = rating
        self._update_signature_matrix(user_id)
        self._update_buckets(user_id)

    def recommend(self, user_id, n_rec, repeated=False):
        """
        Description
            A function which returns recommendations for a user.

        Arguments
            :param user_id: The user identifier.
            :type user_id: int
            :param n_rec: The number of items to recommend.
            :type n_rec: int
            :param repeated: Variable which defines if already rated products\
                can be recommended.
            :type repeated: boolean
        """
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
