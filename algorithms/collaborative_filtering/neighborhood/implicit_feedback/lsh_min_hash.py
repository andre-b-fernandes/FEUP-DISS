from algorithms.collaborative_filtering import CollaborativeFiltering
from numpy.random import permutation
from data_structures import DynamicArray
from collections import defaultdict


class LSHMinHash(CollaborativeFiltering):
    """
    Description
        A class which hashes elements into buckets according
        to the MinHash family. Extends CollaborativeFiltering.
    """
    def __init__(self, matrix=[], signature_matrix=[], buckets=[], n_perms=6,
                 n_bands=2):
        """
        Description
            LSHMinHash's constructor.

        Arguments
            :param matrix: A ratings matrix.
            :type matrix: list
            :param signature_matrix: The signature matrix which contains\
                elements' signatures in the columns.
            :type signature_matrix: DynamicArray
            :param buckets: The buckets where elements are hashed into.
            :type buckets: defaultdict(set)
            :param n_perms: Number of permutations for hashing.
            :type n_perms: int
            :param n_bands: Number of bands which are used for bucketing.
            :type n_bands: int
        """
        super().__init__(matrix)
        self.n_permutations = n_perms
        self.n_bands = n_bands
        self.signature_matrix = self._init_model(
            signature_matrix, self._init_signature_matrix)
        self.buckets = self._init_model(
            buckets, self._init_buckets)

    def _init_signature_matrix(self):
        """
        Description
            A function which computes and returns the signature matrix
            which is a DynamicArray.
        """
        signatures = self._calculate_signatures(self.matrix)
        return DynamicArray(signatures, lambda: DynamicArray())

    def _calculate_signatures(self, matrix):
        """
        Description
            A function which calculates signatures for a matrix.

        Arguments
            :param matrix: A ratings matrix.
            :type matrix: DynamicArray
        """
        signatures = []
        for i in range(self.n_permutations):
            permutated_matrix = DynamicArray(self._permutation(matrix))
            sign = self._generate_signature(permutated_matrix)
            signatures.append(sign)
        return signatures

    def _generate_signature(self, perm_matrix):
        """
        Description
            A function which generates and returns a signature
            for a permutated matrix. Returns a DynamicArray.

        Arguments
            :param perm_matrix: A permutated matrix.
            :type perm_matrix: DynamicArray
        """
        signature = []
        for elem in self._elements():
            vector = self.get_vector(perm_matrix, elem)
            identifier = self._min_hash(vector)
            signature.append(identifier)
        return DynamicArray(signature)

    def _min_hash(self, elem):
        """
        Description
            A function which returns the min hash for an element.

        Arguments
            :param elem: The element to calculate the min hash for.
            :type elem: DynamicArray.
        """
        return next((i for i, x in enumerate(elem) if x == 1), None)

    def _init_buckets(self):
        """
        Description
            A function which computes and returns a defaultdict(set)
            object, containing the bucketed items.
        """
        buckets = defaultdict(set)
        for i in self._elements():
            vector = self.signature_matrix.col(i)
            candidates = self._group_by_bands(vector)
            for candidate in candidates:
                buckets[candidate].add(i)
        return buckets

    def _update_buckets(self, identifier):
        """
        Description
            A function which updates the buckets.

        Arguments
            :param identifier: An element identifier.
            :type identifier: int
        """
        vector = self.get_vector(self.signature_matrix, identifier)
        bands = self._group_by_bands(vector)
        for band in bands:
            self.buckets[band].add(identifier)

    def _group_by_bands(self, vector):
        """
        Description
            A function which groups an element by the number
            of bands.

        Arguments
            :param vector: The element to be grouped.
            :type vector: DynamicArray
        """
        return [tuple(vector[c:c + self.n_bands])
                for c in range(0, len(vector), self.n_bands)]

    def _update_signature_matrix(self, identifier):
        """
        Description
            A function which updates the signature matrix on a
            column.

        Arguments
            :param identifier: The column index.
            :type identifier: int
        """
        vector = self.get_vector(self.matrix, identifier)
        sign = [self._min_hash(permutation(vector))
                for _ in range(self.n_permutations)]
        self.signature_matrix.set_col(identifier, sign)

    def new_rating(self, rating):
        """
        Description
            The function which processes a new iteration. Expects a tuple
            (user, item)

        Arguments
            :param rating: The rating tuple.
            :type rating: tuple
        """
        user_id, item_id = rating
        self.items.add(item_id)
        self.users.add(user_id)
        self.matrix[user_id][item_id] = 1
