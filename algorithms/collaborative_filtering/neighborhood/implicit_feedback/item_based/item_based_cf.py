from data_structures import SymmetricMatrix, DynamicArray
from algorithms.collaborative_filtering import CollaborativeFiltering
from collections import defaultdict
from utils import cosine_similarity
from random import shuffle


class ItemBasedImplicitCF(CollaborativeFiltering):
    """
    Description
        The implicit item based collaborative filtering class which focuses
        on calculating and incrementing similarities.
        Extends CollaborativeFiltering.
    """
    def __init__(
        self, matrix=[], intersections=[], l1=[], inv_index={},
            similarities=[]):
        """
        Description
            ItemBasedImplicitCF's constructor.

        Arguments
            :param matrix: The ratings matrix.
            :type matrix: list
            :param intersections: A matrix of item intersections.
            :type intersections: SymmetricMatrix
            :param l1: An array of items' l1 norms.
            :type l1: DynamicArray
            :param inv_index: An inverted index of users to items.
            :type inv_index: defaultdict(set)
            :param similarities: A similarity matrix.
            :type similarities: SymmetricMatrix
        """
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
        """
        Description
            A function which computes and returns a
            similarity matrix. Returns a SymmetricMatrix.
        """
        sims = SymmetricMatrix(
            len(self.items), value=lambda: 0)
        for item in self.items:
            for another_item in range(item + 1):
                sims[(item, another_item)] = self._init_similarity(
                    item, another_item)
        return sims

    def _init_similarity(self, item, another_item):
        """
        Description
            A function which computes and returns a similarity
            between a pair of items.

        Arguments
            :param item: The first item.
            :type item: int
            :param another_item: The second item.
            :type another_item: int
        """
        return cosine_similarity(
            self.intersections_between(item, another_item),
            self.l1_norm_of(item),
            self.l1_norm_of(another_item)
        )

    def _init_intersections(self):
        """
        Description
            The function which computes and returns a
            SymmetricMatrix item intersections.
        """
        intersections = SymmetricMatrix(len(self.items), lambda: 0)
        for items in self.inv_index.values():
            for item in items:
                others = set(range(item + 1)).intersection(items)
                for another_item in others:
                    intersections[(item, another_item)] += 1
        return intersections

    def _init_l1(self):
        """
        Description
            The function which computes and returns a
            DynamicArray which contains items' l1 norms.
        """
        l1_norms = DynamicArray(
            [0 for _ in self.items], default_value=lambda: 0)
        for items in self.inv_index.values():
            for item in items:
                l1_norms[item] += 1
        return l1_norms

    def _init_inv_index(self):
        """
        Description
            The function which computes and returns a
            defaultdict(set) inverted index of users to
            their rated items.
        """
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
