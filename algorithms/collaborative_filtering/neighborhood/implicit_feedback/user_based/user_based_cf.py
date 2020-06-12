from algorithms.collaborative_filtering.neighborhood import UserBasedCF
from utils import cosine_similarity as cos_sim


class UserBasedImplicitCF(UserBasedCF):
    """
    Description
        A class which deals with similarity computation between pairs of users
        for implicit feedback. Extends UserBasedCF.
    """
    def __init__(self, matrix=[], similarities=[], co_rated=[]):
        """
        Description
            UserBasedImplicitCF's constructor.
        Arguments
            :param matrix: The ratings matrix.
            :type matrix: list
            :param similarities: The user similarity matrix.
            :type similarities: SymmetricMatrix
            :param co_rated: Co-rated items matrix.
            :type co_rated: SymmetricMatrix
        """
        super().__init__(matrix, co_rated)
        self.similarity_default = 0.0
        self.similarities = self._init_model(
            similarities, self._init_similarities)

    def _init_similarity(self, user_id, another_user_id):
        """
        Description
            A function which computes and returns the similarity
            between two users.
        Arguments
            :param user_id: The first user.
            :type user_id: int
            :param another_user_id: The second user.
            :type another_user_id: int
        """
        number_rated_items_user = len(self.co_rated_between(user_id, user_id))
        number_rated_items_another_user = len(self.co_rated_between(
            another_user_id, another_user_id))
        number_of_co_rated_items = len(self.co_rated_between(user_id,
                                                             another_user_id))
        return cos_sim(
            number_of_co_rated_items, number_rated_items_user,
            number_rated_items_another_user)

    def _update_similarities(self, user_id):
        """
        Description
            A function which updates similarities for each pair of user
            where user_id is included.
        Arguments
            :param user_id: A user identifier.
            :type user_id: int
        """
        members = self.users.difference({user_id})
        for another_user_id in members:
            self.similarities[(
                user_id, another_user_id)] = self._init_similarity(
                    user_id, another_user_id)

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
        self.users.add(user_id)
        self.items.add(item_id)
        self.matrix[user_id][item_id] = 1
        self._update_co_rated(user_id, item_id, lambda value: value == 1)
        self._update_similarities(user_id)

    def recommend(self, user_id, n_rec):
        """
        Description
            A function which returns recommendations for a user.

        Arguments
            :param user_id: The user identifier.
            :type user_id: int
            :param n_rec: The number of items to recommend.
            :type n_rec: int
        """
        item_ids = [i for i in self.items if self.matrix[
            user_id][i] is None]
        return sorted(item_ids,
                      key=lambda item_id:
                      self._activation_weight(user_id, item_id))[-n_rec:]

    def _activation_weight(self, user_id, item_id):
        """
        Description
            A function which calculates the activation weight
            of an item for a user.
        Arguments
            :param user_id: A user identifier.
            :type user_id: int
            :param item_id: An item identifier.
            :type item_id: int
        """
        nbs = self.neighborhood_of(user_id)
        len_nbs = len(nbs)
        return sum([self.similarity_between(user_id, another_user_id)
                    for another_user_id in nbs
                    if self.matrix[another_user_id][item_id]
                    is not None]) / len_nbs if len_nbs > 0 else 0
