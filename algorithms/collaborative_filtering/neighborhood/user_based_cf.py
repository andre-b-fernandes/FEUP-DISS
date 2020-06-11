from algorithms.collaborative_filtering import CollaborativeFiltering
from data_structures import SymmetricMatrix


class UserBasedCF(CollaborativeFiltering):
    """
        A class which aims to hold common logic between
        the explicit and implicit approaches regarding
        co-rated items and similarities. Extends
        CollaborativeFiltering.
    """
    def __init__(self, matrix, co_rated):
        """
        Description
            UserBasedCF's constructor.

        Arguments
            :param matrix: The ratings matrix.
            :type matrix: list
            :param co_rated: The co-rated items matrix.
            :type co_rated: SymmetricMatrix
        """
        super().__init__(matrix)
        self.co_rated = self._init_model(co_rated, self._init_co_rated)
        self.users = {u_id for u_id in range(
            len(self.matrix)) if len(self.co_rated_between(u_id, u_id)) > 0}

    def _init_similarities(self):
        """
        Description
            A function which returns the similarity matrix, which is
            a SymmetricMatrix.
        """
        sims = SymmetricMatrix(
            len(self.matrix), lambda: self.similarity_default)
        for user_id in self.users:
            for another_user_id in range(user_id + 1):
                sims[(user_id, another_user_id)] = self._init_similarity(
                    user_id, another_user_id)
        return sims

    # initializing the co rated items with the item id's
    def _init_co_rated(self):
        """
        Description
            A function which returns the co-rated items matrix,
            which is a SymmetricMatrix.
        """
        co_rated = SymmetricMatrix(
            len(self.matrix), lambda: set())
        for index, user in enumerate(self.matrix):
            for another_index, another_user in enumerate(
                    self.matrix[0:index + 1]):
                co_rated[(index, another_index)] = set([
                    user_tuple[0]
                    for user_tuple, another_user_tuple
                    in zip(enumerate(user), enumerate(another_user))
                    if (user_tuple[1] is not None and another_user_tuple[1]
                        is not None)])
        return co_rated

    # updating the co_rated matrix inside the model
    def _update_co_rated(self, user_id, item_id, comp):
        """
        Description
            A function which updates the co-rated items matrix.

        Arguments
            :param user_id: The user identifier.
            :type user_id: int
            :param item_id: The item identifier.
            :type item_id: int
            :param comp: Function which checks if a item was rated or not.
            :type comp: function
        """
        for another_user_id in self.users:
            if comp(self.matrix[another_user_id][item_id]):
                self.co_rated[(user_id, another_user_id)].add(
                    item_id)

    def co_rated_between(self, user_id, another_user_id):
        """
        Description
            A function which returns the co-rated items
            between two users.

        Arguments
            :param user_id: The first user.
            :type user_id: int
            :param another_user_id: The second user.
            :type another_user_id: int
        """
        return self.co_rated[(user_id, another_user_id)]
