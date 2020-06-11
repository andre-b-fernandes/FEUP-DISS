from algorithms.collaborative_filtering.\
    matrix_factorization import MatrixFactorization
from algorithms.collaborative_filtering.matrix_factorization import SGD


class MFImplicitSGD(MatrixFactorization, SGD):
    """
    Description
        Implicit matrix factorization with stochastic
        gradient descent algorithm which extends MatrixFactorization.
    """
    def __init__(
        self, matrix=[], u=[], v=[], lf=2,
            lr=0.01, reg=0.1):
        """
        Description
            MFImplicitSGD's constructor.

        Arguments
            :param matrix: The ratings matrix.
            :type matrix: list
            :param u: U matrix.
            :type u: DynamicArray
            :param v: V matrix.
            :type v: DynamicArray
            :param lf: Learning factor.
            :type lf: int
            :param lr: The learning rate.
            :type lr: int
            :param reg: The regularization factor.
            :type reg: int
        """
        super().__init__(matrix, u, v, lf)
        SGD.__init__(self, lr, reg)
        self._initial_training()

    def _initial_training(self):
        """
        Description
            A function which updates the U, V matrices with
            the ratings matrix.
        """
        for user_id, ratings in enumerate(self.matrix):
            for item_id, value in enumerate(ratings):
                if value is not None:
                    prediction = self.predict(user_id, item_id)
                    error = 1 - prediction
                    self._update_factors(user_id, item_id, error)

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
        prediction = self.predict(user_id, item_id)
        error = 1 - prediction
        self._update_factors(user_id, item_id, error)

    def recommend(self, user_id, n_rec=20, repeated=False):
        """
        Description
            A function which returns recommendations for a user.

        Arguments:
            :param user_id: The user identifier.
            :type user_id: int
            :param n_rec: The number of items to recommend.
            :type n_rec: int
            :param repeated: Variable which defines if already rated products
            can be recommended.
            :type repeated: boolean
        """
        return super().recommend(user_id, n_rec, lambda item_id: abs(
            1 - self.predict(user_id, item_id)), repeated)
