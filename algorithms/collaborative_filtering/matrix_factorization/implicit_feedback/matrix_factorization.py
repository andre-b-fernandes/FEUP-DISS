from algorithms.collaborative_filtering.\
    matrix_factorization import MatrixFactorization
from algorithms.collaborative_filtering.matrix_factorization import SGD


class MFImplicitSGD(MatrixFactorization, SGD):
    def __init__(
        self, matrix=[], u=[], v=[], lf=2,
            lr=0.01, reg=0.1):
        super().__init__(matrix, u, v, lf)
        SGD.__init__(self, lr, reg)
        self._initial_training()

    def _initial_training(self):
        for user_id, ratings in enumerate(self.matrix):
            for item_id, value in enumerate(ratings):
                if value is not None:
                    prediction = self.predict(user_id, item_id)
                    error = 1 - prediction
                    self._update_factors(user_id, item_id, error)

    def new_rating(self, rating):
        user_id, item_id = rating
        self.items.add(item_id)
        self.matrix[user_id][item_id] = 1
        prediction = self.predict(user_id, item_id)
        error = 1 - prediction
        self._update_factors(user_id, item_id, error)

    def recommend(self, user_id, n_rec, repeated=False):
        return super().recommend(user_id, n_rec, lambda item_id: abs(
            1 - self.predict(user_id, item_id)), repeated)
