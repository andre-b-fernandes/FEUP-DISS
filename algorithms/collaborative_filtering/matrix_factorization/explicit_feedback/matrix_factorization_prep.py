from algorithms.collaborative_filtering.matrix_factorization import SGD
from .preprocess_matrix import PreprocessMatrix


class MFExplicitPrepSGD(PreprocessMatrix, SGD):
    def __init__(
            self, matrix=[], u=[], v=[], user_avg=[],
            item_avg=[], lf=2, prep=[], lr=0.01, reg=0.1):
        super().__init__(matrix, u, v, lf, user_avg, item_avg, prep)
        SGD.__init__(self, lr, reg)
        self._initial_training()

    def _initial_training(self):
        for user_id, ratings in enumerate(self.matrix):
            for item_id, value in enumerate(ratings):
                if value is not None:
                    raw_value = self.preprocessed_matrix[user_id][item_id]
                    error = raw_value - self.predict_prep(user_id, item_id)
                    self._update_factors(user_id, item_id, error)

    def new_rating(self, rating):
        user_id, item_id, value = rating
        self.items.add(item_id)
        self.matrix[user_id][item_id] = value
        self.inc_avg(user_id, item_id, value)
        raw_value = value - 0.5*(self.user_avg[user_id] + self.user_avg[
            item_id])
        self.preprocessed_matrix[user_id][item_id] = raw_value
        error = raw_value - self.predict_prep(user_id, item_id)
        self._update_factors(user_id, item_id, error)

    def recommend(self, user_id, n_rec=20, repeated=False):
        return super().recommend(user_id, n_rec, lambda item_id: self.predict(
            user_id, item_id), repeated)
