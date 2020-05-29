from algorithms.collaborative_filtering\
    .matrix_factorization import MatrixFactorization
from algorithms.collaborative_filtering.matrix_factorization import SGD
from data_structures import DynamicArray
from utils import avg, increment_avg
from random import uniform
from collections import defaultdict


class MFExplicitSGD(MatrixFactorization, SGD):
    def __init__(
            self, matrix=[], u=[], v=[], user_avg=[],
            item_avg=[], lf=2, prep=[], lr=0.01, reg=0.1):
        super().__init__(matrix, u, v, lf)
        SGD.__init__(self, lambda: uniform(0, 1), lr, reg)
        self.user_avg = self._init_model(user_avg, self._init_user_avg)
        self.item_avg = self._init_model(item_avg, self._init_item_avg)
        self.preprocessed_matrix = self._init_model(
            prep, self._init_preprocessed_matrix)
        self._initial_training()

    def _init_user_avg(self):
        user_avg = defaultdict(int)
        for user in self.users:
            user_avg[user] = avg(self.matrix[user])
        return user_avg

    def _init_item_avg(self):
        item_avg = defaultdict(int)
        for item in self.items:
            item_avg[item] = avg(self.matrix.col(item))
        return item_avg

    def _initial_training(self):
        for user_id, ratings in enumerate(self.matrix):
            for item_id, value in enumerate(ratings):
                if value is not None:
                    raw_value = self.preprocessed_matrix[user_id][item_id]
                    error = raw_value - self.predict_prep(user_id, item_id)
                    self._update_factors(user_id, item_id, error)

    def _init_preprocessed_matrix(self):
        prep = DynamicArray(default_value=lambda: DynamicArray())

        for user_id, ratings in enumerate(self.matrix):
            row = [
                rating - 0.5*(
                    self.user_avg[user_id] + self.item_avg[
                        item_id])
                if rating is not None else None for item_id,
                rating in enumerate(ratings)]
            prep.append(DynamicArray(row))

        return prep

    def inc_avg(self, user, item, value):
        self.user_avg[user] = increment_avg(
            self.user_avg[user], value, self.matrix[user])
        self.user_avg[item] = increment_avg(
            self.user_avg[item], value, self.matrix.col(item))

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

    def predict_prep(self, user_id, item_id):
        return super().predict(user_id, item_id)

    def predict(self, user_id, item_id):
        u_avg = avg(self.matrix[user_id])
        i_avg = avg(self.matrix.col(item_id))
        inner_prod = self.predict_prep(user_id, item_id)
        return inner_prod + 0.5*(i_avg + u_avg)

    def recommend(self, user_id, n_rec=20, repeated=False):
        return super().recommend(user_id, n_rec, lambda item_id: self.predict(
            user_id, item_id), repeated)
