from numpy import array
from algorithms.collaborative_filtering.\
    matrix_factorization import (
        MatrixFactorization,
        U_DECOMPOSED_KEY,
        V_DECOMPOSED_KEY)
from data_structures import DynamicArray
from random import uniform


class MatrixFactorizationImplicit(MatrixFactorization):
    def __init__(
        self, matrix=[], u=[], v=[], lf=2,
            lr=0.01, reg=0.1):
        super().__init__(matrix, u, v, lf)
        self.learning_rate = lr
        self.reg_factor = reg
        self._initial_training()

    def _initial_training(self):
        for user_id, ratings in enumerate(self.matrix):
            for item_id, value in enumerate(ratings):
                if value is not None:
                    self.new_rating((user_id, item_id))

    def _update_factors(self, user_id, item_id, error):
        u_factors = array(self.model[U_DECOMPOSED_KEY][user_id])
        v_factors = array(self.model[V_DECOMPOSED_KEY].col(item_id))

        updated_u = u_factors + self.learning_rate * (
            error * v_factors - self.reg_factor * u_factors)

        updated_v = v_factors + self.learning_rate * (
            error * u_factors - self.reg_factor * v_factors)

        self.model[U_DECOMPOSED_KEY][user_id] = DynamicArray(
            list(updated_u), default_value=lambda: uniform(0, 1))

        self.model[V_DECOMPOSED_KEY].set_col(item_id, updated_v)

    def new_rating(self, rating):
        user_id, item_id = rating
        self.items.add(item_id)
        self.matrix[user_id][item_id] = 1
        prediction = self.predict(user_id, item_id)
        error = 1 - prediction
        self._update_factors(user_id, item_id, error)

    def recommend(self, user_id, n_rec, repeated=False):
        candidates = self.items

        if not repeated:
            item_ids = {item_id for item_id, rating in enumerate(
                self.matrix[user_id]) if rating is not None}
            candidates = candidates.difference(item_ids)

        return sorted(
            candidates,
            key=lambda item_id: abs(1 - self.predict(user_id, item_id))
            )[0:n_rec]
