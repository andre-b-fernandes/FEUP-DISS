from random import uniform
from algorithms.collaborative_filtering import CollaborativeFiltering
from data_structures import DynamicArray
from numpy import inner


class MatrixFactorization(CollaborativeFiltering):
    def __init__(self, matrix, u, v, lf):
        super().__init__(matrix)
        self.latent_factors = lf
        self.u, self.v = self._init_u_v(u, v)

    def _init_u_v(self, u, v):
        u = self._init_model(u, self._init_u)
        v = self._init_model(v, self._init_v)
        return u, v

    def _init_u(self):
        return DynamicArray([
            DynamicArray(
                [uniform(0, 1) for _ in range(self.latent_factors)],
                default_value=lambda: uniform(0, 1)) for _ in range(
                    len(self.matrix))], default_value=lambda: DynamicArray(
                        default_value=lambda: uniform(0, 1)
                    ))

    def _init_v(self):
        return DynamicArray([
            DynamicArray([uniform(0, 1) for _ in range(len(
                self.items))], default_value=lambda: uniform(
                    0, 1)) for _ in range(
                    self.latent_factors)], default_value=lambda: DynamicArray(
                        default_value=lambda: uniform(0, 1)
                    ))

    def _update_p_factors(self, user_id):
        for item_id in self.items:
            self._update_p(user_id, item_id)

    def predict(self, user_id, item_id):
        u_values = self.u[user_id]
        u_values.extend(self.latent_factors - 1)
        v_values = self.v.col(item_id)
        return inner(u_values, v_values)
