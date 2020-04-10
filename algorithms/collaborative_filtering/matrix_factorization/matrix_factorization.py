from random import uniform
from algorithms.collaborative_filtering import CollaborativeFiltering
from data_structures import DynamicArray
from numpy import inner

U_DECOMPOSED_KEY = "U"
V_DECOMPOSED_KEY = "V"


class MatrixFactorization(CollaborativeFiltering):
    def __init__(self, matrix, u, v, lf):
        super().__init__(matrix)
        self.latent_factors = lf
        self._init_u_v(u, v)

    def _init_u_v(self, u, v):
        self._init_model(u, U_DECOMPOSED_KEY, self._init_u)
        self._init_model(v, V_DECOMPOSED_KEY, self._init_v)

    def _init_u(self):
        self.model[U_DECOMPOSED_KEY] = DynamicArray([
            DynamicArray(
                [uniform(0, 1) for _ in range(self.latent_factors)],
                default_value=lambda: uniform(0, 1)) for _ in range(
                    len(self.matrix))], default_value=lambda: DynamicArray(
                        default_value=lambda: uniform(0, 1)
                    ))

    def _init_v(self):
        self.model[V_DECOMPOSED_KEY] = DynamicArray([
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
        u_values = self.model[U_DECOMPOSED_KEY][user_id]
        u_values.extend(self.latent_factors - 1)
        v_values = self.model[V_DECOMPOSED_KEY].col(item_id)
        return inner(u_values, v_values)

    def u(self):
        return self.model[U_DECOMPOSED_KEY]

    def v(self):
        return self.model[V_DECOMPOSED_KEY]
