from random import uniform
from algorithms.collaborative_filtering import CollaborativeFiltering
from data_structures import DynamicArray
from numpy import inner


class MatrixFactorization(CollaborativeFiltering):
    """
    Description
        Matrix factorization general class which extends
        CollaborativeFiltering.
    """
    def __init__(self, matrix, u, v, lf):
        """
        Description
            MatrixFactorization's constructor.

        Arguments
            :param matrix: The ratings matrix.
            :type matrix: list
            :param u: U matrix.
            :type u: DynamicArray
            :param v: V matrix.
            :type v: DynamicArray
            :param lf: Latent factors.
            :type lf: int
        """
        super().__init__(matrix)
        self.latent_factors = lf
        self.u, self.v = self._init_u_v(u, v)

    def _init_u_v(self, u, v):
        """
        Description
            A function which returns a tuple (u, v) containing
            and initializes the U, V matrices.

        Arguments
            :param u: U matrix.
            :type u: DynamicArray
            :param v: V matrix.
            :type v: DynamicArray
        """
        u = self._init_model(u, self._init_u)
        v = self._init_model(v, self._init_v)
        return u, v

    def _init_u(self):
        """
        Description
            A function which returns a computed U matrix.

        Arguments
            :param u: U matrix.
            :type u: DynamicArray
        """
        return DynamicArray([
            DynamicArray(
                [uniform(0, 1) for _ in range(self.latent_factors)],
                default_value=lambda: uniform(0, 1)) for _ in range(
                    len(self.matrix))], default_value=lambda: DynamicArray(
                        default_value=lambda: uniform(0, 1)
                    ))

    def _init_v(self):
        """
        Description
            A function which returns a computed V matrix.

        Arguments
            :param v: V matrix.
            :type v: DynamicArray
        """
        return DynamicArray([
            DynamicArray([uniform(0, 1) for _ in range(len(
                self.items))], default_value=lambda: uniform(
                    0, 1)) for _ in range(
                    self.latent_factors)], default_value=lambda: DynamicArray(
                        default_value=lambda: uniform(0, 1)
                    ))

    def predict(self, user_id, item_id):
        """
        Description
            A function which returns a predition of a user's rating to
            an item.

        Arguments
            :param user_id: The user identifier.
            :type user_id: int
            :param item_id: The item identifier.
            :type item_id: int
        """
        u_values = self.u[user_id]
        u_values.extend(self.latent_factors - 1)
        v_values = self.v.col(item_id)
        return inner(u_values, v_values)

    def recommend(self, user_id, n_rec, heuristic, repeated=False):
        """
        Description
            A function which returns recommendations for a user.

        Arguments
            :param user_id: The user identifier.
            :type user_id: int
            :param n_rec: The number of items to recommend.
            :type n_rec: int
            :param heuristic: A function which takes a item and computes
            a value to be used while sorting.
            :type heuristic: function
            :param repeated: Can be previously rated products be recommended.
            :type repeated: boolean
        """
        candidates = self.items

        if not repeated:
            item_ids = {item_id for item_id, rating in enumerate(
                self.matrix[user_id]) if rating is not None}
            candidates = candidates.difference(item_ids)

        return sorted(
            candidates,
            key=heuristic)[0:n_rec]
