from algorithms.collaborative_filtering\
    .matrix_factorization import MatrixFactorization
from data_structures import DynamicArray
from utils import avg


class MatrixFactorizationExplicit(MatrixFactorization):
    def __init__(self, matrix=[], u=[], v=[], lf=2, prep=[]):
        super().__init__(matrix, u, v, lf)
        self.preprocessed_matrix = self._init_model(
            prep, self._init_preprocessed_matrix)
        self._initial_training()

    def _initial_training(self):
        for user_id, ratings in enumerate(self.matrix):
            for item_id, value in enumerate(ratings):
                if value is not None:
                    self.new_rating((user_id, item_id, value))

    def _init_preprocessed_matrix(self):
        prep = DynamicArray(default_value=lambda: DynamicArray())
        u_avg, i_avg = {}, {}

        for u_id, ratings in enumerate(self.matrix):
            u_avg[u_id] = avg(ratings)

        for i_id in self.items:
            i_avg[i_id] = avg(self.matrix.col(i_id))

        for user_id, ratings in enumerate(self.matrix):
            row = [
                rating - 0.5*(
                    u_avg[user_id] + i_avg[
                        item_id])
                if rating is not None else None for item_id,
                rating in enumerate(ratings)]
            prep.append(DynamicArray(row))

        return prep

    def _update_u_factors(self, user_id):
        for lf in range(self.latent_factors):
            new_u = self._calculate_factor_u(user_id, lf)
            self.u[user_id][lf] = new_u

    def _calculate_factor_u(self, user_id, index_factor):
        ratings = self.preprocessed_matrix[user_id]

        latent_factors = list(range(self.latent_factors))
        latent_factors.remove(index_factor)

        f1 = sum([self.v[index_factor][j] * (
            rating - sum(
                [self.u[user_id][k] * self.v[k][j] for k in latent_factors]
                ))
            if rating is not None else 0 for j, rating in enumerate(
                ratings)])
        f2 = sum([self.v[index_factor][j]**2 for j in range(
                    len(ratings))])

        return f1 / f2 if f2 != 0 else 0

    def _update_v_factors(self, item_id):
        for lf in range(self.latent_factors):
            new_v = self._calculate_factor_v(item_id, lf)
            self.v[lf][item_id] = new_v

    def _calculate_factor_v(self, item_id, index_factor):
        ratings = self.preprocessed_matrix.col(item_id)

        latent_factors = list(range(self.latent_factors))
        latent_factors.remove(index_factor)

        f1 = sum([self.u[i][index_factor] * (
            rating - sum(
                [self.u[i][k] * self.v[k][item_id] for k in latent_factors]
                ))
            if rating is not None else 0 for i, rating in enumerate(
                ratings)])
        f2 = sum([self.u[i][index_factor]**2 for i in range(
                    len(ratings))])

        return f1/f2 if f2 != 0 else 0

    def new_rating(self, rating):
        user_id, item_id, value = rating
        self.items.add(item_id)
        self.matrix[user_id][item_id] = value
        user_avg = avg(self.matrix[user_id])
        item_avg = avg(self.matrix.col(item_id))
        raw_value = value - 0.5*(user_avg + item_avg)
        self.preprocessed_matrix[user_id][item_id] = raw_value
        self._update_u_factors(user_id)
        self._update_v_factors(item_id)

    def predict_prep(self, user_id, item_id):
        return super().predict(user_id, item_id)

    def predict(self, user_id, item_id):
        u_avg = avg(self.matrix[user_id])
        i_avg = avg(self.matrix.col(item_id))
        inner_prod = self.predict_prep(user_id, item_id)
        return inner_prod + 0.5*(i_avg + u_avg)

    def recommend(self, user_id, n_rec=20, repeated=False):
        candidates = self.items
        if not repeated:
            item_ids = {item_id for item_id, rating in enumerate(
                self.matrix[user_id]) if rating is not None}
            candidates = candidates.difference(item_ids)

        return sorted(
            candidates,
            key=lambda item_id: self.predict(user_id, item_id))[-n_rec:]
