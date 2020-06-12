from utils import avg, increment_avg
from data_structures import DynamicArray
from collections import defaultdict
from algorithms.collaborative_filtering.\
    matrix_factorization import MatrixFactorization


class PreprocessMatrix(MatrixFactorization):
    """
    Description
        A matrix preprocessing class used for matrix
        factorization which extends MatrixFactorization.
    """
    def __init__(
            self, matrix=[], u=[], v=[],
            lf=2, user_avg=[], item_avg=[], prep=[]):
        """
        Description
            PreprocessMatrix's constructor.

        Arguments
            :param matrix: The ratings matrix.
            :type matrix: list
            :param u: U matrix.
            :type u: DynamicArray
            :param v: V matrix.
            :type v: DynamicArray
            :param lf: Learning factor.
            :type lf: int
            :param user_avg: The average ratings of users.
            :type user_avg: defaultdict(int)
            :param item_avg: The average ratings of items.
            :type item_avg: defaultdict(int)
            :param prep: The preprocessed matrix.
            :type prep: DynamicArray
        """
        super().__init__(matrix, u, v, lf)
        self.user_avg = self._init_model(user_avg, self._init_user_avg)
        self.item_avg = self._init_model(item_avg, self._init_item_avg)
        self.preprocessed_matrix = self._init_model(
            prep, self._init_preprocessed_matrix)

    def _init_user_avg(self):
        """
        Description
            A function which returns the users' average ratings as
            a defaultdict.
        """
        user_avg = defaultdict(int)
        for user in self.users:
            user_avg[user] = avg(self.matrix[user])
        return user_avg

    def _init_item_avg(self):
        """
        Description
            A function which returns the items' average ratings as
            a defaultdict.
        """
        item_avg = defaultdict(int)
        for item in self.items:
            item_avg[item] = avg(self.matrix.col(item))
        return item_avg

    def _initial_training(self):
        """
        Description
            A function which updates the U, V matrices with
            the ratings matrix.
        """
        for user_id, ratings in enumerate(self.matrix):
            for item_id, value in enumerate(ratings):
                if value is not None:
                    raw_value = self.preprocessed_matrix[user_id][item_id]
                    error = raw_value - self.predict_prep(user_id, item_id)
                    self._update_factors(user_id, item_id, error)

    def _init_preprocessed_matrix(self):
        """
        Description
            A function which computes and returns a preprocessed matrix as a
            DynamicArray.
        """
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
        """
        Description
            A function which increments a user and item's average
            rating.

        Arguments:
            :param user: The user identifier.
            :type user: int
            :param item: The item identifier.
            :type item: int
            :param value: The value of the rating.
            :type value: int
        """
        self.user_avg[user] = increment_avg(
            self.user_avg[user], value, self.matrix[user])
        self.item_avg[item] = increment_avg(
            self.item_avg[item], value, self.matrix.col(item))

    def predict_prep(self, user_id, item_id):
        """
        Description:
            Returns a preprocessed prediction of a rating.

        Arguments:
            :param user_id: The user identifier.
            :type user_id: int
            :param item_id: The item identifier.
            :type item_id: int
        """
        return super().predict(user_id, item_id)

    def predict(self, user_id, item_id):
        """
        Description:
            Returns a postprocessed prediction of a rating.

        Arguments:
            :param user_id: The user identifier.
            :type user_id: int
            :param item_id: The item identifier.
            :type item_id: int
        """
        u_avg = self.user_avg[user_id]
        i_avg = self.item_avg[item_id]
        inner_prod = self.predict_prep(user_id, item_id)
        return inner_prod + 0.5*(i_avg + u_avg)
