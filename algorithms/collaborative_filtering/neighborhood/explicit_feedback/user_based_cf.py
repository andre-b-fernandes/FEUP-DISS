from algorithms.collaborative_filtering.neighborhood.\
    user_neighborhood import (
        NeighborhoodUserCF,
        CO_RATED_KEY,
        SIMILARITIES_KEY,
        NEIGHBORS_KEY
    )
from utils import pearson_correlation_terms, pearson_correlation, avg
from data_structures import PairVariances
from data_structures import DynamicArray


AVG_RATINGS_KEY = "avg_ratings"
COVARIANCE_KEY = "covariance"
VARIANCES_KEY = "variances"
SIM_VALUE_KEY = "similarity_value"


class UserBasedExplicitCF(NeighborhoodUserCF):
    """
        The defion of the user based collaborative filtering algorithm.
        It extends the collaborative filtering base class.
        This class is to be used for explicit feedback.

        Attributes
        ----------

        matrix : array
            The ratings matrix.
        model: array.
            The user-user similarities.


    """
    def __init__(self, matrix=[], similarities=[], avg_ratings=dict(),
                 co_rated=[], neighbors=[], n_neighbors=5):
        """
            Constructor

            Parameters
            ----------
            matrix : array
                The ratings matrix.
            similarities: array.
                A user-user matrix containing similarities.
            avg_ratings:
                A dictionary indexed by user_id with average ratings
        """
        super().__init__(matrix, co_rated, neighbors, n_neighbors)
        self.similarity_default = dict(
            {
                COVARIANCE_KEY: 0,
                VARIANCES_KEY: PairVariances(),
                SIM_VALUE_KEY: 0
            })
        self._init_model(avg_ratings, AVG_RATINGS_KEY, self._init_avg_ratings)
        self._init_model(similarities, SIMILARITIES_KEY,
                         self._init_similarities)
        self._init_model(neighbors, NEIGHBORS_KEY, self._init_neighborhood)

    # initializing the average ratings
    def _init_avg_ratings(self):
        self.model[AVG_RATINGS_KEY] = DynamicArray(default_value=0)
        for index, user in enumerate(self.matrix):
            self.model[AVG_RATINGS_KEY][index] = avg(user)

    # Assuming user in rows and items in columns
    def _init_similarity(self, user_id, another_user_id):
        user = self.matrix[user_id]
        another_user = self.matrix[another_user_id]
        sim = pearson_correlation_terms(self.model[CO_RATED_KEY][
            (user_id, another_user_id)], user, another_user,
            self.model[AVG_RATINGS_KEY][user_id],
            self.model[AVG_RATINGS_KEY][another_user_id])
        self._update_similarities(user_id, another_user_id, sim[0],
                                  sim[1], sim[2], sim[3])

    # (user_id, item_id, new_rating, new_avg_rating)
    def _unpack_values(self, rating, another_user_id):
        user_id = rating[0]
        current_item_id = rating[1]
        new_rating = rating[2]
        new_avg_rating = rating[3]
        another_user_avg_rating = self.model[AVG_RATINGS_KEY][another_user_id]
        another_user_ratings = self.matrix[another_user_id]
        co_rated = self.model[CO_RATED_KEY][(user_id, another_user_id)]
        old_avg_rating = self.model[AVG_RATINGS_KEY][user_id]
        difference_avg = new_avg_rating - old_avg_rating
        co_rated_length = len(co_rated)
        another_user_rating = another_user_ratings[current_item_id]

        return (user_id, current_item_id, new_rating, new_avg_rating,
                another_user_avg_rating, another_user_ratings, co_rated,
                old_avg_rating, difference_avg, co_rated_length,
                another_user_rating)

    def _update_similarities(self, user_id, another_user_id, cov,
                             variance_first, variance_second, result):
        self.model[SIMILARITIES_KEY][(user_id, another_user_id)] = dict()
        self.model[SIMILARITIES_KEY][
            (user_id, another_user_id)][COVARIANCE_KEY] = cov
        self.model[SIMILARITIES_KEY][
            (user_id, another_user_id)][VARIANCES_KEY] = PairVariances()
        self.model[SIMILARITIES_KEY][
            (user_id, another_user_id)][VARIANCES_KEY].set_variance(
                user_id, another_user_id, variance_first, variance_second)
        self.model[SIMILARITIES_KEY][(
            user_id, another_user_id)][SIM_VALUE_KEY] = result

    def _update_similarities_with_terms(self, user_id, another_user_id,
                                        e, f, g):
        cov = self.covariance_between(user_id, another_user_id) + e
        variance_first = self.variance(user_id, another_user_id) + f
        variance_second = self.variance(another_user_id, user_id) + g
        if variance_first < 0:
            variance_first = 0
        if variance_second < 0:
            variance_second = 0
        corr = pearson_correlation(cov, variance_first, variance_second)
        self._update_similarities(user_id, another_user_id, cov,
                                  variance_first, variance_second, corr)

    # arguments as ((user_id, item_id, new_rating, new_avg_rating ),
    #                       another_user_id)
    def _new_rating_terms(self, rating, another_user_id):
        (user_id, _current_item_id, new_rating, new_avg_rating,
         another_user_avg_rating, another_user_ratings, co_rated,
         old_avg_rating, difference_avg, co_rated_length,
         another_user_rating) = self._unpack_values(rating, another_user_id)
        # had rated
        e, f, g = 0, 0, 0
        if another_user_rating is not None:
            e = (new_rating - new_avg_rating) * (
                another_user_rating - another_user_avg_rating) - sum(
                    [difference_avg * (
                        another_user_ratings[item_id] - another_user_avg_rating
                    ) for item_id in co_rated])
            f = (new_rating - new_avg_rating)**2 + co_rated_length * (
                difference_avg**2) - 2 * sum([difference_avg * (
                    self.matrix[user_id][item_id] - old_avg_rating)
                    for item_id in co_rated])
            g = (another_user_rating - another_user_avg_rating)**2
        # had not rated
        else:
            e = - sum(
                [(difference_avg) * (
                    another_user_ratings[item_id] - another_user_avg_rating)
                    for item_id in co_rated])
            f = co_rated_length * (difference_avg)**2 - 2 * sum([
                difference_avg * (
                    self.matrix[user_id][item_id] - old_avg_rating)
                for item_id in co_rated])
        return e, f, g

    def _new_rating_avg(self, user_id, value):
        old_avg_rating = self.model[AVG_RATINGS_KEY][user_id]
        q = len(self.model[CO_RATED_KEY][(user_id, user_id)])
        new_avg_rating = (value / (q + 1)) + (old_avg_rating * q / (q + 1))
        return new_avg_rating

    # new rating incoming as (user_id, item_id, rating)
    def _new_rating(self, user_id, item_id, value):
        new_avg_rating = self._new_rating_avg(user_id, value)
        members = list(range(0, len(self.matrix)))
        members.remove(user_id)
        for another_user_id in members:
            e, f, g = self._new_rating_terms(
                (user_id, item_id, value, new_avg_rating), another_user_id)
            self._update_similarities_with_terms(
                user_id, another_user_id, e, f, g)

        self.model[AVG_RATINGS_KEY][user_id] = new_avg_rating

    # new stream incoming as (user_id, item_id, rating)
    def new_rating(self, rating):
        user_id, item_id, value = rating[0], rating[1], rating[2]
        # rating update
        if self.matrix[user_id][item_id] is not None:
            # TODO correct papagelis equations
            pass
        # new rating
        else:
            self._new_rating(user_id, item_id, value)
        self.matrix[user_id][item_id] = value
        self._update_co_rated(user_id, item_id)
        self._init_neighborhood()

    def predict(self, user_id, item_id):
        if self.matrix[user_id][item_id] is None:
            nbs = self.neighborhood_of(user_id)
            nbs_ratings = [self.matrix[u_id][item_id] for u_id in nbs]
            return avg(nbs_ratings)
        else:
            return self.matrix[user_id][item_id]

    def recommend(self, user_id, n_rec):
        item_ids = [i for i in range(0, len(self.matrix[user_id]))
                    if self.matrix[user_id][i] is None]
        nbs = self.neighborhood_of(user_id)
        nbs_predictions = {
            i: [self.predict(n, i) for n in nbs] for i in item_ids}
        predictions = {
            key: avg(nbs_predictions[key]) for key in nbs_predictions}
        return sorted(
            item_ids,
            key=lambda item_id: predictions[item_id])[:-n_rec]

    def similarity_between(self, user, another_user):
        return self.model[SIMILARITIES_KEY][
            (user, another_user)][SIM_VALUE_KEY]

    def covariance_between(self, user, another_user):
        return self.model[SIMILARITIES_KEY][
            (user, another_user)][COVARIANCE_KEY]

    def variance(self, user, another_user):
        return self.model[SIMILARITIES_KEY][
            (user, another_user)][VARIANCES_KEY].variance(user, another_user)

    def avg_rating(self, user_id):
        return self.model[AVG_RATINGS_KEY][user_id]

    def avg_ratings(self):
        return self.model[AVG_RATINGS_KEY]
