from time import time
from evaluators.prequential.\
    prequential_evaluator import PrequentialEvaluator


class PrequentialEvaluatorExplicit(PrequentialEvaluator):
    """
    Description
        A class which implements a prequential evaluator
        for explicit feedback. Extends PrequentialEvaluator.
    """
    def __init__(self, explicit_model, window=None, n_ratings=5, n_rec=20):
        """
        Description
            PrequentialEvaluPrequentialEvaluatorExplicitatorImplicit's
            constructor.

        Arguments
            :param model: A recommendation algorithm object.
            :type model: CollaborativeFiltering
            :param window: The size of the evaluation window.
            :type window: int
            :param n_ratings: Number of possible ratings. E.g 2 for binary \
                feedback.
            :type n_ratings: int
            :param n_rec: Number of items to be recommended.
            :type n_rec: int
        """
        super().__init__(explicit_model, window, n_rec)
        self.n_ratings = n_ratings

    def evaluate(self, user_id, item_id, value):
        """
        Description
            A function which evaluates a prediction of a user
            to an item according to the real value.

        Arguments
            :param user_id: The user identifier.
            :type user_id: int
            :param item_id: The item identifier.
            :type item_id: int
            :param value: The actual real rating value.
            :type value: int
        """
        prediction = self.model.predict(user_id, item_id)
        # print(f"Prediction is {prediction}.")
        start = time()
        self.model.recommend(user_id, n_rec=self.n_rec)
        end = time()
        error = abs(prediction - value) / self.n_ratings
        diff = end - start
        return error, diff

    def new_rating(self, rating):
        """
        Description
            The function which processes a new iteration. Expects a tuple
            (user, item)

        Arguments
            :param rating: The rating tuple.
            :type rating: tuple
        """
        user_id, item_id, value = rating[0], rating[1], rating[2]
        evaluation, elap_pred = self.evaluate(user_id, item_id, value)
        self.window_data.append(evaluation)
        self._increment_counter()
        self._check_counter()
        start = time()
        self.model.new_rating(rating)
        end = time()
        elap_nr = end - start
        # print(f"Elapsed Recommendation Time: {elap_pred}")
        # print(f"Elapsed New Rating Time: {elap_nr}")
        # print(f"Average Window Error: {self.window_avg_error}")
        return self.window_avg_error, elap_pred, elap_nr
