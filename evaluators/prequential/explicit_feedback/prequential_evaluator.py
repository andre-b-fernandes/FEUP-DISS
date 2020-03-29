from time import time
from evaluators.prequential.\
    prequential_evaluator import PrequentialEvaluator


class PrequentialEvaluatorExplicit(PrequentialEvaluator):
    def __init__(self, explicit_model, window=None, n_ratings=10):
        super().__init__(explicit_model, window)
        self.n_ratings = n_ratings

    def _calculate_window_error(self):
        self.window_avg_error = (sum(
            self.window_data) / self.window_counter)

    def evaluate(self, user_id, item_id, value):
        start = time()
        prediction = self.model.predict(user_id, item_id)
        end = time()
        error = abs(prediction - value) / self.n_ratings
        diff = end - start
        return error, diff

    def new_rating(self, rating):
        user_id, item_id, value = rating[0], rating[1], rating[2]
        evaluation, elap = self.evaluate(user_id, item_id, value)
        self.window_data.append(evaluation)
        self._increment_counter()
        self._check_counter()
        self.model.new_rating(rating)
        return self.window_avg_error, elap
