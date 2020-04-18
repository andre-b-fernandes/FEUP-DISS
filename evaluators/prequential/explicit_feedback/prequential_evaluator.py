from time import time
from evaluators.prequential.\
    prequential_evaluator import PrequentialEvaluator


class PrequentialEvaluatorExplicit(PrequentialEvaluator):
    def __init__(self, explicit_model, window=None, n_ratings=5, n_rec=20):
        super().__init__(explicit_model, window, n_ratings, n_rec)

    def _calculate_window_error(self):
        self.window_avg_error = (sum(
            self.window_data) / self.window_counter)

    def evaluate(self, user_id, item_id, value):
        prediction = self.model.predict(user_id, item_id)
        print(f"Prediction {prediction}")
        start = time()
        self.model.recommend(user_id, n_rec=self.n_rec)
        end = time()
        error = abs(prediction - value) / self.n_ratings
        diff = end - start
        return error, diff

    def new_rating(self, rating):
        user_id, item_id, value = rating[0], rating[1], rating[2]
        evaluation, elap_pred = self.evaluate(user_id, item_id, value)
        self.window_data.append(evaluation)
        self._increment_counter()
        self._check_counter()
        start = time()
        self.model.new_rating(rating)
        end = time()
        elap_nr = end - start
        print(f"Elapsed Recommendation Time: {elap_pred}")
        print(f"Elapsed New Rating Time: {elap_nr}")
        print(f"Average Window Error: {self.window_avg_error}")
        return self.window_avg_error, elap_pred, elap_nr
