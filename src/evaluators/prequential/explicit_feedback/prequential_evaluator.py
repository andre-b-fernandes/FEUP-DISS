from src.evaluators.prequential.prequential_evaluator import PrequentialEvaluator


class PrequentialEvaluatorExplicit(PrequentialEvaluator):
    def __init__(self, explicit_model, window=None, n_ratings=10):
        super().__init__(explicit_model, window)
        self.n_ratings = n_ratings

    def _calculate_window_error(self):
        self.window_avg_error = (sum(
            self.window_data) / self.window_counter)

    def evaluate(self, user_id, item_id, value):
        prediction = self.model.predict(user_id, item_id)
        error = abs(prediction - value) / self.n_ratings
        print(f"Predicted rating as an error of {error} for {item_id}")
        return error

    def new_rating(self, rating):
        user_id, item_id, value = rating[0], rating[1], rating[2]
        evaluation = self.evaluate(user_id, item_id, value)
        self.window_data.append(evaluation)
        self._increment_counter()
        self._check_counter()
        self.model.new_rating(rating)
        print(f"Current window error: {self.window_avg_error}")
        return self.window_avg_error
