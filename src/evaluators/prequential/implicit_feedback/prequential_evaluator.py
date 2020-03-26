from src.evaluators.prequential.prequential_evaluator import PrequentialEvaluator


class PrequentialEvaluatorImplicit(PrequentialEvaluator):

    def __init__(self, implicit_model, window=None):
        super().__init__(implicit_model, window)

    def evaluate(self, user_id, item_id):
        item_ids = self.model.recommend(user_id)
        return item_id in item_ids

    def new_rating(self, rating):
        user_id, item_id = rating[0], rating[1]
        evaluation = self.evaluate(user_id, item_id)
        self.window_data.append(int(evaluation))
        self._increment_counter()
        self._check_counter()
        self.model.new_rating(rating)
        return self.window_avg_error
