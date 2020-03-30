from time import time
from evaluators.prequential.\
    prequential_evaluator import PrequentialEvaluator


class PrequentialEvaluatorImplicit(PrequentialEvaluator):

    def __init__(self, implicit_model, window=None):
        super().__init__(implicit_model, window)

    def evaluate(self, user_id, item_id):
        start = time()
        item_ids = self.model.recommend(user_id, 10)
        end = time()
        diff = end - start
        return (item_id in item_ids), diff

    def new_rating(self, rating):
        user_id, item_id = rating[0], rating[1]
        evaluation, elap_eval = self.evaluate(user_id, item_id)        
        self.window_data.append(int(evaluation))
        self._increment_counter()
        self._check_counter()
        start = time()
        self.model.new_rating(rating)
        end = time()
        elap_nr = end - start
        return self.window_avg_error, elap_eval, elap_nr
