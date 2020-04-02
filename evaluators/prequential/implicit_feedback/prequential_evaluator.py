from time import time
from evaluators.prequential.\
    prequential_evaluator import PrequentialEvaluator


class PrequentialEvaluatorImplicit(PrequentialEvaluator):

    def __init__(self, implicit_model, window=None, n_ratings=2, n_rec=20):
        super().__init__(implicit_model, window, n_ratings, n_rec)

    def evaluate(self, user_id, item_id):
        start = time()
        item_ids = self.model.recommend(user_id, self.n_rec)
        end = time()
        diff = end - start
        return (item_id in item_ids), diff

    def new_rating(self, rating):
        user_id, item_id, value = rating[0], rating[1], rating[2]
        value = int(value >= self.n_ratings/2)
        elap_eval = 0
        if value == 1:
            evaluation, elap_eval = self.evaluate(user_id, item_id)
            self.window_data.append(int(evaluation))
        self._increment_counter()
        self._check_counter()
        start = time()
        self.model.new_rating((user_id, item_id, value))
        end = time()
        elap_nr = end - start
        # print(f"Window Average Error: {self.window_avg_error}")
        # print(f"Elapsed Time on Rec: {elap_eval}")
        # print(f"Elapsed Time on NR: {elap_nr}")
        return self.window_avg_error, elap_eval, elap_nr
