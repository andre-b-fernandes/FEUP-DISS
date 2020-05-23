from time import time
from evaluators.prequential.\
    prequential_evaluator import PrequentialEvaluator


class PrequentialEvaluatorImplicit(PrequentialEvaluator):

    def __init__(self, implicit_model, window=None, n_ratings=2, n_rec=20):
        super().__init__(implicit_model, window, n_ratings, n_rec)

    def evaluate(self, user_id, item_id):
        start = time()
        item_ids = self.model.recommend(user_id, self.n_rec)
        # print(f"Is {item_id} in {item_ids} ?")
        end = time()
        diff = end - start
        return (item_id not in item_ids), diff

    def new_rating(self, rating):
        user_id, item_id = rating
        evaluation, elap_eval = self.evaluate(user_id, item_id)
        self.window_data.append(int(evaluation))
        self._increment_counter()
        self._check_counter()
        start = time()
        self.model.new_rating((user_id, item_id))
        end = time()
        elap_nr = end - start
        # print(f"Elapsed Recommendation Time: {elap_eval}")
        # print(f"Elapsed New Rating Time: {elap_nr}")
        # print(f"Average Window Error: {self.window_avg_error}")
        return self.window_avg_error, elap_eval, elap_nr
