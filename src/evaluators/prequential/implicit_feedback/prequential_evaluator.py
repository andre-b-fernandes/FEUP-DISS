class PrequentialEvaluator:

    def __init__(self, implicit_model, window=10):
        self.model = implicit_model
        self.evaluations = 0
        self.window = window
        self.window_counter = 0
        self.window_data = []
        self.error_data = []

    def _calculate_window_error(self):
        error = 1 - (sum(self.window_data)/len(self.window_data))
        print("Window Average Error: " + str(error))
        return error

    def _increment_counter(self):
        self.window_counter += 1
        if self.window_counter > self.window:
            self.window_counter = 0
            err = self._calculate_window_error()
            self.error_data.append(err)
            self.window_data = []

    def evaluate(self, user_id, item_id):
        item_ids = self.model.recommend(user_id)
        return item_id in item_ids

    def new_stream(self, rating):
        user_id, item_id = rating[0], rating[1]
        evaluation = self.evaluate(user_id, item_id)
        self.window_data.append(int(evaluation))
        self._increment_counter()
        self.model.new_stream(rating)
