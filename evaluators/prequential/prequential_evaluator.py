from abc import ABC


class PrequentialEvaluator(ABC):
    def __init__(self, model, window, n_ratings, n_rec):
        self.model = model
        self.window = window
        self.window_counter = 0
        self.window_data = []
        self.window_avg_error = 1
        self.n_ratings = n_ratings
        self.n_rec = n_rec

    def _calculate_window_error(self):
        self.window_avg_error = (sum(
            self.window_data) / self.window_counter)

    def _increment_counter(self):
        self.window_counter += 1

    def _check_counter(self):
        if self.window is None:
            self._calculate_window_error()
        elif self.window_counter >= self.window:
            self._calculate_window_error()
            self.window_data = []
            self.window_counter = 0
