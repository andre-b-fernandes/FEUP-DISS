from abc import ABC


class PrequentialEvaluator(ABC):
    def __init__(self, model, window=None):
        self.model = model
        self.window = window
        self.window_counter = 0
        self.window_data = []
        self.window_avg_error = 1

    def _calculate_window_error(self):
        self.window_avg_error = 1 - (sum(
            self.window_data) / self.window_counter)

    def _increment_counter(self):
        self.window_counter += 1

    def _check_counter(self):
        if self.window is None:
            self._calculate_window_error()
        elif self.window_counter > self.window:
            self._calculate_window_error()
            self.window_data = []
            self.window_counter = 0
