from matplotlib import pyplot as plt
from matplotlib import animation


class EvaluationAnimation:
    def __init__(self, stream, evaluator):
        self.stream = stream
        self.evaluator = evaluator
        self.fig, self.ax = plt.subplots()
        self.line, = plt.plot([], [], 'r.', label="Accuracy")
        self.time_line, = plt.plot([], [], 'g.', label="Elapsed Time")
        plt.legend([self.line, self.time_line], ['Accuracy', 'Elapsed Time'])
        self.x, self.y = [], []
        self.time_y = []
        self.animation = animation.FuncAnimation(
            self.fig, self._animate, range(len(stream)), self._init_eval,
            repeat=False
        )

    def _init_eval(self):
        self.ax.set_xlim(0, len(self.stream))
        self.ax.set_ylim(0, 1)
        self.line.set_data(self.x, self.y)
        self.time_line.set_data(self.x, self.time_y)
        return self.line, self.time_line,

    def _animate(self, i):
        err, elap = self.evaluator.new_rating(self.stream[i])
        self.x.append(i)
        self.y.append(err)
        self.time_y.append(elap)
        self.line.set_data(self.x, self.y)
        self.time_line.set_data(self.x, self.time_y)
        return self.line, self.time_line,

    def show(self):
        plt.show()
