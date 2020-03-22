from matplotlib import pyplot as plt
from matplotlib import animation
import time


class EvaluationAnimation:
    def __init__(self, streams, evaluator):
        self.streams = streams
        self.evaluator = evaluator
        self.fig, self.ax = plt.subplots()
        self.line, = plt.plot([], [], 'r.')
        self.x, self.y = [], []
        self.animation = animation.FuncAnimation(
            self.fig, self._animate, range(len(streams)), self._init_eval,
            repeat=False
        )

    def _init_eval(self):
        self.ax.set_xlim(0, len(self.streams))
        self.ax.set_ylim(0, 1)
        self.line.set_data(self.x, self.y)
        return self.line,
    
    def _animate(self, i):
        start = time.time()
        err = self.evaluator.new_stream(self.streams[i])
        end = time.time()
        diff = end - start
        print(f"Elapsed time: {diff} seconds on stream {i}")
        self.x.append(i)
        self.y.append(err)
        self.line.set_data(self.x, self.y)
        return self.line,

    def show(self):
        plt.show()
