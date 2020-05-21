from matplotlib import pyplot as plt
from matplotlib import animation


class EvaluationAnimation:
    def __init__(self, stream, evaluator):
        self.stream = stream
        self.evaluator = evaluator
        self.fig, self.ax = plt.subplots()
        self.line, = plt.plot([], [], 'r.', label="Error")
        self.time_l_rec, = plt.plot([], [], 'g.', label="Elapsed Rec Time")
        self.time_l_nr, = plt.plot([], [], 'b.', label="Elapsed Nr Time")
        plt.legend(
            [self.line, self.time_l_rec, self.time_l_nr],
            ['Error', 'Elapsed Rec Time', "Elapsed NR time"])
        self.x, self.y = [], []
        self.time_rec, self.time_nr = [], []
        self.animation = animation.FuncAnimation(
            self.fig, self._animate, range(len(stream)), self._init_eval,
            repeat=False
        )

    def _init_eval(self):
        self.ax.set_xlim(0, len(self.stream))
        self.ax.set_ylim(0, 1)
        self.line.set_data(self.x, self.y)
        self.time_l_rec.set_data(self.x, self.time_rec)
        self.time_l_nr.set_data(self.x, self.time_nr)
        return self.line, self.time_l_rec, self.time_l_nr

    def _animate(self, i):
        # print(f"It no {i} -> {self.stream[i]}")
        err, elap, elap_nr = self.evaluator.new_rating(self.stream[i])
        self.x.append(i)
        self.y.append(err)
        self.time_rec.append(elap)
        self.time_nr.append(elap_nr)
        self.line.set_data(self.x, self.y)
        self.time_l_rec.set_data(self.x, self.time_rec)
        self.time_l_nr.set_data(self.x, self.time_nr)
        return self.line, self.time_l_rec, self.time_l_nr

    def show(self):
        plt.show()
