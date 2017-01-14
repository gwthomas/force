import matplotlib.pyplot as plt

class Listener(object):
    def __init__(self, data):
        data.listeners.add(self)

    def callback(self, data):
        raise NotImplementedError


class Saver(Listener):
    def __init__(self, data, file=None):
        super(Saver, self).__init__(data)
        self._file = file if file is not None else data.name + '.npz'

    def callback(self, data):
        data.save(self.file)


def default_plot(data, fig, ax):
    ax.plot(data.get())

class Plotter(Listener):
    def __init__(self, data, plotfn=default_plot, live=False):
        super(Plotter, self).__init__(data)
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        self.plotfn = plotfn
        self.live = live
        if self.live:
            plt.ion()
        self.update(data)

    def callback(self, data):
        if self.live:
            self.update(data)
            plt.pause(0.001) # hack

    def update(self, data):
        self.plotfn(data, self._fig, self._ax)
        plt.draw()
