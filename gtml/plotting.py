from collections import OrderedDict

import matplotlib.pyplot as plt


class Plot:
    def __init__(self, xlabel, ylabel, title=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.curves = OrderedDict()

    def add(self, name, data):
        self.curves[name] = data

    def plot(self, show=True, save_path=None):
        plt.figure()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if self.title is not None:
            plt.title(self.title)

        names = list(self.curves.keys())
        for name in names:
            data = self.curves[name]
            if callable(data):
                data = data()
            plt.plot(data, 'b-')
        if len(names) > 1:
            plt.legend(names)

        if save_path is not None:
            plt.savefig(path, transparent=transparent, bbox_inches="tight")

        if show:
            plt.plot()
