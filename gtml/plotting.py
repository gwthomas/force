from collections import OrderedDict

import matplotlib.pyplot as plt


class Plot:
    def __init__(self, xlabel, ylabel, title=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.curves = OrderedDict()

    def add(self, name, y, x=None):
        self.curves[name] = {
            'y': y,
            'x': x
        }

    def plot(self, figsize=None, ylim=None, show=True, save_path=None):
        plt.figure(figsize=figsize)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if self.title is not None:
            plt.title(self.title)
        if ylim is not None:
            plt.ylim(ylim)

        names = list(self.curves.keys())
        for name in names:
            data = self.curves[name]
            y = data['y']
            x = data['x'] if data['x'] is not None else range(len(y))
            plt.plot(x, y)
        if len(names) > 1:
            plt.legend(names)

        if save_path is not None:
            plt.savefig(path, transparent=transparent, bbox_inches="tight")

        if show:
            plt.plot()
