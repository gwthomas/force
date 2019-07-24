from collections import OrderedDict

import matplotlib.pyplot as plt

from force.constants import DEFAULT_FIGURE_SIZE


class Plot:
    def __init__(self, xlabel, ylabel, title=None, ylim=None, kind='plot'):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.ylim = ylim
        if kind not in ('plot', 'bar', 'scatter'):
            raise ValueError('Invalid kind: {}'.format(kind))
        self.kind = kind
        self.items = OrderedDict()

    def add(self, name, y, x=None, alpha=None):
        info = {'y': y}
        if x is not None:
            info['x'] = x
        if alpha is not None:
            info['alpha'] = alpha
        self.items[name] = info

    def draw(self, ax):
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        names = list(self.items.keys())
        if self.kind == 'plot':
            for name in names:
                item = self.items[name]
                y = item['y']
                x = item['x'] if 'x' in item else range(len(y))
                ax.plot(x, y)
            if len(names) > 1:
                ax.legend(names)
        elif self.kind == 'bar':
            x, height = [], []
            for i, name in enumerate(names):
                item = self.items[name]
                x.append(i+1)
                height.append(item['y'])
            ax.bar(x, height)
            ax.set_xticks(x)
            ax.set_xticklabels(names)
        elif self.kind == 'scatter':
            for name in names:
                item = self.items[name]
                x = item['x']
                y = item['y']
                alpha = item['alpha'] if 'alpha' in item else None
                ax.scatter(x, y, s=2, alpha=alpha)
            if len(names) > 1:
                ax.legend(names)


class Figure:
    def __init__(self, plot=None, title=None, size=DEFAULT_FIGURE_SIZE):
        self.title = title
        self.size = size

        if plot is None:
            self.plots = []
        elif isinstance(plot, Plot):
            self.plots = [plot]
        elif type(plot) in (list, tuple):
            self.plots = list(plot)
        else:
            raise ValueError('Argument to Figure constructor should be a Plot object or a list/tuple of Plot objects')

    def add_plot(self, plot):
        self.plots.append(plot)

    def draw(self, arrangement='horizontal', save_path=None, show=True):
        n_plots = len(self.plots)
        if n_plots == 0:
            print('No plots to draw')

        fig = plt.figure(figsize=self.size)
        if self.title is not None:
            fig.suptitle(self.title)

        if arrangement == 'horizontal':
            nrows, ncols = 1, n_plots
        elif arrangement == 'vertical':
            nrows, ncols = n_plots, 1
        else:
            nrows, ncols = arrangement
        axes = fig.subplots(nrows=nrows, ncols=ncols)
        if n_plots == 1:
            axes = [axes]
        else:
            plt.tight_layout()

        for i, plot in enumerate(self.plots):
            if min(nrows, ncols) == 1:
                ax = axes[i]
            else:
                row = i // ncols
                col = i % ncols
                ax = axes[row,col]
            plot.draw(ax)

        if save_path is not None:
            plt.savefig(path, transparent=transparent, bbox_inches="tight")
        if show:
            plt.show()
