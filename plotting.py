from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

_context_stack = []

class Context:
    @classmethod
    def current(cls):
        for context in reversed(_context_stack):
            if isinstance(context, cls):
                return context
        return None
    
    def _on_enter(self): pass
    def _on_exit(self): pass
        
    def __enter__(self):
        _context_stack.append(self)
        self._on_enter()
        return self
        
    def __exit__(self, type, value, tb):
        assert self.__class__.current() is self
        _context_stack.pop()
        self._on_exit()
        
class PlotContext(Context):
    def __init__(self, title=None, xlabel=None, ylabel=None,
                 xlim=None, ylim=None,
                 size=(10,6), close_on_exit=True):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.size = size
        self.close_on_exit = close_on_exit

        self.reset_color_index()
        self.data = []
        
    def _on_enter(self):
        self.fig = plt.figure(figsize=self.size)
        
    def _on_exit(self):
        if self.close_on_exit:
            self.close()

    def set_color_index(self, index):
        self.color_index = index

    def reset_color_index(self):
        self.set_color_index(0)
            
    def get_color(self, advance=True):
        ret_color = 'C' + str(self.color_index)
        if advance:
            self.color_index += 1
        return ret_color
        
    def effective_xlim(self):
        if self.xlim is not None:
            return self.xlim
        else:
            xmin = min([min(x) for (x,y) in self.data if x is not None])
            xmax = max([max(x) for (x,y) in self.data if x is not None])
            return (xmin, xmax)
            
    def _do_plot(self, x, y, yerr=None, **kwargs):
        color = self.get_color()
        x, y = np.array(x), np.array(y)
        if yerr is not None:
            plt.fill_between(x, y-yerr, y+yerr,
                             color=color,
                             alpha=0.2)
        plt.plot(x, y, color=color, **kwargs)
        
    def plot(self, x, y, yerr=None, **kwargs):
        self.data.append((x,y))
        self._do_plot(x, y, yerr=yerr, **kwargs)
        
    def horizontal(self, y, yerr=None, **kwargs):
        self.data.append((None, y))
        xmin, xmax = self.effective_xlim()
        self._do_plot([xmin, xmax], [y, y], yerr=yerr, **kwargs)
            
    def finalize(self, save_path=None, show=True):
        if self.title is not None:
            plt.title(self.title)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        if self.xlim is not None:
            plt.xlim(self.xlim)
        if self.ylim is not None:
            plt.ylim(self.ylim)
        
        plt.legend(loc='lower right')
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        
    def close(self):
        plt.close(self.fig)


# def return_curves_by_seed(data: pd.DataFrame, title: str, xlabel: str):
#     variants = data.variant.unique()
#     seeds = sorted(data.seed.unique())
#     for seed in seeds:
#         if title is None:
#             plot_title = 'Seed {}'.format(seed)
#         else:
#             plot_title = '{} (seed {})'.format(title, seed)
#         with PlotContext(plot_title, xlabel=xlabel, ylabel='Return') as pc:
#             seed_data = data[data.seed == seed]
#             for var_index, variant in enumerate(variants):
#                 var_data = seed_data[seed_data.variant == variant]
#                 x = var_data.samples_taken.copy()
#                 pc.plot(x, var_data.return_mean, yerr=var_data.return_sd, label=variant)
#             pc.finalize()

def smoothed(arr, alpha):
    n = len(arr)
    ret = np.empty_like(arr)
    ret[0] = ema = arr[0]
    for i in range(1, len(arr)):
        ema = alpha * arr[i] + (1-alpha) * ema
        ret[i] = ema
    return ret

def plot_seed_averaged(data: pd.DataFrame, ykey: str, labeler=None, smooth=None, show_err=True,
                       linestyle=None):
    variants = data.variant.unique()
    seeds = sorted(data.seed.unique())
    for variant in variants:
        var_data = data[data.variant == variant]
        valid_seeds = [seed for seed in seeds if (var_data.seed == seed).sum() > 0]
        if valid_seeds != seeds:
            print('WARNING: no data for seeds', set(seeds) - set(valid_seeds))
        min_max_x = np.min([
            var_data[var_data.seed == seed].x.values.max() for seed in valid_seeds
        ])
        common_x = var_data.x[var_data.x <= min_max_x].unique()
        values = []
        for seed in valid_seeds:
            seed_data = var_data[var_data.seed == seed]
            seed_vals = []
            for x in common_x:
                # rows = seed_data[var_data.x == x]
                rows = seed_data[seed_data.x == x]
                nrows = len(rows)
                assert nrows == 1, f'Expected exactly one value, but found {nrows}, for variant {variant} seed {seed}'
                seed_vals.append(rows.iloc[0][ykey])
            values.append(np.array(seed_vals))
        value_matrix = np.column_stack(values)
        if smooth is not None:
            value_matrix = smoothed(value_matrix, smooth)
        means = np.mean(value_matrix, axis=1)
        if show_err:
            errs = stats.sem(value_matrix, axis=1)
        else:
            errs = None
        label = variant if labeler is None else labeler(variant)
        PlotContext.current().plot(common_x, means, yerr=errs, label=label, linestyle=linestyle)