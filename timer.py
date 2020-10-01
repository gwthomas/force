from math import floor
import time

import numpy as np


class Timer:
    def __init__(self):
        self.times = []

    def mark(self):
        self.times.append(time.time())

    def mean_elapsed_time(self):
        n_segments = len(self.times) - 1
        if n_segments < 1:
            return np.nan
        # return np.mean([self.times[i+1] - self.times[i] for i in range(n_segments)])
        return (self.times[-1] - self.times[0]) / n_segments


class IterateTimer(Timer):
    def __init__(self, iterable):
        super().__init__()
        self.iterator = iter(iterable)
        self.iterations_completed = 0
        try:
            self._len = len(iterable)
        except:
            self._len = None

    def mark_iters(self):
        self.mark()
        for item in self.iterator:
            yield item
            self.mark()
            self.iterations_completed += 1

    def project(self, format_remaining=True, date_format_str=None):
        assert self._len is not None, 'Cannot project over iterator which does not define len'
        time_remaining = self.mean_elapsed_time() * (self._len - self.iterations_completed)
        if np.isnan(time_remaining):
            time_done = None
        else:
            time_done = time.localtime(time.time() + time_remaining)
            if format_remaining is not None:
                hours = floor(time_remaining / 3600)
                minutes = floor((time_remaining - hours*3600) / 60)
                seconds = floor(time_remaining - 3600 * hours - 60 * minutes)
                time_remaining = f'{seconds} s'
                if minutes > 0:
                    time_remaining = f'{minutes} m, ' + time_remaining
                if hours > 0:
                    time_remaining = f'{hours} h, ' + time_remaining
            if date_format_str is not None:
                time_done = time.strftime(date_format_str, time_done)
        return time_remaining, time_done


if __name__ == '__main__':
    t = IterateTimer(range(10))
    remaining_format = '{} h, {} m, {} s'
    date_format = '%b %d %H:%M:%S'
    for i in t.mark_iters():
        remaining, eta = t.project(remaining_format, date_format)
        print(i, 'mean elapsed time:', t.mean_elapsed_time(), 'estimated time remaining:', remaining, 'estimated end time:', eta)
        time.sleep(1.0)
    print('Mean elapsed time over', len(t.times)-1, 'segments:', t.mean_elapsed_time(), 'actual end time:', time.strftime(date_format))