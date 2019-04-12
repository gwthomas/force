from collections import defaultdict


class CallbackManager:
    def __init__(self):
        self._callbacks = defaultdict(list)

    def add_callback(self, event, callback):
        self._callbacks[event].append(callback)

    def run_callbacks(self, event, *args, **kwargs):
        for callback in self._callbacks[event]:
            callback(*args, **kwargs)


class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Periodic(Callback):
    def __init__(self, period, impl):
        self.period = period
        self.impl = impl
        self.countdown = period

    def __call__(self, *args, **kwargs):
        self.countdown -= 1
        if self.countdown == 0:
            self.impl(*args, **kwargs)
            self.countdown = self.period


class Multiple(Callback):
    def __init__(self, *callbacks):
        self.callbacks = callbacks

    def __call__(self, *args, **kwargs):
        for callback in self.callbacks:
            callback(*args, **kwargs)
