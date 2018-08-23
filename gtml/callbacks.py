from collections import defaultdict


class CallbackManager:
    def __init__(self):
        self._callbacks = defaultdict(list)

    def add_callback(self, event, callback):
        self._callbacks[event].append(callback)

    def run_callbacks(self, event, **kwargs):
        for callback in self._callbacks[event]:
            callback(self, **kwargs)
