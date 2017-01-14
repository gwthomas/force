import os

class Experiment(object):
    def __init__(self, name, dir=None):
        self._name = name
        self._dir = dir if dir is not None else name
        self._datas = {}
        self.cfg = None

    def configure(self, *configs):
        for config in configs:
            path = os.path.join(self.dir, 'output', config)
            with open(path, 'r') as f:
                self._do_config(f)

    def _do_config(self, f):
        data = json.load(f)
        if not self.cfg:
            self.cfg = data
        else:
            pass

    def register_data(self, data):
        if data.name in self.datas:
            raise Exception('Trying to register data named {}, but a data with that name already exists'.format(data.name))
        self.datas[data.name] = data
