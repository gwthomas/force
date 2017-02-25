# Superclass for parameterized policies, value functions, etc.
class Container:
    def __init__(self, implementation):
        self.implementation = implementation

    def __getattr__(self, name):
        return getattr(self.implementation, name)


class Network:
    def __init__(self, output_layers, name):
        self.output_layers = output_layers
