class Policy(object):
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        raise NotImplementedError
