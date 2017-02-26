from gtml.util.tf import get_sess


# Outputs should be a list of layers
class Network:
    def __init__(self, outputs):
        self.outputs = {output.name: output for output in outputs}

    def eval(self, names, feed_dict, sess=None):
        return get_sess(sess).run({name: self.outputs[name].get_output() for name in names}, feed_dict=feed_dict)
