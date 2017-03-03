import tensorflow as tf

from gtml.util.tf import get_sess, periodic_saver


# Outputs should be a list of layers
class Network:
    def __init__(self, outputs, name):
        self._outputs = {output.name: output for output in outputs}
        self._bookkeeping()
        self.name = name

    def get_outputs(self):
        return self._outputs

    def get_param_vars(self):
        return self._param_vars

    def get_params(self, sess=None):
        return get_sess(sess).run(self._param_vars)

    def add(self, output):
        self._outputs[output.name] = output
        self._bookkeeping()

    def eval(self, names, feed_dict, sess=None):
        return get_sess(sess).run({name: self._outputs[name].get_output() for name in names}, feed_dict=feed_dict)

    def _bookkeeping(self):
        # There's probably a faster way to implement this, but it shouldn't be called frequently
        self._param_vars = []
        for output in self._outputs.values():
            for param_var in output.get_all_param_vars():
                if param_var not in self._param_vars:
                    self._param_vars.append(param_var)

    def load_params(self, path, sess=None):
        tf.train.Saver(self._param_vars).restore(get_sess(sess), path)

    def load_latest_params(self, log_dir, sess=None):
        self.load_params(tf.train.latest_checkpoint(log_dir), sess=sess)

    def new_periodic_saver(self, period):
        return periodic_saver(self._param_vars, self.name, period)
