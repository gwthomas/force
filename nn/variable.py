from __future__ import print_function

from collections import defaultdict
import tensorflow as tf

from gtml.defaults import FLOAT_T
from gtml.common.tf import get_sess


# Still uses tf scoping, but more flexible than tf.get_variable
class VariableManager:
    def __init__(self):
        self._defaults = defaultdict(dict)
        self._defaults[''] = {
            'dense': tf.contrib.layers.xavier_initializer(),
            'conv': tf.contrib.layers.xavier_initializer_conv2d(),
            'bias': tf.zeros_initializer()
        }
        self._variables = {}
        self._sync_group = None  # can be of any hashable type; independent of variable scopes
        self._sync_ops = {}
        self.on_found = 'reuse'
        self.on_not_found = 'new'

    def set_default(self, kind, default):
        scope = tf.get_variable_scope()
        self._defaults[scope][kind] = default

    def use_sync_group(self, group):
        self._sync_group = group
        self.on_found = 'copy'
        if group not in self._sync_ops:
            self._sync_ops[group] = []

    def get_variable(self, name, shape=None, dtype=FLOAT_T, initializer=None, kind=None):
        scope = tf.get_variable_scope().name
        if initializer is None:
            if kind is None:
                raise RuntimeError('Must provide at least one of "kind", "initializer"')
            else:
                # search all scopes, from most specific to least specific
                scope_hierarchy = scope.split('/')
                depth = len(scope_hierarchy)
                for i in range(depth + 1):  # +1 to also check root scope (i.e., '')
                    current_scope = '/'.join(scope_hierarchy[:depth-i])
                    if current_scope in self._defaults and kind in self._defaults[current_scope]:
                        initializer = self._defaults[current_scope][kind]
                        break   # don't check higher-level scopes
                if initializer is None: # didn't find it
                    raise RuntimeError('Failed to find initializer for kind {}'.format(kind))

        key = (scope, name)
        if key in self._variables:
            if self.on_found == 'reuse':
                variable = self._variables[key]
            elif self.on_found == 'throw':
                raise RuntimeError('Variable {} already exists in scope {}'.format(name, scope))
            elif self.on_found == 'copy':
                if self._sync_group is None:
                    raise RuntimeError('Using "copy" setting but no group has been set')

                # creates a non-trainable variable and an op to copy the source's value
                copy_name = '{}_{}_copy'.format(name, self._sync_group)
                variable = tf.get_variable(copy_name, shape=shape, dtype=dtype, initializer=initializer, trainable=False)
                sync_op = tf.assign(variable, self._variables[key])
                self._sync_ops[self._sync_group].append(sync_op)
            else:
                raise RuntimeError('Invalid on_found option: {}'.format(self.on_found))
        else:
            if self.on_not_found == 'new':
                variable = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer)
                self._variables[key] = variable
            elif self.on_not_found == 'throw':
                raise RuntimeError('Variable {} does not exist in scope {}'.format(name, scope))
            else:
                raise RuntimeError('Invalid on_not_found option: {}'.format(self.on_not_found))
        return variable

    def sync(self, group, sess=None):
        if group not in self._sync_ops:
            raise RuntimeError('Sync group {} not found'.format(group))
        sess = get_sess(sess)
        sess.run(self._sync_ops[group])


_default_variable_manager = VariableManager()
def get_default_variable_manager(): return _default_variable_manager
