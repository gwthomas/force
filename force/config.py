from copy import deepcopy
import inspect


JSON_TYPES = {bool, int, float, str, dict, list}


class Field:
    """Represents a configurable field.
    A field has an associated type and may be required or have a default value if not specified.
    Optionally, an extra check may be specified (e.g. to impose numerical bounds)
    """

    def __init__(self, dtype_or_default, required=None, check=None):
        if isinstance(dtype_or_default, type):
            # Type specified, no default
            self.dtype = dtype_or_default
            self.default = None
            if required is None:
                required = True
        else:
            # Default value specified, type inferred
            self.default = dtype_or_default
            self.dtype = type(dtype_or_default)
            if required is None:
                required = False

        self.value = None
        self.required = required
        self.extra_check = check

    def check(self, value):
        assert type(value) == self.dtype
        if self.extra_check is not None:
            assert self.extra_check(value)

    def set(self, value):
        self.check(value)
        self.value = value

    def __repr__(self):
        s = f'Field({self.dtype.__name__}, default = {self.default}, value = {self.value}'
        return s + ')'


class Choice(Field):
    """Field that allows selection from a finite set of options.
    A default may be specified. If not, a value is required.
    """
    def __init__(self, options, default=None):
        options = set(options)

        # All options must have same type
        dtype = type(next(iter(options)))
        for option in options:
            assert type(option) == dtype

        if default is None:
            # No default, so a value must be specified.
            # Type inferred from options.
            super().__init__(dtype)
        else:
            assert default in options
            super().__init__(default)

        self.options = options

    def check(self, value):
        super().check(value)
        assert value in self.options

    def __repr__(self):
        s = ', '.join(self.options)
        if self.default is not None:
            s = s + f', default = {self.default}'
        return f'Choice({s})'


TAG_KEY = '_tag'
class TaggedUnion:
    """
    Tagged union allows different choices with their associated configs
    """
    def __init__(self, possible_configs):
        self.possible_configs = dict(possible_configs)
        for cfg in self.possible_configs.values():
            assert isinstance(cfg, BaseConfig)

    def parse(self, d):
        assert isinstance(d, dict)
        assert TAG_KEY in d, f'TaggedUnion expects key {TAG_KEY}'
        tag = d[TAG_KEY]
        cfg = deepcopy(self.possible_configs[tag])
        cfg.update(d)
        return cfg


class BaseConfig:
    """Base class for all configs"""

    def __init__(self):
        # Create an instance-specific copy of all class variables
        for key, val in inspect.getmembers(self.__class__):
            # Skip methods and private identifiers
            if inspect.isroutine(val) or key.startswith('_'):
                continue
            setattr(self, key, deepcopy(val))

    def vars(self):
        vars = {}
        for key in dir(self):
            val = getattr(self, key)
            # Skip methods and private identifiers
            if inspect.ismethod(val) or key.startswith('_'):
                continue
            vars[key] = val
        return vars

    def vars_recursive(self):
        vars = self.vars()
        for key in vars:
            if isinstance(vars[key], BaseConfig):
                vars[key] = vars[key].vars_recursive()
        return vars

    def set(self, key, value):
        assert type(value) in JSON_TYPES
        path_list = key.split('.')
        result, info = self._set_helper(path_list, value)
        if result == 'not found':
            raise ValueError(f'Cannot override non-existent key {key}')
        elif result == 'wrong type':
            raise ValueError(f'Got wrong type for key {key}: expected {info} but got {type(value)}')
        elif result == 'check failed':
            raise ValueError(f'Check failed for key {key}')
        else:
            assert result == 'success'

    def _set_helper(self, path_list, value):
        path0 = path_list[0]
        if len(path_list) == 1:
            cfg_val = getattr(self, path0)
            if type(cfg_val) in JSON_TYPES:
                expected_type = type(cfg_val)
                if type(value) == expected_type:
                    setattr(self, path0, value)
                    return 'success', None
                else:
                    return 'wrong type', expected_type
            elif isinstance(cfg_val, Field):
                try:
                    cfg_val.set(value)
                except AssertionError:
                    return 'check failed', None
                return 'success', None
            else:
                raise ValueError(f'Invalid Config: should be JSON type or Field but got {cfg_val}')
        elif hasattr(self, path0):
            subconfig = getattr(self, path0)
            assert isinstance(subconfig, BaseConfig)
            return subconfig._set_helper(path_list[1:], value)
        else:
            return False

    def update(self, d):
        for key, val in d.items():
            if key == TAG_KEY:
                # TAG_KEY should be left in config dict (for later reference and possible loading),
                # but it shouldn't directly affect values in the config, so we just skip it
                continue

            assert hasattr(self, key), f'Cannot set non-existent key {key} in {self}'
            cfg_val = getattr(self, key)
            if inspect.isclass(cfg_val):
                assert issubclass(cfg_val, Configurable)
                assert isinstance(val, dict)
                cfg = cfg_val.Config()
                cfg.update(val)
                setattr(self, key, cfg)
            elif isinstance(cfg_val, TaggedUnion):
                assert isinstance(val, dict)
                setattr(self, key, cfg_val.parse(val))
            else:
                self.set(key, val)

    def resolve(self):
        to_set = {}
        for key, val in self.vars().items():
            if inspect.isclass(val):
                assert issubclass(val, Configurable)
                cfg = val.Config()
                cfg.resolve()
                to_set[key] = cfg
            elif isinstance(val, BaseConfig):
                val.resolve()
            elif isinstance(val, Field):
                if val.value is not None:
                    to_set[key] = val.value
                elif val.required:
                    raise ValueError(f'Required value not specified for key {key}')
                else:
                    to_set[key] = val.default
        for k, v in to_set.items():
            setattr(self, k, v)

    def __repr__(self):
        cls_name = self.__class__.__name__
        args = ', '.join(f'{key}={val}' for key, val in vars(self).items())
        return f'{cls_name}({args})'


class Configurable:
    """
    Base class for all objects that expose hyperparameters to the configuration system.
    All subclasses must define a nested class called Config, which should be a descendant of BaseConfig.
    """

    Config = BaseConfig

    def __init__(self, cfg):
        assert type(cfg) is self.__class__.Config, f'Wrong config type for {self}'
        self.cfg = deepcopy(cfg)