from copy import deepcopy
import inspect


# Python types that can be parsed out of the config file
CONFIG_TYPES = (bool, int, float, str, type(None), list, dict)


class Field:
    """Represents a configurable field, typically a single hyperparameter.
    A field has an associated type and may be required or have a default value if not specified.
    """

    def __init__(self, dtype_or_default, required=None):
        if isinstance(dtype_or_default, type):
            # Type specified, no default
            self.dtype = dtype_or_default
            self.default = None
            if required is None:
                required = True
        else:
            # Default value specified, type inferred
            self.default = deepcopy(dtype_or_default)
            self.dtype = type(dtype_or_default)
            if required is None:
                required = False
            assert not required, 'Required fields should not have default values'

        self.value = None
        self.required = required

    # Can be overridden
    def check(self, value):
        assert type(value) == self.dtype,\
            f'Expected type {self.dtype.__name__}, got value {value} of type {type(value).__name__}'

    def _get(self, path_list):
        assert isinstance(self.value, BaseConfig)
        return self.value._get(path_list)

    def set(self, value):
        if issubclass(self.dtype, BaseConfig):
            assert isinstance(value, dict), 'Configs should be specified by a dict'
            value = self.dtype(**value)
        else:
            self.check(value)
        self.value = value

    def resolve(self):
        if self.required:
            assert self.value is not None, f'Required field has not been set'
        if isinstance(self.value, BaseConfig):
            self.value = self.value.resolve()
        return self.value if self.value is not None else self.default

    def __repr__(self):
        s = f'Field({self.dtype.__name__}, default = {self.default}, value = {self.value}'
        return s + ')'


class Optional(Field):
    """Convenience class that allows for an optional, i.e. non-required,
    hyperparameter. Will resolve to None if no value is provided."""
    def __init__(self, dtype: type):
        assert isinstance(dtype, type)
        super().__init__(dtype, required=False)


class Choice(Field):
    """Allows selection from a fixed set of options.
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
            assert inspect.isclass(cfg)
            assert issubclass(cfg, BaseConfig)

    def parse(self, d):
        assert isinstance(d, dict)
        assert TAG_KEY in d, f'TaggedUnion expects key {TAG_KEY}'
        tag = d.pop(TAG_KEY)
        cfg = self.possible_configs[tag]()
        cfg.update(d)
        setattr(cfg, "_tag", tag)
        return cfg


class BaseConfig:
    """Base class for all configs"""

    def __init__(self, **kwargs):
        self._field_keys = []

        # Create fields for all class variables
        for key, val in inspect.getmembers(self.__class__):
            # Skip methods and private identifiers
            if inspect.isroutine(val) or key.startswith('_'):
                continue

            if isinstance(val, Field):
                # Already a field
                field = deepcopy(val)
            elif inspect.isclass(val):
                if issubclass(val, BaseConfig):
                    # Config class; instantiate
                    field = val()
                elif val in CONFIG_TYPES:
                    # Simple type; construct a field
                    field = Field(val)
                else:
                    raise ValueError(f'Invalid Field type specifier: {val}')
            elif isinstance(val, CONFIG_TYPES):
                # Default value specified
                field = Field(val)
            elif isinstance(val, TaggedUnion):
                field = val
            else:
                raise ValueError(f'Invalid Field specifier: {val}')

            self._field_keys.append(key)
            setattr(self, key, field)

        self._resolved = False
        self.update(kwargs)

    def field_keys(self):
        return tuple(self._field_keys)

    def is_resolved(self):
        return self._resolved

    def _get(self, path_list):
        path0 = path_list[0]
        item = getattr(self, path0)
        if len(path_list) == 1:
            return item
        else:
            return item._get(path_list[1:])

    def get(self, key):
        return getattr(self, key).value

    def set(self, key, value):
        assert not self._resolved
        try:
            field = self._get(key.split('.'))
        except:
            raise ValueError(f'Key {key} not found in config')
        assert isinstance(field, Field)
        try:
            field.set(value)
        except Exception as e:
            raise RuntimeError(f'Failed to set {key}. Exception: {e}')

    def _set_polymorphic(self, key, value):
        assert not self._resolved
        field = getattr(self, key)
        if isinstance(field, Field):
            field.set(value)
        elif isinstance(field, BaseConfig):
            assert isinstance(value, dict), f'Expected a dict, got {value}'
            field.update(value)
        elif isinstance(field, TaggedUnion):
            assert isinstance(value, dict), f'Expected a dict, got {value}'
            setattr(self, key, field.parse(value))
        else:
            raise ValueError(f'Invalid field: {field}')

    def update(self, cfgd: dict):
        assert isinstance(cfgd, dict)
        for k, v in cfgd.items():
            self._set_polymorphic(k, v)

    def resolve(self):
        assert not self.is_resolved()

        for k in self._field_keys:
            field = getattr(self, k)
            try:
                resolved_value = field.resolve()
                setattr(self, k, resolved_value)
            except Exception as e:
                key_list = e.key_list if hasattr(e, 'key_list') else []
                exception = Exception(str(e))
                exception.key_list = [k] + key_list
                raise exception

        self._resolved = True
        return self

    def to_builtin_types(self):
        assert self._resolved, 'Must call resolve() before exporting config'
        fields = {}
        for k in self.field_keys():
            v = getattr(self, k)
            if isinstance(v, CONFIG_TYPES):
                fields[k] = v
            elif isinstance(v, BaseConfig):
                fields[k] = v.to_builtin_types()
            else:
                assert False
        return fields

    def to_yaml(self):
        import yaml
        return yaml.dump(self.to_builtin_types())

    def __repr__(self):
        cls_name = self.__class__.__name__
        args = ', '.join(f'{k}={getattr(self, k)}' for k in self._field_keys)
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