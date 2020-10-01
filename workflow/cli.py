from argparse import ArgumentParser, Namespace
from copy import deepcopy
import json
from pathlib import Path



def yes_no_prompt(prompt, default=None):
    yes_str, no_str = 'y', 'n'
    if default == 'y':
        yes_str = '[y]'
    elif default == 'n':
        no_str = '[n]'
    elif default is None:
        pass
    else:
        raise ValueError('default argument to yes_no_prompt must be \'y\', \'n\', or None')

    response = input(f'{prompt} {yes_str}/{no_str} ')
    if response == 'y':
        return True
    elif response == 'n':
        return False
    elif response == '':
        if default == 'y':
            return True
        elif default == 'n':
            return False
        raise RuntimeError('No response to yes/no prompt, and no default given')
    else:
        raise RuntimeError('Invalid response to yes/no prompt (expected y, n, or nothing)')

SIMPLE_TYPES = {bool, int, float, str}

def _list_check(l):
    for item in l:
        if type(item) in SIMPLE_TYPES:
            pass
        elif isinstance(item, list):
            _list_check(item)
        else:
            raise ValueError('Lists in configs can contain only other lists or simple types')

def nested_namespace_update(ns, jsond):
    """
    Recursively copies values from given jsond (a dict) into ns (a Namespace)
    """
    for key, value in jsond.items():
        if type(value) in SIMPLE_TYPES:
            setattr(ns, key, value)
        elif isinstance(value, dict):
            if hasattr(ns, key):
                assert isinstance(getattr(ns, key), Namespace)
            else:
                setattr(ns, key, Namespace())
            nested_namespace_update(getattr(ns, key), value)
        elif isinstance(value, list):
            _list_check(value)
            setattr(ns, key, deepcopy(value))
        else:
            raise ValueError(f'Object of unexpected type: {value} ({type(value)})')

def _nested_namespace_set_recurse(ns, path, value):
    """
    Calling this with (ns, ['path', 'to', 'thing'], x) does ns.path.to.thing = x
    """
    if len(path) == 1:
        if hasattr(ns, path[0]):
            setattr(ns, path[0], value)
            return True
        else:
            return False
    else:
        return _nested_namespace_set_recurse(getattr(ns, path[0]), path[1:], value)

def nested_namespace_set(ns, path, value):
    """
    Calling this with (ns, ['path', 'to', 'thing'], x) does ns.path.to.thing = x
    """
    assert isinstance(path, list) and len(path) > 0
    if not _nested_namespace_set_recurse(ns, path, value):
        key = '.'.join(path)
        raise ValueError(f'Cannot override non-existent key {key}')

def try_parse(s):
    try:
        return eval(s)
    except:
        return s


def find_required(cfg):
    for key, value in vars(cfg).items():
        if isinstance(value, Namespace):
            maybe = check_required(value)
            if maybe is not None:
                return f'{key}.{maybe}'
        else:
            if value == 'REQUIRED':
                return key

def check_required(cfg):
    maybe = find_required(cfg)
    if maybe is not None:
        raise ValueError(f'Must specify a value for required key {maybe}')


def main(main, **kwargs):
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default=[], action='append')
    parser.add_argument('-s', '--set', default=[], action='append', nargs=2)
    args = parser.parse_args()
    cfg = Namespace()

    for cfg_path in args.config:
        with Path(cfg_path).open('r') as f:
            nested_namespace_update(cfg, json.load(f))

    for ns_path, value in args.set:
        nested_namespace_set(cfg, ns_path.split('.'), try_parse(value))

    check_required(cfg)

    main(cfg)