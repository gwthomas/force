import json
from os import PathLike
from pathlib import Path


def is_valid_run_dir(dir: Path):
    assert dir.is_dir()
    config_path = dir/'config.json'
    log_path = dir/'log.txt'
    status_path = dir/'status.txt'
    return config_path.is_file() and log_path.is_file() and status_path.is_file()


def get_run_dirs(root_dir: Path) -> list:
    assert root_dir.is_dir()
    if is_valid_run_dir(root_dir):
        return [root_dir]
    else:
        run_dirs = []
        for subdir in root_dir.iterdir():
            if not subdir.is_dir():
                continue
            run_dirs.extend(get_run_dirs(subdir))
        return run_dirs


def get_nested(cfgd, spec):
    assert isinstance(cfgd, dict)
    first = spec[0]
    if len(spec) == 1:
        return cfgd[first]
    else:
        return get_nested(cfgd[first], spec[1:])


def filter_check(run_dir: Path, filters: dict) -> bool:
    with open(run_dir/'config.json', 'r') as f:
        config = json.load(f)

    for key, value in filters.items():
        cfg_value = get_nested(config, key.split('.'))
        if isinstance(value, set):
            print(cfg_value, value)
            if cfg_value not in value:
                return False
        else:
            if cfg_value != value:
                return False
    return True


def filter_experiments(root_dir: PathLike, filters: dict) -> list:
    root_dir = Path(root_dir)
    run_dirs = get_run_dirs(root_dir)
    filter_func = lambda p: filter_check(p, filters)
    return list(filter(filter_func, run_dirs))


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('root_dir', type=str)
    parser.add_argument('-f', '--filter', nargs=2, action='append', default=[])
    args = parser.parse_args()

    assert len(args.filter) > 0, 'Must specify at least one filter'
    run_dirs = filter_experiments(args.root_dir, dict(args.filter))
    if len(run_dirs) == 0:
        print('No matching experiments found')
    else:
        print('The following experiments matched:')
        for run_dir in run_dirs:
            print(str(run_dir))


if __name__ == '__main__':
    main()