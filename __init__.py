def __load():
    from pathlib import Path
    import json
    path = Path.home() / '.force' / 'machine.json'
    assert path.is_file()
    return json.loads(path.read_text())

MACHINE_VARIABLES = __load()