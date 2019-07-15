import json
from pathlib import Path
import sys


# name_variant should be a function mapping variants to strings
def generate_variants(base_dir):
    base_dir = Path(base_dir)
    assert base_dir.is_dir()

    sys.path.append(str(base_dir.resolve()))
    try:
        import main
    except:
        print('Failed to import main')
        raise

    variants_dir = base_dir / 'variants'
    variants_dir.mkdir(exist_ok=True)

    for variant in main.variants:
        try:
            main.config_info.parse_dict(variant)
        except:
            print('Variant was not accepted by config!')
            raise

        variant_name = main.name_variant(variant)
        variant_dir = variants_dir / variant_name
        variant_dir.mkdir(exist_ok=True)

        # Write config
        cfg_text = json.dumps(variant, indent=4)
        cfg_path = variant_dir / 'config.json'
        cfg_path.write_text(cfg_text)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('base_dir', type=str)
    args = parser.parse_args()
    generate_variants(args.base_dir)
