from os import PathLike

from force.sampling.rollout import ez_rollout


def rollout_cmd(env_name: str, policy_path: PathLike,
                num_episodes: int,
                out_path: PathLike | None = None,
                device: str | None = None,
                seed: int | None = None,
                verbose: bool = False) -> str:
    parts = [
        f'python {__file__}',
        f'-e {env_name}',
        f'-p {policy_path}',
        f'-n {num_episodes}'
    ]
    if out_path is not None:
        parts.append(f'-o {out_path}')
    if device is not None:
        parts.append(f'-d {device}')
    if seed is not None:
        parts.append(f'-s {seed}')
    if verbose:
        parts.append('-v')
    return ' '.join(parts)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-e', '--env-name', type=str, required=True)
    parser.add_argument('-p', '--policy', type=str, required=True)
    parser.add_argument('-n', '--num-episodes', type=int, required=True)
    parser.add_argument('-o', '--out-path', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    ez_rollout(**vars(args))