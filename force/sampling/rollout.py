"""This file can be used in two ways:
1) import the rollout function, or
2) call it as a script (which invokes said function).
"""

from os import PathLike
from pathlib import Path
import subprocess

import h5py
import torch
from torch.export import ExportedProgram
import torch.multiprocessing as mp

from force.env import BaseEnv as Env, GymEnv
from force.data import TransitionBuffer
from force.nn.util import get_device
from force.policies import BasePolicy as Policy, PolicyMode
from force.types import PolicyFunction
from force.util import set_seed, stats_dict, prefix_dict_keys


type PolicySpecifier = Policy | ExportedProgram | PolicyFunction | PathLike


def get_policy_fn(policy: PolicySpecifier, mode: PolicyMode | None = None) -> PolicyFunction:
    if isinstance(policy, Policy):
        return policy.functional(mode)
    
    if isinstance(policy, PathLike):
        policy = torch.export.load(policy)
            
    if isinstance(policy, ExportedProgram):
        policy = policy.module()
        
    assert callable(policy), str(policy)
    return policy


def _write_trajectories_if_path(trajectories: list[TransitionBuffer],
                                path: PathLike | None):
    if path is None:
        return

    with h5py.File(path, 'w') as f:
        for i, traj in enumerate(trajectories):
            traj.save_to_file(f, prefix=f'trajectory_{i}')


def rollout(envs: list[Env], policy: PolicySpecifier,
            policy_mode: PolicyMode | None = None,
            out_path: PathLike | None = None) -> list[TransitionBuffer]:
    """Collects a batch of episode (one episode per env)."""

    policy_fn = get_policy_fn(policy, policy_mode)

    # Buffers to store the trajectories (will be returned)
    T = envs[0].info.horizon
    trajectories = [TransitionBuffer(env.info, T) for env in envs]

    # Keep track of which envs have not terminated,
    active_indices = [i for i, env in enumerate(envs)]
    # as well as their most recent observations
    observations = [env.reset()[0] for env in envs]

    for t in range(T):
        with torch.no_grad():
            actions = policy_fn(torch.stack(observations))
        # Sanity check
        assert len(actions) == len(active_indices)
        
        next_active_indices = []
        next_observations = []

        # Loop over active envs
        for env_idx, observation, action in zip(active_indices, observations, actions):
            next_observation, reward, terminal, truncated, info = envs[env_idx].step(action)
            
            # Add transition to appropriate buffer
            trajectories[env_idx].append(
                observations=observation,
                actions=action,
                next_observations=next_observation,
                rewards=reward,
                terminals=terminal,
                truncateds=truncated
            )

            done = terminal or truncated
            if not done:
                next_active_indices.append(env_idx)
                next_observations.append(next_observation)
        
        if len(next_active_indices) == 0:
            # All episodes in the batch terminated early
            break
        
        active_indices = next_active_indices
        observations = next_observations

    _write_trajectories_if_path(trajectories, out_path)
    return trajectories


def ez_rollout(env_name: str, policy: str, num_episodes: int,
               out_path: str | None = None,
               device: str | None = None,
               seed: int | None = None,
               verbose: bool = False) -> list[TransitionBuffer]:
    """Convenience function that allows the policy to be specified by a path to
    a file containing an exported program.
    Notably, all inputs and outputs can be specified via basic built-in types,
    making this function suitable for invocation from the command line.
    """
    if seed is not None:
        set_seed(seed)
    policy = Path(policy)
    device = get_device(device)
    envs = [GymEnv(env_name, device=device) for _ in range(num_episodes)]
    trajectories = rollout(envs, policy, out_path=out_path)
    if verbose:
        returns = torch.tensor([ep.get('rewards').sum() for ep in trajectories])
        lengths = torch.tensor([len(ep) for ep in trajectories])
        stats = {
            **prefix_dict_keys('return', stats_dict(returns)),
            **prefix_dict_keys('length', stats_dict(lengths))
        }
        print(stats)
    return trajectories