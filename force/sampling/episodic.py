import gymnasium as gym
import numpy as np
import torch

from force.env import TorchVectorWrapper
from force.data import TransitionBuffer
from force.nn.util import torchify, numpyify, random_indices
from force.policies import Policy
from force.util import discounted_sum


def sample_episode(env, policy: Policy,
                   eval=False,
                   max_steps=None,
                   device=None,
                   recorder=None, render=False):
    T = max_steps if max_steps is not None else env.spec.max_episode_steps
    episode = TransitionBuffer(env.observation_space, env.action_space, T, device=device)
    obs, info = env.reset()

    if recorder:
        recorder.capture_frame()
    elif render:
        env.unwrapped.render()

    for t in range(T):
        action = policy.act1(obs, eval=eval)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode.append(
            observations=obs,
            actions=action,
            next_observations=next_obs,
            rewards=reward,
            terminals=terminated,
            truncateds=truncated
        )

        if recorder:
            recorder.capture_frame()
        elif render:
            env.unwrapped.render()

        if terminated or truncated:
            break
        else:
            obs = next_obs.clone()

    return episode


# Uses a vectorized environment to collect either a specific number of episodes
# or a variable number of episodes with minimum total samples
def sample_episodes_batched(env: TorchVectorWrapper, policy: Policy,
                            num_traj=None, min_samples=None,
                            eval=False, max_episode_steps=None):
    assert isinstance(env, TorchVectorWrapper)
    assert int(num_traj is None) + int(min_samples is None) == 1, \
        'Must specify num_traj or min_samples'

    T = max_episode_steps if max_episode_steps is not None else env.spec.max_episode_steps
    traj_buffer_factory = lambda: TransitionBuffer(env.observation_space, env.action_space, T)
    traj_buffers = [traj_buffer_factory() for _ in range(env.num_envs)]
    complete_episodes = []
    total_steps = 0

    obs, info = env.reset()
    while True:
        with torch.no_grad():
            actions = policy.act(obs, eval=eval)
        next_obs, rewards, terminals, truncateds, infos = env.step(actions)
        dones = terminals | truncateds

        for i in range(env.num_envs):
            if dones[i]:
                next_obs_i = torchify(infos['final_observation'][i])
            else:
                next_obs_i = next_obs[i]
            traj_buffers[i].append(
                observations=obs[i],
                actions=actions[i],
                next_observations=next_obs_i,
                rewards=rewards[i],
                terminals=terminals[i],
                truncateds=truncateds[i]
            )
            if dones[i]:
                complete_episodes.append(traj_buffers[i])
                total_steps += len(traj_buffers[i])
                if (num_traj is not None and len(complete_episodes) == num_traj) or \
                        (min_samples is not None and total_steps >= min_samples):
                    # Done!
                    return complete_episodes

                traj_buffers[i] = traj_buffer_factory()

        obs = next_obs


def evaluate_policy(env, policy: Policy, num_episodes=10, discount=1):
    if isinstance(env, gym.vector.VectorEnv):
        episodes = sample_episodes_batched(env, policy, num_episodes, eval=True)
    else:
        episodes = [sample_episode(env, policy, eval=True) for _ in range(num_episodes)]

    returns = []
    for episode in episodes:
        rewards = episode.get('rewards')
        returns.append(discounted_sum(rewards, discount))
    return torch.stack(returns)