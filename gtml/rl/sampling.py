from collections import defaultdict
import gym
import os
import torch
import torch.multiprocessing as mp

from gtml.constants import INF
from gtml.rl.gae import estimate_advantages_and_value_targets
import gtml.util as util


class Episode:
    def __init__(self, env):
        self.env = env
        self.t = 0
        self.lobservations = []
        self.lactions = []
        self.lrewards = []
        self.done = False

    def latest_observation(self):
        return self.lobservations[-1] if self.lobservations else self.observations[-1]

    # Act for a given number of steps or until the episode ends
    def run(self, policy, max_steps=INF):
        taken = 0

        if self.t == 0:
            observation = self.env.reset()
            self.lobservations.append(observation)

        while taken < max_steps:
            self.env.maybe_render()
            with torch.no_grad():
                action = policy.act1(self.latest_observation())
            if self.env.discrete_actions:
                action = int(action)
            self.lactions.append(action)
            observation, reward, done, info = self.env.step(action)
            self.lobservations.append(observation)
            self.lrewards.append(reward)
            self.done = done
            taken += 1
            self.t += 1
            if done:
                break

        self._commit()
        return taken

    def _commit(self):
        if self.lobservations is None:
            return

        self.observations = torch.stack(self.lobservations)
        self.rewards = torch.Tensor(self.lrewards)
        if self.env.discrete_actions:
            self.actions = torch.Tensor(self.lactions)
        else:
            self.actions = torch.stack(self.lactions)

        if self.done:
            self.total_reward = float(self.rewards.sum())
            self.lobservations = None
            self.lactions = None
            self.lrewards = None


def rollout(env, policy, num_episodes, horizon=INF):
    episodes = []
    for _ in range(num_episodes):
        episode = Episode(env)
        episode.run(policy, horizon)
        episodes.append(episode)
    return episodes


def actor_process_main(proc_index, env, in_q, out_q, extras):
    util.set_random_seed(proc_index, env=env)
    policy = None   # Must be sent via the in_q
    value_fn = None # Optional, but should also be sent via in_q if needed
    episode = Episode(env)
    prev_policy_params = None
    while True:
        description, value = in_q.get()  # Wait for command
        if description == 'quit':
            return
        elif description == 'policy':
            policy = value
        elif description == 'value_fn':
            value_fn = value
        elif description == 'policy_state':
            policy.net.load_state_dict(value)
        elif description == 'value_fn_state':
            value_fn.load_state_dict(value)
        elif description == 'run':
            assert policy is not None
            T = value
            steps_taken = episode.run(policy, T)
            observations = episode.observations[-(steps_taken+1):]
            actions = episode.actions[-steps_taken:]
            rewards = episode.rewards[-steps_taken:]
            data = {
                'actions': actions,
                'rewards': rewards
            }

            if value_fn is not None:
                with torch.no_grad():
                    values = torch.squeeze(value_fn(observations).data)
                data['values'] = values

            if 'gae_lambda' in extras:
                advantages, value_targets = estimate_advantages_and_value_targets(
                    rewards, episode.done, values,
                    gamma=env.discount, lam=extras['gae_lambda']
                )
                data['advantages'] = advantages
                data['value_targets'] = value_targets

            data['observations'] = observations[:-1]

            if episode.done:
                data['returns'] = episode.total_reward
                episode = Episode(env)

            out_q.put(data)
        else:
            raise ValueError('Actor thread received unknown description: {}'.format(description))


class ParallelSampler:
    def __init__(self, env, n_actors, extras):
        self.env = env
        self.n_actors = n_actors

        self.proc_objs = []
        for proc_index in range(n_actors):
            in_q = mp.SimpleQueue() #mp.Queue()
            out_q = mp.SimpleQueue() #mp.Queue()
            args = (proc_index, env, in_q, out_q, extras)
            proc = mp.Process(target=actor_process_main, args=args)
            self.proc_objs.append((proc, in_q, out_q))
        self._started = False
        self._joined = False

    def start(self):
        if self._started:
            return
        for proc, _, _ in self.proc_objs:
            proc.start()
        self._started = True

    def join(self):
        if self._joined or not self._started:
            return
        for proc, _, _ in self.proc_objs:
            proc.join()
        self._joined = True

    def send(self, description, value):
        for _, in_q, _ in self.proc_objs:
            in_q.put((description, value))

    def sample(self, max_steps=INF):
        if not self._started:
            raise RuntimeError('Must start sampler before sampling')
        all_data = defaultdict(list)
        self.send('run', max_steps)
        for _, _, out_q in self.proc_objs:
            data = out_q.get()
            for key, value in data.items():
                all_data[key].append(value)

        cat_data = {}
        for key, value in all_data.items():
            if isinstance(value[0], torch.Tensor):
                cat_data[key] = torch.cat(value)
            elif isinstance(value[0], float):
                cat_data[key] = torch.tensor(value)
        return cat_data
