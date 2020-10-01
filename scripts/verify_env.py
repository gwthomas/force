#!/usr/bin/env python

from itertools import count

import numpy as np

from force.dynamics import OracleDynamics
from force.policy import RandomPolicy
from force.sampling import Sampler
from force.env.util import get_env, get_max_episode_steps
import pdb

def verify(args):
    env = get_env(args.env)
    env_class = env.unwrapped.__class__
    data, _ = Sampler(env).run(RandomPolicy(env.action_space), get_max_episode_steps(env) - 1)
    oracle = OracleDynamics(env_class)

    states, actions, next_states, rewards, dones = data.get()

    if args.doneness:
        oracle_dones = env_class.done(next_states)
        if np.all(dones == oracle_dones):
            print('Correct doneness! Found', np.sum(dones), 'dones')
        else:
            print('Wrong doneness!')
            pdb.set_trace()

    if args.dynamics:
        good = True
        for t, state, action, next_state in zip(count(), states, actions, next_states):
            oracle_next_state, oracle_reward = oracle(state, action)
            if not np.allclose(next_state, oracle_next_state, atol=args.atol, rtol=args.rtol):
                print(f'Wrong dynamics at t = {t}!')
                diff = next_state - oracle_next_state
                print('Infinity norm of error:', np.abs(diff).max())
                good = False
        if good:
            print('Correct dynamics')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('--doneness', action='store_true')
    parser.add_argument('--dynamics', action='store_true')
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()
    if args.dynamics or args.doneness:
        verify(args)
    else:
        print('Nothing to verify')