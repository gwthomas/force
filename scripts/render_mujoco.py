#!/usr/bin/env python

from pathlib import Path
import time

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from force.alg import SAC, TD3
from force.checkpoint import Checkpointer, CheckpointableData
from force.util import set_seed
from force.env.util import get_env, env_dims
from force import MACHINE_VARIABLES


def render_rollout(env, policy, max_episode_steps, path):
    record = path is not None
    if record:
        rec = VideoRecorder(env, path=path)
    state = env.reset()
    if record:
        rec.capture_frame()
    else:
        env.unwrapped.render()
    ret, t, done = 0., 0, False
    while not done and t < max_episode_steps:
        action = policy.act1(state, eval=True)
        state, reward, done, info = env.step(action)
        if record:
            rec.capture_frame()
        else:
            env.unwrapped.render()
        time.sleep(0.01)
        ret += reward
        t += 1
    if record:
        rec.close()
    return ret, t

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--max-episode-steps', default=1000, type=int)
    parser.add_argument('--alg', default='SAC')
    parser.add_argument('--expert', type=int, default=None)
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--filename', default=None)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--reps', default=1, type=int)
    parser.add_argument('--video-path', default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    env = get_env(args.env, seed=args.seed)

    state_dim, action_dim = env_dims(env)
    max_action = int(env.action_space.high[0])

    if args.alg == 'SAC':
        policy = SAC(state_dim, action_dim, max_action)
    elif args.alg == 'TD3':
        policy = TD3(state_dim, action_dim, max_action)
    else:
        raise NotImplementedError

    if args.expert is None:
        assert args.dir is not None
        subdir = args.dir
        filename = args.filename
        checkpointables = {'solver': policy, 'data': CheckpointableData()}
    else:
        subdir = f'expert_{args.env}-{args.max_episode_steps}_{args.alg}_{args.seed}'
        filename = f'solver@{args.expert}.pth'
        checkpointables = policy

    dir = Path(MACHINE_VARIABLES['root-dir'])/'logs'/subdir
    checkpointer = Checkpointer(checkpointables, dir, filename)
    checkpointer.load()

    for _ in range(args.reps):
        ret, episode_length = render_rollout(env, policy, args.max_episode_steps, args.video_path)
        print(f'Episode terminated with return {ret} after {episode_length} steps')