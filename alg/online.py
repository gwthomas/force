from force.policy import RandomPolicy
from force.sampling import SampleBuffer
from force.env.util import env_dims, get_max_episode_steps


def train_online(env, max_timesteps, solver, exploration_policy,
                 start_timesteps=0,
                 max_episode_steps=None,
                 max_buffer=int(1e6),
                 post_episode_callback=None):
    state_dim, action_dim = env_dims(env)
    replay_buffer = SampleBuffer(state_dim, action_dim, max_buffer)
    random_policy = RandomPolicy(env.action_space)

    state, done = env.reset(), False
    episode_num = 0
    max_episode_steps = get_max_episode_steps(env) if max_episode_steps is None else max_episode_steps
    episode_data = SampleBuffer(state_dim, action_dim, max_episode_steps)

    for t in range(max_timesteps):
        if len(replay_buffer) < start_timesteps:
            action = random_policy.act1(state)
        else:
            action = exploration_policy.act1(state)

        next_state, reward, done, info = env.step(action)
        episode_data.append(state, action, next_state, reward, done)
        replay_buffer.append(state, action, next_state, reward, done)

        state = next_state

        if len(replay_buffer) >= start_timesteps:
            solver.update(replay_buffer)

        if done or len(episode_data) == max_episode_steps:
            if post_episode_callback is not None:
                post_episode_callback(episode_num, episode_data)

            state, done = env.reset(), False
            episode_num += 1
            episode_data = SampleBuffer(state_dim, action_dim, max_episode_steps)