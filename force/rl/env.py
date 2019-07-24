import gym
import numpy as np

from force.rl.memory import Memory
from force.constants import DEFAULT_DISCOUNT


# Implements preprocessing method described in the paper by Mnih, et al.
class AtariWrapper(gym.ObservationWrapper):
    def __init__(self, env, should_render=False, m=4, size=(84,84)):
        super().__init__(env)
        self.m = m
        self.size = size
        self.raw_obs_history = Memory(history)

        obs_shape = list(size) + [m]
        self.observation_space = gym.spaces.Box(0,1, shape=obs_shape)

    def luminance(self, img):
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        return 0.2126*r + 0.7152*g + 0.0722*b

    def observation(self, raw_observation):
        recent_raw = self.raw_obs_history.recent(self.m)

        # Make sure there are enough frames (duplicate latest if not)
        latest = recent_raw[-1]
        while len(recent_raw) < self.m + 1:
            recent_raw.append(latest)

        # Calculate luminance and resize
        recent_frames = []
        for i in range(self.m):
            maxed = np.maximum(recent_raw[-(i+1)], recent_raw[-(i+2)])
            luma = self.luminance(maxed)
            resized = imresize(luma, self.size).astype('float32')
            recent_frames.append(resized)

        # Stack and normalize pixel values
        return np.stack(recent_frames) / 255.0


ATARI_NAMES = [
        'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids',
        'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Bowling',
        'Boxing', 'Breakout', 'Centipede', 'ChopperCommand', 'CrazyClimber',
        'DemonAttack', 'DoubleDunk', 'Enduro', 'FishingDerby', 'Freeway',
        'Frostbite', 'Gopher', 'Gravitar', 'IceHockey', 'Jamesbond',
        'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman',
        'NameThisGame', 'Pitfall', 'Pong', 'PrivateEye', 'Qbert',
        'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'SpaceInvaders',
        'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
        'Venture', 'VideoPinball', 'WizardOfWor', 'Zaxxon'
]

def get_gym_env(name):
    env = gym.make(name)
    basename = name.split('-')[0]
    return AtariWrapper(env) if name in ATARI_NAMES else env


def integral_dimensionality(space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Box):
        return np.prod(space.shape)
