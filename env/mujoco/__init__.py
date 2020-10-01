from .ant import AntEnv
from .ant_angle import AntAngleEnv
from .half_cheetah import HalfCheetahEnv
from .half_cheetah_vel_jump import HalfCheetahVelJumpEnv
from .hopper import HopperEnv
from .humanoid import HumanoidEnv
from .swimmer import SwimmerEnv
from .walker2d import Walker2dEnv


MUJOCO_ENVS = {
    'ant': AntEnv,
    'ant-angle': AntAngleEnv,
    'halfcheetah': HalfCheetahEnv,
    'halfcheetah-jump': HalfCheetahVelJumpEnv,
    'hopper': HopperEnv,
    'humanoid': HumanoidEnv,
    'swimmer': SwimmerEnv,
    'walker2d': Walker2dEnv,
}