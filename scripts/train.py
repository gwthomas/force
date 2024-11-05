import torch

from force.alg import NAMED_ALGORITHMS
from force.config import TaggedUnion, Field
from force.defaults import DEVICE
from force.experiment.rl import RLExperiment
from force.nn.util import torchify, numpyify

try:
    import d4rl
except:
    print('Failed to import D4RL')
    pass


class ExampleExperiment(RLExperiment):
    class Config(RLExperiment.Config):
        agent = TaggedUnion({
            name: cls.Config for name, cls in NAMED_ALGORITHMS.items()
        })

    def get_initial_data(self):
        # Get initial data from d4rl, if applicable
        if hasattr(self.env, 'get_dataset'):
            self.log('Loading initial data from d4rl...')
            initial_data = torchify(d4rl.qlearning_dataset(self.env), device=DEVICE)

            # TODO: get actual D4RL truncation flags
            initial_data['truncateds'] = torch.zeros_like(initial_data['terminals'])
            return initial_data
        else:
            return None

    def create_agent(self):
        cfg = self.cfg
        env_root = cfg.env.name.split('-')[0]
        agent_kwargs = {'device': DEVICE}

        # Algorithm-dependent settings
        if cfg.algorithm == 'MBPO':
            from force.env.mujoco.termination_functions import TERMINATION_FUNCTIONS
            agent_kwargs['termination_fn'] = TERMINATION_FUNCTIONS[env_root]
        if cfg.algorithm in {'MBPO', 'REDQ'}:
            from force.alg.mbpo import MUJOCO_TARGET_ENTROPIES
            if cfg.algorithm == 'REDQ':
                agent_cfg = cfg.agent
            else:
                agent_cfg = cfg.agent.solver
            agent_cfg.target_entropy = MUJOCO_TARGET_ENTROPIES[env_root]

        return NAMED_ALGORITHMS[cfg.algorithm](
            cfg.agent, self.observation_space, self.action_space,
            **agent_kwargs
        )


if __name__ == '__main__':
    ExampleExperiment.main()