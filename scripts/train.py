import torch

from force import defaults
from force.alg import BufferedRLAlgorithm, NAMED_ALGORITHMS
from force.config import TaggedUnion, Field
from force.env.util import get_gym_env, space_dim
from force.workflow import Experiment
from force.nn.util import torchify, numpyify
from force.policies import UniformPolicy
from force.sampling import SimpleSampler


class RLExperiment(Experiment):
    class Config(Experiment.Config):
        wrapper = BufferedRLAlgorithm
        agent = TaggedUnion({
            name: cls.Config() for name, cls in NAMED_ALGORITHMS.items()
        })
        initial_steps = Field(int, required=False)

    def setup(self, cfg):
        # Create env and get its info
        env_factory = lambda: get_gym_env(cfg.domain)
        env = env_factory()

        # Get initial data from d4rl or sampling from a random policy
        if hasattr(env, 'get_dataset'):
            import d4rl
            self.log('Loading initial data from d4rl...')
            initial_data = torchify(d4rl.qlearning_dataset(env))

            # TODO: get actual D4RL truncation flags
            initial_data['truncateds'] = torch.zeros_like(initial_data['terminals'])
        elif cfg.initial_steps is not None:
            self.log('Sampling initial data with uniform policy...')
            initial_data = SimpleSampler(env).run(
                UniformPolicy(env),
                num_steps=cfg.initial_steps
            )
        else:
            initial_data = None

        agent_kwargs = {'device': defaults.DEVICE}
        if cfg.algorithm == 'MBPO':
            from force.env.termination_functions import TERMINATION_FUNCTIONS
            domain_root = cfg.domain.split('-')[0]
            agent_kwargs['termination_fn'] = TERMINATION_FUNCTIONS[domain_root]
        agent = NAMED_ALGORITHMS[cfg.algorithm](
            cfg.agent, env.observation_space, env.action_space,
            **agent_kwargs
        )
        self.log(f'Agent: {agent}')
        return BufferedRLAlgorithm(
            cfg.wrapper, env_factory, agent,
            initial_data=initial_data
        )


if __name__ == '__main__':
    RLExperiment.main()