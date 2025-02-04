import torch

from force.alg import NAMED_ALGORITHMS
from force.config import TaggedUnion, Field
from force.experiment.rl import RLExperiment
from force.nn.util import torchify, numpyify

try:
    import d4rl
except:
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
            return super().get_initial_data()

    def create_agent(self):
        cfg = self.cfg
        env_root = cfg.env_name.split('-')[0]

        # Algorithm-dependent settings
        agent_kwargs = {'device': self.device}
        alg_name = cfg.agent._tag
        if alg_name == 'MBPO':
            from force.env.mujoco.termination_functions import TERMINATION_FUNCTIONS
            agent_kwargs['termination_fn'] = TERMINATION_FUNCTIONS[env_root]
        if alg_name in {'MBPO', 'REDQ'}:
            from force.alg.mbpo import MUJOCO_TARGET_ENTROPIES
            if alg_name == 'REDQ':
                agent_cfg = cfg.agent
            else:
                agent_cfg = cfg.agent.solver
            agent_cfg.target_entropy = MUJOCO_TARGET_ENTROPIES[env_root]

        alg_class = NAMED_ALGORITHMS[alg_name]
        return alg_class(cfg.agent, self.env.info, **agent_kwargs)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    ExampleExperiment.main()